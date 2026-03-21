#!/usr/bin/env python3
"""Main training script: self-play + training loop with live dashboard."""

import argparse
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table

from training.config import load_config
from training.model import ChessNet
from training.data import ReplayBuffer, deserialize_positions
from training.trainer import Trainer
from training.export import export_weights

console = Console()


# ── Chart helpers ────────────────────────────────────────────────────
def _ema(values: list[float], alpha: float = 0.15) -> list[float]:
    """Exponential moving average for smoothing."""
    if not values:
        return []
    out = [values[0]]
    for v in values[1:]:
        out.append(alpha * v + (1 - alpha) * out[-1])
    return out


def mini_chart(
    values: list[float],
    width: int = 50,
    height: int = 7,
    color: str = "bright_white",
    show_smooth: bool = True,
) -> str:
    """Render a multi-row braille-style area chart with optional EMA overlay.

    Returns a string of `height` lines, each `width` characters wide.
    Y-axis labels on the left, latest value + delta on the right.
    """
    if len(values) < 2:
        return "\n".join([" " * (width + 16)] * height)

    # Take the last `width` data points
    tail = values[-width:]
    smooth = _ema(tail, alpha=0.12) if show_smooth else tail

    lo = min(min(tail), min(smooth))
    hi = max(max(tail), max(smooth))
    rng = hi - lo if hi - lo > 1e-9 else 1.0
    # Pad range slightly so points aren't clipped at edges
    lo -= rng * 0.05
    hi += rng * 0.05
    rng = hi - lo

    # Braille block characters for the area fill (bottom to top density)
    fill_chars = " ░▒▓█"

    rows: list[str] = []
    for row in range(height):
        # Row 0 = top of chart, row height-1 = bottom
        row_top = hi - (row / height) * rng
        row_bot = hi - ((row + 1) / height) * rng

        # Y-axis label: show on top, middle, bottom rows
        if row == 0:
            y_label = f"{hi:>7.3f} │"
        elif row == height - 1:
            y_label = f"{lo:>7.3f} │"
        elif row == height // 2:
            mid = (hi + lo) / 2
            y_label = f"{mid:>7.3f} │"
        else:
            y_label = "        │"

        line = ""
        for i, (raw_v, sm_v) in enumerate(zip(tail, smooth)):
            # How much of this cell is filled by the raw value
            if raw_v >= row_top:
                fill = 4  # full
            elif raw_v <= row_bot:
                fill = 0  # empty
            else:
                fill = int(((raw_v - row_bot) / (row_top - row_bot)) * 4)
                fill = max(0, min(4, fill))

            # Check if the smooth line passes through this cell
            is_smooth = row_bot <= sm_v <= row_top

            if is_smooth and show_smooth:
                line += "─"
            else:
                line += fill_chars[fill]

        # Pad if tail < width
        line += " " * (width - len(tail))

        # Right-side annotation on specific rows
        if row == 0:
            annotation = f"  [{color}]{tail[-1]:>8.4f}[/] now"
        elif row == 1 and len(values) > 10:
            delta = tail[-1] - tail[0]
            arrow = "↓" if delta < 0 else "↑" if delta > 0 else "→"
            d_color = "green" if delta < 0 else "red" if delta > 0 else "dim"
            annotation = f"  [{d_color}]{arrow} {abs(delta):.4f}[/]"
        elif row == 2:
            annotation = f"  [green]{min(tail):>8.4f}[/] min"
        else:
            annotation = ""

        rows.append(f"[dim]{y_label}[/][{color}]{line}[/]{annotation}")

    return "\n".join(rows)


def phase_label(alpha: float, beta: float) -> tuple[str, str]:
    """Return (phase name, color) based on current schedule weights."""
    if alpha >= 0.95:
        return "Phase 1: RL-dominant", "bright_blue"
    elif alpha <= 0.35:
        return "Phase 3: SupCon-dominant", "bright_magenta"
    else:
        return "Phase 2: Transition", "bright_yellow"


# ── Loss history tracker ──────────────────────────────────────────────
class MetricsHistory:
    def __init__(self, max_len: int = 500):
        self.total: deque[float] = deque(maxlen=max_len)
        self.policy: deque[float] = deque(maxlen=max_len)
        self.value: deque[float] = deque(maxlen=max_len)
        self.supcon: deque[float] = deque(maxlen=max_len)
        self.alpha: deque[float] = deque(maxlen=max_len)
        self.beta: deque[float] = deque(maxlen=max_len)
        self.lr: deque[float] = deque(maxlen=max_len)

    def record(self, m: dict):
        self.total.append(m.get("total_loss", 0))
        self.policy.append(m.get("policy_loss", 0))
        self.value.append(m.get("value_loss", 0))
        self.supcon.append(m.get("supcon_loss", 0))
        self.alpha.append(m.get("alpha", 0))
        self.beta.append(m.get("beta", 0))
        self.lr.append(m.get("lr", 0))


# ── Dashboard panels ─────────────────────────────────────────────────
def build_header(config, param_count: int, cycle: int, total_cycles: int) -> Panel:
    grid = Table.grid(padding=(0, 2))
    grid.add_column(justify="left")
    grid.add_column(justify="left")
    grid.add_column(justify="right")
    grid.add_row(
        f"[bold cyan]AlphaZero + SupCon Chess Engine[/]",
        f"[dim]{config.network.num_blocks}x{config.network.num_filters} "
        f"({param_count:,} params)[/]",
        f"[bold]Cycle {cycle}/{total_cycles}[/]",
    )
    return Panel(grid, border_style="bright_cyan")


def build_loss_panels(hist: MetricsHistory) -> Layout:
    """Build a 2x2 grid of loss charts, each with its own Y-scale."""
    charts = Layout()
    charts.split_column(
        Layout(name="top_row", size=12),
        Layout(name="bot_row", size=12),
    )
    charts["top_row"].split_row(
        Layout(name="total"),
        Layout(name="policy"),
    )
    charts["bot_row"].split_row(
        Layout(name="value"),
        Layout(name="supcon"),
    )

    for name, series, color, slot in [
        ("Total Loss", hist.total, "bright_white", "total"),
        ("Policy Loss", hist.policy, "bright_blue", "policy"),
        ("Value Loss", hist.value, "bright_green", "value"),
        ("SupCon Loss", hist.supcon, "bright_magenta", "supcon"),
    ]:
        vals = list(series)
        chart_str = mini_chart(vals, width=45, height=8, color=color)
        border = {"bright_white": "white", "bright_blue": "blue",
                  "bright_green": "green", "bright_magenta": "magenta"}[color]
        charts[slot].update(Panel(
            chart_str,
            title=f"[bold {color}]{name}[/]",
            border_style=border,
            padding=(0, 1),
        ))

    return charts


def build_schedule_panel(hist: MetricsHistory, global_step: int, total_steps: int) -> Panel:
    alpha = list(hist.alpha)[-1] if hist.alpha else 0.95
    beta = list(hist.beta)[-1] if hist.beta else 0.05
    lr = list(hist.lr)[-1] if hist.lr else 0.02
    phase_name, phase_color = phase_label(alpha, beta)

    progress_frac = global_step / max(total_steps, 1)
    bar_width = 36
    filled = int(progress_frac * bar_width)
    bar = "█" * filled + "░" * (bar_width - filled)

    lines = [
        f"[{phase_color} bold]{phase_name}[/]",
        "",
        f"  α (RL weight)     [bright_blue]{'█' * int(alpha * 20)}{'░' * (20 - int(alpha * 20))}[/] {alpha:.3f}",
        f"  β (SupCon weight) [bright_magenta]{'█' * int(beta * 20)}{'░' * (20 - int(beta * 20))}[/] {beta:.3f}",
        "",
        f"  Learning rate     {lr:.6f}",
        f"  Global step       {global_step:,} / {total_steps:,}",
        f"  Progress          [{phase_color}]{bar}[/] {progress_frac * 100:.1f}%",
    ]
    return Panel("\n".join(lines), title="[bold]Schedule[/]", border_style="yellow")


def build_selfplay_panel(
    games_total: int,
    positions_total: int,
    buffer_size: int,
    buffer_cap: int,
    last_game_time: float,
    outcomes: dict,
    status: str = "",
    games_per_sec: float = 0.0,
) -> Panel:
    fill_pct = buffer_size / max(buffer_cap, 1) * 100
    bar_w = 20
    filled = int(fill_pct / 100 * bar_w)
    buf_bar = f"[green]{'█' * filled}[/]{'░' * (bar_w - filled)}"

    w = outcomes.get("white", 0)
    b = outcomes.get("black", 0)
    d = outcomes.get("draw", 0)
    total_games_out = w + b + d or 1

    lines = []
    if status:
        lines.append(f"  [bold yellow]{status}[/]")
        lines.append("")

    lines += [
        f"  Games played      [bold]{games_total:,}[/]",
        f"  Positions total   [bold]{positions_total:,}[/]",
        f"  Buffer            {buf_bar} {fill_pct:.0f}%",
        f"  ({buffer_size:,}/{buffer_cap:,})",
    ]
    if games_per_sec > 0:
        lines.append(f"  Speed             [bold]{games_per_sec:.1f}[/] games/s")
    if last_game_time > 0:
        lines.append(f"  Last batch time   {last_game_time:.1f}s")
    lines += [
        "",
        f"  Outcomes  [bright_white]W {w}[/] | [dim]D {d}[/] | [bright_red]L {b}[/]"
        f"  ({w / total_games_out * 100:.0f}% / {d / total_games_out * 100:.0f}% / {b / total_games_out * 100:.0f}%)",
    ]
    return Panel("\n".join(lines), title="[bold]Self-Play[/]", border_style="green")


def build_dashboard(
    config, param_count, cycle, hist, trainer, sp_stats
) -> Layout:
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="body", size=24),
        Layout(name="footer", size=14),
    )
    # Body: 2x2 loss charts
    layout["body"].update(build_loss_panels(hist))
    # Footer: schedule + self-play side by side
    layout["footer"].split_row(
        Layout(name="schedule", ratio=3),
        Layout(name="selfplay", ratio=2),
    )

    layout["header"].update(build_header(config, param_count, cycle, config.training.cycles))
    layout["footer"]["schedule"].update(
        build_schedule_panel(hist, trainer.global_step, config.training.training_steps)
    )
    layout["footer"]["selfplay"].update(build_selfplay_panel(**sp_stats))
    return layout


# ── Self-play ─────────────────────────────────────────────────────────
def generate_random_positions(num_positions: int, replay_buffer: ReplayBuffer):
    for _ in range(num_positions):
        features = np.random.randn(21, 8, 8).astype(np.float32) * 0.1
        policy = np.random.dirichlet(np.ones(1858)).astype(np.float32)
        value = np.random.choice([-1.0, 0.0, 1.0])
        replay_buffer.add(features, policy, value)


def selfplay_batch(config, weights_path: Path, batch_size: int):
    """Run a small batch of self-play games. Returns positions or None if engine unavailable."""
    try:
        import chess_engine_py as engine
        data = engine.run_selfplay(
            str(weights_path),
            num_games=batch_size,
            simulations=config.engine.mcts_simulations,
            c_puct=config.engine.c_puct,
            dirichlet_alpha=config.engine.dirichlet_alpha,
            dirichlet_epsilon=config.engine.dirichlet_epsilon,
            temperature=config.engine.temperature,
        )
        return deserialize_positions(data)
    except ImportError:
        return None


# ── Main ──────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Train chess engine")
    parser.add_argument("--config", type=str, default="config/default.toml")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="output")
    parser.add_argument("--selfplay-only", action="store_true")
    parser.add_argument("--train-only", action="store_true")
    parser.add_argument("--random-data", action="store_true", help="Use random data (for pipeline testing)")
    args = parser.parse_args()

    config = load_config(args.config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    weights_dir = output_dir / "weights"
    weights_dir.mkdir(exist_ok=True)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    model = ChessNet.from_config(config.network)
    param_count = sum(p.numel() for p in model.parameters())
    trainer = Trainer(config, model, device)

    if args.checkpoint:
        trainer.load_checkpoint(args.checkpoint)

    replay_buffer = ReplayBuffer(config.training.replay_buffer_size)
    hist = MetricsHistory()

    # Self-play stats
    sp_stats = dict(
        games_total=0,
        positions_total=0,
        buffer_size=0,
        buffer_cap=config.training.replay_buffer_size,
        last_game_time=0.0,
        outcomes={"white": 0, "black": 0, "draw": 0},
        status="",
        games_per_sec=0.0,
    )

    console.print()
    console.print("[bold bright_cyan]  AlphaZero + SupCon Chess Engine Trainer[/]")
    console.print(f"  Model: {config.network.num_blocks}x{config.network.num_filters} "
                  f"({param_count:,} params) | Device: {device}")
    if args.checkpoint:
        console.print(f"  Resumed from step {trainer.global_step:,}")
    console.print()

    with Live(console=console, refresh_per_second=4, screen=False) as live:
        for cycle in range(config.training.cycles):
            # ── Self-play ──
            if not args.train_only:
                weights_path = weights_dir / f"weights_{cycle:04d}.bin"
                sp_stats["status"] = "Exporting weights..."
                live.update(build_dashboard(config, param_count, cycle + 1, hist, trainer, sp_stats))
                export_weights(model, weights_path, fuse_batchnorm=True)

                total_games = config.training.selfplay_games_per_cycle
                sp_batch = max(1, min(8, total_games // 4))  # small batches for progress
                games_done = 0
                sp_t0 = time.time()
                use_engine = not args.random_data

                while games_done < total_games:
                    batch_n = min(sp_batch, total_games - games_done)
                    elapsed = time.time() - sp_t0
                    gps = games_done / elapsed if elapsed > 0.1 else 0.0
                    sp_stats["status"] = (
                        f"Self-play: {games_done}/{total_games} games"
                        f" ({gps:.1f} g/s)" if gps > 0 else
                        f"Self-play: {games_done}/{total_games} games"
                    )
                    sp_stats["games_per_sec"] = gps
                    live.update(build_dashboard(config, param_count, cycle + 1, hist, trainer, sp_stats))

                    if use_engine:
                        positions = selfplay_batch(config, weights_path, batch_n)
                        if positions is None:
                            # Engine not available, fall back to random
                            use_engine = False
                            continue
                        replay_buffer.add_batch(positions)
                        sp_stats["positions_total"] += len(positions)
                        for _, _, v in positions:
                            if v > 0.5:
                                sp_stats["outcomes"]["white"] += 1
                            elif v < -0.5:
                                sp_stats["outcomes"]["black"] += 1
                            else:
                                sp_stats["outcomes"]["draw"] += 1
                    else:
                        n_pos = batch_n * 80
                        generate_random_positions(n_pos, replay_buffer)
                        sp_stats["positions_total"] += n_pos

                    games_done += batch_n
                    sp_stats["games_total"] += batch_n

                sp_stats["last_game_time"] = time.time() - sp_t0
                sp_stats["buffer_size"] = len(replay_buffer)
                sp_stats["games_per_sec"] = total_games / max(time.time() - sp_t0, 0.01)
                sp_stats["status"] = ""

            if args.selfplay_only:
                live.update(build_dashboard(config, param_count, cycle + 1, hist, trainer, sp_stats))
                continue

            if len(replay_buffer) < config.training.batch_size:
                continue

            # ── Training ──
            steps_per_cycle = config.training.training_steps // config.training.cycles
            sp_stats["status"] = f"Training: 0/{steps_per_cycle} steps"
            live.update(build_dashboard(config, param_count, cycle + 1, hist, trainer, sp_stats))

            def on_step(step_i, metrics):
                hist.record(metrics)
                if step_i % 5 == 0:
                    sp_stats["status"] = f"Training: {step_i + 1}/{steps_per_cycle} steps"
                    sp_stats["buffer_size"] = len(replay_buffer)
                    live.update(build_dashboard(
                        config, param_count, cycle + 1, hist, trainer, sp_stats
                    ))

            trainer.train_epoch(replay_buffer, steps_per_cycle, step_callback=on_step)

            sp_stats["status"] = ""
            live.update(build_dashboard(config, param_count, cycle + 1, hist, trainer, sp_stats))

            # Checkpoint
            if (cycle + 1) % 10 == 0:
                cp_path = checkpoint_dir / f"checkpoint_{trainer.global_step:07d}.pt"
                trainer.save_checkpoint(cp_path)

    # Final save
    final_weights = weights_dir / "weights_final.bin"
    export_weights(model, final_weights, fuse_batchnorm=True)
    trainer.save_checkpoint(checkpoint_dir / "checkpoint_final.pt")

    console.print()
    console.print(f"[bold green]Training complete.[/] Final weights: {final_weights}")
    console.print(f"Total steps: {trainer.global_step:,}")


if __name__ == "__main__":
    main()
