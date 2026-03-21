#!/usr/bin/env python3
"""Main training script: self-play + training loop with live dashboard."""

import argparse
import math
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
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
from rich.table import Table
from rich.text import Text

from training.config import load_config
from training.model import ChessNet
from training.data import ReplayBuffer, deserialize_positions
from training.trainer import Trainer
from training.export import export_weights

console = Console()


# ── Sparkline helper ──────────────────────────────────────────────────
def sparkline(values: list[float], width: int = 40) -> str:
    """Render a list of floats as a unicode sparkline."""
    if not values:
        return ""
    blocks = " ▁▂▃▄▅▆▇█"
    lo, hi = min(values), max(values)
    rng = hi - lo if hi - lo > 1e-8 else 1.0
    # Take last `width` values
    tail = values[-width:]
    return "".join(blocks[min(8, int((v - lo) / rng * 8))] for v in tail)


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


def build_loss_panel(hist: MetricsHistory) -> Panel:
    table = Table(show_header=True, header_style="bold", box=None, padding=(0, 1))
    table.add_column("Loss", style="bold", width=8)
    table.add_column("Current", justify="right", width=9)
    table.add_column("Min", justify="right", width=9, style="green")
    table.add_column("Trend (last 40 steps)", width=42)

    for name, series, color in [
        ("Total", hist.total, "bright_white"),
        ("Policy", hist.policy, "bright_blue"),
        ("Value", hist.value, "bright_green"),
        ("SupCon", hist.supcon, "bright_magenta"),
    ]:
        vals = list(series)
        cur = f"{vals[-1]:.4f}" if vals else "—"
        mn = f"{min(vals):.4f}" if vals else "—"
        spark = sparkline(vals)
        table.add_row(
            f"[{color}]{name}[/]",
            f"[{color}]{cur}[/]",
            mn,
            f"[{color}]{spark}[/]",
        )

    return Panel(table, title="[bold]Loss Curves[/]", border_style="blue")


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
) -> Panel:
    fill_pct = buffer_size / max(buffer_cap, 1) * 100
    bar_w = 20
    filled = int(fill_pct / 100 * bar_w)
    buf_bar = f"[green]{'█' * filled}[/]{'░' * (bar_w - filled)}"

    w = outcomes.get("white", 0)
    b = outcomes.get("black", 0)
    d = outcomes.get("draw", 0)
    total_games_out = w + b + d or 1

    lines = [
        f"  Games played      [bold]{games_total:,}[/]",
        f"  Positions total   [bold]{positions_total:,}[/]",
        f"  Buffer            {buf_bar} {fill_pct:.0f}%",
        f"  ({buffer_size:,}/{buffer_cap:,})",
        f"  Last batch time   {last_game_time:.1f}s",
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
        Layout(name="body"),
    )
    layout["body"].split_row(
        Layout(name="left", ratio=3),
        Layout(name="right", ratio=2),
    )
    layout["left"].split_column(
        Layout(name="losses"),
        Layout(name="schedule", size=12),
    )

    layout["header"].update(build_header(config, param_count, cycle, config.training.cycles))
    layout["losses"].update(build_loss_panel(hist))
    layout["schedule"].update(build_schedule_panel(hist, trainer.global_step, config.training.training_steps))
    layout["right"].update(build_selfplay_panel(**sp_stats))
    return layout


# ── Self-play ─────────────────────────────────────────────────────────
def generate_random_positions(num_positions: int, replay_buffer: ReplayBuffer):
    for _ in range(num_positions):
        features = np.random.randn(21, 8, 8).astype(np.float32) * 0.1
        policy = np.random.dirichlet(np.ones(1858)).astype(np.float32)
        value = np.random.choice([-1.0, 0.0, 1.0])
        replay_buffer.add(features, policy, value)


def selfplay_with_rust(config, weights_path: Path, num_games: int):
    try:
        import chess_engine_py as engine
        data = engine.run_selfplay(
            str(weights_path),
            num_games=num_games,
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
                export_weights(model, weights_path, fuse_batchnorm=True)

                sp_t0 = time.time()
                if args.random_data:
                    n_pos = config.training.selfplay_games_per_cycle * 80
                    generate_random_positions(n_pos, replay_buffer)
                    sp_stats["games_total"] += config.training.selfplay_games_per_cycle
                    sp_stats["positions_total"] += n_pos
                else:
                    positions = selfplay_with_rust(
                        config, weights_path, config.training.selfplay_games_per_cycle
                    )
                    if positions is None:
                        n_pos = config.training.selfplay_games_per_cycle * 80
                        generate_random_positions(n_pos, replay_buffer)
                        sp_stats["games_total"] += config.training.selfplay_games_per_cycle
                        sp_stats["positions_total"] += n_pos
                    else:
                        replay_buffer.add_batch(positions)
                        sp_stats["games_total"] += config.training.selfplay_games_per_cycle
                        sp_stats["positions_total"] += len(positions)
                        # Count outcomes from values
                        for _, _, v in positions:
                            if v > 0.5:
                                sp_stats["outcomes"]["white"] += 1
                            elif v < -0.5:
                                sp_stats["outcomes"]["black"] += 1
                            else:
                                sp_stats["outcomes"]["draw"] += 1

                sp_stats["last_game_time"] = time.time() - sp_t0
                sp_stats["buffer_size"] = len(replay_buffer)

            if args.selfplay_only:
                live.update(build_dashboard(config, param_count, cycle + 1, hist, trainer, sp_stats))
                continue

            if len(replay_buffer) < config.training.batch_size:
                continue

            # ── Training ──
            steps_per_cycle = config.training.training_steps // config.training.cycles

            def on_step(step_i, metrics):
                hist.record(metrics)
                if step_i % 5 == 0:  # Update dashboard every 5 steps
                    sp_stats["buffer_size"] = len(replay_buffer)
                    live.update(build_dashboard(
                        config, param_count, cycle + 1, hist, trainer, sp_stats
                    ))

            trainer.train_epoch(replay_buffer, steps_per_cycle, step_callback=on_step)

            # Final dashboard update for this cycle
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
