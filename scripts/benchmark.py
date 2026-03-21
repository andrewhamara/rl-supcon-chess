#!/usr/bin/env python3
"""Benchmark neural network inference and MCTS search speed."""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
from training.model import ChessNet
from training.config import load_config
from training.export import export_weights


def benchmark_pytorch_inference(model, num_iters=100):
    """Benchmark PyTorch CPU inference."""
    model.eval()
    x = torch.randn(1, 21, 8, 8)

    # Warmup
    for _ in range(10):
        with torch.no_grad():
            model(x)

    t0 = time.time()
    for _ in range(num_iters):
        with torch.no_grad():
            model(x)
    elapsed = time.time() - t0

    us_per_inference = elapsed / num_iters * 1_000_000
    print(f"PyTorch CPU inference: {us_per_inference:.0f} us/position ({num_iters/elapsed:.0f} pos/sec)")

    # Batch inference
    x_batch = torch.randn(8, 21, 8, 8)
    t0 = time.time()
    for _ in range(num_iters):
        with torch.no_grad():
            model(x_batch)
    elapsed = time.time() - t0
    us_per_pos_batch = elapsed / (num_iters * 8) * 1_000_000
    print(f"PyTorch CPU batch-8:  {us_per_pos_batch:.0f} us/position ({num_iters * 8 / elapsed:.0f} pos/sec)")


def benchmark_rust_inference():
    """Benchmark Rust int8 inference via PyO3."""
    try:
        import chess_engine_py as engine
        print(f"Rust engine version: {engine.version()}")

        # Quick self-play to test throughput
        t0 = time.time()
        data = engine.run_selfplay_random(num_games=5, simulations=50)
        elapsed = time.time() - t0
        print(f"Rust self-play (5 games, 50 sims): {elapsed:.2f}s")

    except ImportError:
        print("chess_engine_py not available — build with: maturin develop --release")


def main():
    config = load_config("config/default.toml")
    model = ChessNet.from_config(config.network)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model: {config.network.num_blocks}x{config.network.num_filters} ({param_count:,} params)")
    print()

    benchmark_pytorch_inference(model)
    print()
    benchmark_rust_inference()


if __name__ == "__main__":
    main()
