"""Dataset and DataLoader for self-play training data."""

import struct
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from collections import deque
from pathlib import Path

INPUT_PLANES = 21
BOARD_SIZE = 8
INPUT_SIZE = INPUT_PLANES * BOARD_SIZE * BOARD_SIZE  # 1344
POLICY_SIZE = 1858


class ReplayBuffer:
    """Ring buffer storing training positions from self-play."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer: deque = deque(maxlen=capacity)

    def add(self, features: np.ndarray, policy_target: np.ndarray, value_target: float):
        """Add a single position to the buffer.

        Args:
            features: [21, 8, 8] float32 board features
            policy_target: [1858] float32 policy distribution (sparse OK, will be densified)
            value_target: float32 game outcome
        """
        self.buffer.append((features, policy_target, value_target))

    def add_batch(self, positions: list[tuple[np.ndarray, np.ndarray, float]]):
        """Add multiple positions."""
        self.buffer.extend(positions)

    def sample(self, batch_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Sample a random batch."""
        indices = np.random.randint(0, len(self.buffer), size=batch_size)
        features = np.stack([self.buffer[i][0] for i in indices])
        policies = np.stack([self.buffer[i][1] for i in indices])
        values = np.array([self.buffer[i][2] for i in indices], dtype=np.float32)
        return features, policies, values

    def __len__(self):
        return len(self.buffer)


class ChessDataset(Dataset):
    """PyTorch dataset wrapping the replay buffer."""

    def __init__(self, replay_buffer: ReplayBuffer):
        self.buffer = replay_buffer

    def __len__(self):
        return len(self.buffer)

    def __getitem__(self, idx):
        features, policy, value = self.buffer.buffer[idx]
        return (
            torch.from_numpy(features).float(),
            torch.from_numpy(policy).float(),
            torch.tensor(value, dtype=torch.float32),
        )


def deserialize_positions(data: bytes) -> list[tuple[np.ndarray, np.ndarray, float]]:
    """Deserialize training positions from the Rust binary format.

    Format:
        u32: num_positions
        For each position:
            f32 * 1344: features
            u16: num_moves
            (u16, f32) * num_moves: sparse policy
            f32: value_target
    """
    positions = []
    offset = 0

    num_pos = struct.unpack_from("<I", data, offset)[0]
    offset += 4

    for _ in range(num_pos):
        # Features
        features = np.frombuffer(data, dtype=np.float32, count=INPUT_SIZE, offset=offset)
        features = features.reshape(INPUT_PLANES, BOARD_SIZE, BOARD_SIZE).copy()
        offset += INPUT_SIZE * 4

        # Sparse policy
        num_moves = struct.unpack_from("<H", data, offset)[0]
        offset += 2

        policy = np.zeros(POLICY_SIZE, dtype=np.float32)
        for _ in range(num_moves):
            idx = struct.unpack_from("<H", data, offset)[0]
            offset += 2
            prob = struct.unpack_from("<f", data, offset)[0]
            offset += 4
            if idx < POLICY_SIZE:
                policy[idx] = prob

        # Normalize policy
        total = policy.sum()
        if total > 0:
            policy /= total

        # Value target
        value = struct.unpack_from("<f", data, offset)[0]
        offset += 4

        positions.append((features, policy, value))

    return positions


def create_dataloader(
    replay_buffer: ReplayBuffer,
    batch_size: int,
    num_workers: int = 0,
) -> DataLoader:
    """Create a DataLoader from a replay buffer."""
    dataset = ChessDataset(replay_buffer)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=True,
    )
