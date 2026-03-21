"""AlphaZero-style ResNet with dual heads and SupCon projection."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Pre-activation residual block: Conv -> BN -> ReLU -> Conv -> BN + skip -> ReLU."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = F.relu(out + residual)
        return out


class PolicyHead(nn.Module):
    """Policy head: Conv1x1 -> BN -> ReLU -> FC -> policy logits."""

    def __init__(self, in_channels: int, policy_channels: int, policy_size: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, policy_channels, 1, bias=True)
        self.bn = nn.BatchNorm2d(policy_channels)
        self.fc = nn.Linear(policy_channels * 64, policy_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn(self.conv(x)))
        out = out.view(out.size(0), -1)
        return self.fc(out)


class ValueHead(nn.Module):
    """Value head with branched output: scalar value + SupCon projection.

    Architecture:
        Conv1x1 -> BN -> ReLU -> FC(64, hidden) -> ReLU -> [branch]
        Branch 1: FC(hidden, 1) -> Tanh  (scalar value)
        Branch 2: Projection MLP -> L2Norm  (SupCon embeddings, training only)
    """

    def __init__(self, in_channels: int, hidden_dim: int, projection_dim: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 1, 1, bias=True)
        self.bn = nn.BatchNorm2d(1)
        self.fc1 = nn.Linear(64, hidden_dim)
        # Scalar value output
        self.fc_value = nn.Linear(hidden_dim, 1)
        # SupCon projection head (discarded at inference)
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (value, value_embedding, projected_embedding)."""
        out = F.relu(self.bn(self.conv(x)))
        out = out.view(out.size(0), -1)
        embedding = F.relu(self.fc1(out))

        value = torch.tanh(self.fc_value(embedding))
        projected = F.normalize(self.projection(embedding), dim=1)

        return value, embedding, projected


class ChessNet(nn.Module):
    """AlphaZero-style network for chess.

    Input: [batch, 21, 8, 8] board features
    Output: (policy_logits, value, value_embedding, supcon_projection)
    """

    def __init__(
        self,
        input_planes: int = 21,
        num_filters: int = 128,
        num_blocks: int = 8,
        policy_channels: int = 32,
        policy_size: int = 1858,
        value_hidden_dim: int = 128,
        projection_dim: int = 128,
    ):
        super().__init__()
        self.num_filters = num_filters
        self.num_blocks = num_blocks

        # Stem
        self.stem_conv = nn.Conv2d(input_planes, num_filters, 3, padding=1, bias=True)
        self.stem_bn = nn.BatchNorm2d(num_filters)

        # Residual tower
        self.residual_tower = nn.Sequential(
            *[ResidualBlock(num_filters) for _ in range(num_blocks)]
        )

        # Heads
        self.policy_head = PolicyHead(num_filters, policy_channels, policy_size)
        self.value_head = ValueHead(num_filters, value_hidden_dim, projection_dim)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: Board features [batch, 21, 8, 8]

        Returns:
            policy_logits: [batch, 1858]
            value: [batch, 1] in [-1, 1]
            value_embedding: [batch, hidden_dim] (pre-projection)
            supcon_projection: [batch, projection_dim] (L2-normalized)
        """
        # Stem
        out = F.relu(self.stem_bn(self.stem_conv(x)))
        # Residual tower
        out = self.residual_tower(out)
        # Heads
        policy_logits = self.policy_head(out)
        value, value_embedding, supcon_projection = self.value_head(out)

        return policy_logits, value, value_embedding, supcon_projection

    @classmethod
    def from_config(cls, cfg) -> "ChessNet":
        """Create network from a NetworkConfig."""
        return cls(
            input_planes=cfg.input_planes,
            num_filters=cfg.num_filters,
            num_blocks=cfg.num_blocks,
            policy_channels=cfg.policy_channels,
            policy_size=cfg.policy_size,
            value_hidden_dim=cfg.value_hidden_dim,
            projection_dim=cfg.projection_dim,
        )
