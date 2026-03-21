"""Training loop with hybrid RL + SupCon loss."""

import time
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.optim as optim

from training.model import ChessNet
from training.data import ReplayBuffer
from training.supcon import SupConLoss, discretize_value, prepare_supcon_features, filter_small_bins
from training.schedule import HybridSchedule
from training.config import Config


class Trainer:
    """Hybrid RL + SupCon trainer for the chess network."""

    def __init__(self, config: Config, model: ChessNet, device: torch.device | None = None):
        self.config = config
        self.device = device or torch.device("cpu")
        self.model = model.to(self.device)

        # Optimizer
        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=config.training.learning_rate,
            momentum=0.9,
            weight_decay=config.training.weight_decay,
        )

        # LR scheduler
        self.lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.training.training_steps,
            eta_min=1e-5,
        )

        # SupCon loss
        self.supcon_criterion = SupConLoss(
            temperature=config.supcon.temperature,
            contrast_mode=config.supcon.contrast_mode,
            base_temperature=config.supcon.base_temperature,
        )

        # Hybrid schedule
        self.schedule = HybridSchedule(
            total_steps=config.training.training_steps,
            phase1_end=config.supcon.schedule.phase1_end,
            phase2_end=config.supcon.schedule.phase2_end,
            alpha_start=config.supcon.schedule.alpha_start,
            alpha_end=config.supcon.schedule.alpha_end,
            beta_start=config.supcon.schedule.beta_start,
            beta_end=config.supcon.schedule.beta_end,
        )

        self.global_step = 0

    def train_step(
        self,
        features: torch.Tensor,
        policy_target: torch.Tensor,
        value_target: torch.Tensor,
    ) -> dict[str, float]:
        """Execute a single training step.

        Args:
            features: [batch, 21, 8, 8] board features
            policy_target: [batch, 1858] MCTS policy distribution
            value_target: [batch] game outcome in [-1, 1]

        Returns:
            Dict of loss components for logging.
        """
        self.model.train()
        self.optimizer.zero_grad()

        # Forward pass
        policy_logits, value_pred, value_embedding, supcon_proj = self.model(features)

        # --- RL losses ---
        # Policy: cross-entropy with MCTS visit distribution
        log_probs = F.log_softmax(policy_logits, dim=1)
        policy_loss = -(policy_target * log_probs).sum(dim=1).mean()

        # Value: MSE
        value_loss = F.mse_loss(value_pred.squeeze(-1), value_target)

        rl_loss = (
            self.config.training.loss.policy_weight * policy_loss
            + self.config.training.loss.value_weight * value_loss
        )

        # --- SupCon loss ---
        supcon_loss = torch.tensor(0.0, device=self.device)

        if self.config.supcon.enabled:
            # Discretize value targets into bins
            value_bins = discretize_value(
                value_target,
                num_bins=self.config.supcon.num_bins,
                strategy=self.config.supcon.bin_strategy,
            )

            # Filter out bins with too few samples
            valid_mask = filter_small_bins(value_bins, self.config.supcon.min_bin_size)

            if valid_mask.sum() >= 2 * self.config.supcon.min_bin_size:
                valid_proj = supcon_proj[valid_mask]
                valid_bins = value_bins[valid_mask]

                # Create two views for SupCon
                supcon_features = prepare_supcon_features(
                    valid_proj, noise_std=self.config.supcon.noise_std
                )

                supcon_loss = self.supcon_criterion(supcon_features, labels=valid_bins)

        # --- Hybrid loss ---
        alpha, beta = self.schedule.get_weights(self.global_step)
        total_loss = alpha * rl_loss + beta * supcon_loss

        # Backward pass
        total_loss.backward()
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        self.lr_scheduler.step()

        self.global_step += 1

        return {
            "total_loss": total_loss.item(),
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "supcon_loss": supcon_loss.item(),
            "alpha": alpha,
            "beta": beta,
            "lr": self.optimizer.param_groups[0]["lr"],
        }

    def train_epoch(
        self,
        replay_buffer: ReplayBuffer,
        steps_per_epoch: int,
        step_callback=None,
    ) -> dict[str, float]:
        """Run multiple training steps from the replay buffer.

        Args:
            step_callback: Optional callable(step_in_epoch, metrics_dict) called after each step.

        Returns:
            Averaged metrics over the epoch.
        """
        metrics_sum: dict[str, float] = {}
        count = 0
        batch_size = self.config.training.batch_size

        if len(replay_buffer) < batch_size:
            return {}

        for step_i in range(steps_per_epoch):
            features_np, policy_np, value_np = replay_buffer.sample(batch_size)

            features = torch.from_numpy(features_np).float().to(self.device)
            policy_target = torch.from_numpy(policy_np).float().to(self.device)
            value_target = torch.from_numpy(value_np).float().to(self.device)

            metrics = self.train_step(features, policy_target, value_target)

            for k, v in metrics.items():
                metrics_sum[k] = metrics_sum.get(k, 0.0) + v
            count += 1

            if step_callback is not None:
                step_callback(step_i, metrics)

        return {k: v / count for k, v in metrics_sum.items()} if count > 0 else {}

    def save_checkpoint(self, path: str | Path):
        """Save model checkpoint."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "lr_scheduler_state_dict": self.lr_scheduler.state_dict(),
                "global_step": self.global_step,
            },
            path,
        )

    def load_checkpoint(self, path: str | Path):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])
        self.global_step = checkpoint["global_step"]
