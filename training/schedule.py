"""Loss weight scheduling for hybrid RL + SupCon training."""

import math


class HybridSchedule:
    """Three-phase cosine annealing schedule for RL/SupCon loss weighting.

    Phase 1 (0 -> phase1_end):     Constant RL-dominant weights
    Phase 2 (phase1_end -> phase2_end): Cosine annealing from RL to SupCon
    Phase 3 (phase2_end -> 1.0):   Constant SupCon-dominant weights
    """

    def __init__(
        self,
        total_steps: int,
        phase1_end: float = 0.2,
        phase2_end: float = 0.7,
        alpha_start: float = 0.95,
        alpha_end: float = 0.3,
        beta_start: float = 0.05,
        beta_end: float = 0.7,
    ):
        self.total_steps = total_steps
        self.phase1_end = phase1_end
        self.phase2_end = phase2_end
        self.alpha_start = alpha_start
        self.alpha_end = alpha_end
        self.beta_start = beta_start
        self.beta_end = beta_end

        # Compute step boundaries
        self.step1 = int(total_steps * phase1_end)
        self.step2 = int(total_steps * phase2_end)

    def get_weights(self, step: int) -> tuple[float, float]:
        """Get (alpha, beta) weights for the given training step.

        Args:
            step: Current training step

        Returns:
            (alpha, beta) — RL weight and SupCon weight
        """
        if step <= self.step1:
            # Phase 1: constant RL-dominant
            return self.alpha_start, self.beta_start

        if step >= self.step2:
            # Phase 3: constant SupCon-dominant
            return self.alpha_end, self.beta_end

        # Phase 2: cosine annealing
        progress = (step - self.step1) / (self.step2 - self.step1)
        # Cosine decay from start to end
        cos_factor = 0.5 * (1.0 + math.cos(math.pi * progress))

        alpha = self.alpha_end + (self.alpha_start - self.alpha_end) * cos_factor
        beta = self.beta_end + (self.beta_start - self.beta_end) * cos_factor

        return alpha, beta

    def get_rl_weight(self, step: int) -> float:
        return self.get_weights(step)[0]

    def get_supcon_weight(self, step: int) -> float:
        return self.get_weights(step)[1]
