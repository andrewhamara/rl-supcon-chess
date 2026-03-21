"""Configuration loading from TOML."""

from dataclasses import dataclass, field
from pathlib import Path

try:
    import tomllib
except ImportError:
    import tomli as tomllib


@dataclass
class EngineConfig:
    num_threads: int = 8
    mcts_simulations: int = 800
    c_puct: float = 2.5
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25
    temperature: float = 1.0
    temperature_threshold_move: int = 30
    batch_size_inference: int = 8
    transposition_table_mb: int = 128


@dataclass
class NetworkConfig:
    num_blocks: int = 8
    num_filters: int = 128
    policy_channels: int = 32
    value_hidden_dim: int = 128
    projection_dim: int = 128
    input_planes: int = 21
    policy_size: int = 1858


@dataclass
class LossConfig:
    policy_weight: float = 1.0
    value_weight: float = 1.0


@dataclass
class TrainingConfig:
    learning_rate: float = 0.02
    lr_schedule: str = "cosine"
    weight_decay: float = 1e-4
    batch_size: int = 512
    replay_buffer_size: int = 1_000_000
    training_steps: int = 500_000
    checkpoint_interval: int = 5000
    selfplay_games_per_cycle: int = 256
    cycles: int = 200
    loss: LossConfig = field(default_factory=LossConfig)


@dataclass
class SupConScheduleConfig:
    phase1_end: float = 0.2
    phase2_end: float = 0.7
    alpha_start: float = 0.99
    alpha_end: float = 0.3
    beta_start: float = 0.01
    beta_end: float = 0.7


@dataclass
class SupConConfig:
    enabled: bool = True
    temperature: float = 0.07
    base_temperature: float = 0.07
    contrast_mode: str = "all"
    num_bins: int = 12
    bin_strategy: str = "equal_width"
    min_bin_size: int = 4
    noise_std: float = 0.01
    projection_layers: int = 2
    schedule: SupConScheduleConfig = field(default_factory=SupConScheduleConfig)


@dataclass
class QuantizationConfig:
    enabled: bool = True
    scheme: str = "int8_symmetric"
    calibration_positions: int = 10000


@dataclass
class Config:
    engine: EngineConfig = field(default_factory=EngineConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    supcon: SupConConfig = field(default_factory=SupConConfig)
    quantization: QuantizationConfig = field(default_factory=QuantizationConfig)


def load_config(path: str | Path) -> Config:
    """Load configuration from a TOML file."""
    path = Path(path)
    with open(path, "rb") as f:
        raw = tomllib.load(f)

    config = Config()

    if "engine" in raw:
        for k, v in raw["engine"].items():
            setattr(config.engine, k, v)
    if "network" in raw:
        for k, v in raw["network"].items():
            setattr(config.network, k, v)
    if "training" in raw:
        for k, v in raw["training"].items():
            if k == "loss":
                for lk, lv in v.items():
                    setattr(config.training.loss, lk, lv)
            else:
                setattr(config.training, k, v)
    if "supcon" in raw:
        for k, v in raw["supcon"].items():
            if k == "schedule":
                for sk, sv in v.items():
                    setattr(config.supcon.schedule, sk, sv)
            else:
                setattr(config.supcon, k, v)
    if "quantization" in raw:
        for k, v in raw["quantization"].items():
            setattr(config.quantization, k, v)

    return config
