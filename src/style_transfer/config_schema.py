"""Structured-config schemas — the run's typed contract.

These dataclasses describe the shape Hydra validates `configs/` against: every
field has a type, and either a default or `MISSING` (required, supplied by a
group). Registering the root in the ConfigStore makes Hydra reject wrong types
and unknown fields at startup, instead of crashing deep inside the run.
"""

from __future__ import annotations

from omegaconf import MISSING
from dataclasses import field, dataclass
from hydra.core.config_store import ConfigStore

########################################
#            Group schemas             #
########################################


@dataclass
class BackboneConfig:
    """Which feature extractor and which layers feed each loss."""

    name: str = MISSING
    content_layers: list[str] = MISSING
    style_layers: list[str] = MISSING


@dataclass
class DataConfig:
    """The content / style input pair."""

    content: str = MISSING
    style: str = MISSING


@dataclass
class TransferParams:
    """Optimization recipe: loss weights, optimizer, learning rate, steps."""

    content_weight: float = 1.0
    style_weight: float = 1e6
    tv_weight: float = 1.0
    optimizer: str = "lbfgs"
    learning_rate: float = 1.0
    steps: int = 50


@dataclass
class RuntimeConfig:
    """Execution + I/O: working resolution, device, output filename."""

    image_size: int = 512
    device: str = "auto"
    output: str = "result.jpg"


########################################
#           Root & registry            #
########################################


@dataclass
class Config:
    """The composed run config (one group per concern)."""

    backbone: BackboneConfig = field(default_factory=BackboneConfig)
    data: DataConfig = field(default_factory=DataConfig)
    transfer: TransferParams = field(default_factory=TransferParams)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)


def register() -> None:
    # ── Expose the schema as `config_schema` for config.yaml's defaults ──
    ConfigStore.instance().store(name="config_schema", node=Config)
