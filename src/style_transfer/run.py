"""Hydra entry point for style transfer.

A run is composed from `configs/` (backbone / data / transfer / runtime groups)
and written into Hydra's per-run output directory, so repeated runs and
`--multirun` sweeps never collide. Things to try::

    python -m style_transfer.run --cfg job          # print the config
    python -m style_transfer.run backbone=vgg16     # swap a group
    python -m style_transfer.run runtime.image_size=256   # override
    python -m style_transfer.run --multirun data=pair1,pair2,pair3
"""

from __future__ import annotations

import hydra

from pathlib import Path
from omegaconf import DictConfig
from hydra.core.hydra_config import HydraConfig

from style_transfer import config_schema
from style_transfer.device import pick_device
from style_transfer.models import build_extractor
from style_transfer.transforms import load_image, save_image
from style_transfer.transfer import TransferConfig, run_style_transfer

# Register the structured-config schema so Hydra type-checks the composed YAML.
config_schema.register()

########################################
#               Core run               #
########################################


def run(cfg: DictConfig, out_dir: Path) -> Path:
    """Optimize one pair, write the result into ``out_dir``, return its path."""
    # ── 1. Resolve the device and build the frozen backbone ──
    device = pick_device(cfg.runtime.device)
    extractor = build_extractor(cfg.backbone.name).to(device)

    # ── 2. Load the content / style pair (from the `data` group) ──
    content = load_image(cfg.data.content, cfg.runtime.image_size, device)
    style = load_image(cfg.data.style, cfg.runtime.image_size, device)

    # ── 3. Bridge the composed config into TransferConfig ──
    transfer_cfg = TransferConfig(
        content_layers=list(cfg.backbone.content_layers),
        style_layers=list(cfg.backbone.style_layers),
        content_weight=cfg.transfer.content_weight,
        style_weight=cfg.transfer.style_weight,
        tv_weight=cfg.transfer.tv_weight,
        optimizer=cfg.transfer.optimizer,
        learning_rate=cfg.transfer.learning_rate,
        steps=cfg.transfer.steps,
    )

    # ── 4. Optimize and write the result into the run directory ──
    result = run_style_transfer(content, style, extractor, transfer_cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / str(cfg.runtime.output)
    save_image(result.image, out_path)
    return out_path


########################################
#             Entry point              #
########################################


@hydra.main(
    version_base=None,
    config_path="../../configs",
    config_name="config",
)
def main(cfg: DictConfig) -> None:
    # ── Hydra gives each run (and each --multirun job) its own output dir ──
    out_dir = Path(HydraConfig.get().runtime.output_dir)
    out_path = run(cfg, out_dir)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
