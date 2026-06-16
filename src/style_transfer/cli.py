"""Command-line interface for neural style transfer.

Stylize one or many content/style pairs in a single call::

    # one pair
    style-transfer --content cat.jpg --style scream.jpg --output out.jpg

    # several pairs at once (paired by position), written to --out-dir
    style-transfer --content a.jpg b.jpg --style x.jpg y.jpg --out-dir results/

    # one style applied to many contents (the single list is broadcast)
    style-transfer --content a.jpg b.jpg c.jpg --style scream.jpg --out-dir results/
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .models import build_extractor
from .transfer import TransferConfig, run_style_transfer
from .transforms import load_image, save_image

# Sensible default layer choices per backbone.
DEFAULTS = {
    "vgg16": (["conv4_2"], ["conv1_1", "conv2_1", "conv3_1", "conv4_1", "conv5_1"]),
    "vgg19": (["conv4_2"], ["conv1_1", "conv2_1", "conv3_1", "conv4_1", "conv5_1"]),
    "resnet50": (
        ["stage3_block2"],
        ["stage1_block0", "stage2_block0", "stage3_block0", "stage4_block0"],
    ),
}


def pick_device(
    requested: str,
) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def pair_inputs(
    contents: list[str],
    styles: list[str],
) -> list[tuple[str, str]]:
    """Pair content and style paths, broadcasting a single-element list."""
    if len(contents) == len(styles):
        return list(zip(contents, styles, strict=True))
    if len(styles) == 1:
        return [(c, styles[0]) for c in contents]
    if len(contents) == 1:
        return [(contents[0], s) for s in styles]
    raise SystemExit(
        f"cannot pair {len(contents)} content and {len(styles)} style images: "
        "provide equal counts, or a single content/style to broadcast"
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Neural style transfer (Gatys et al.).")
    p.add_argument("--content", required=True, nargs="+", help="content image path(s)")
    p.add_argument("--style", required=True, nargs="+", help="style image path(s)")
    p.add_argument("--output", help="output path (single pair only)")
    p.add_argument("--out-dir", default=".", help="directory for batch outputs")
    p.add_argument("--backbone", default="vgg19", choices=list(DEFAULTS))
    p.add_argument("--image-size", type=int, default=512)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--optimizer", default="lbfgs", choices=["lbfgs", "adam"])
    p.add_argument("--lr", type=float, default=1.0)
    p.add_argument("--style-weight", type=float, default=1e6)
    p.add_argument("--content-weight", type=float, default=1.0)
    p.add_argument("--tv-weight", type=float, default=1.0)
    p.add_argument("--device", default="auto")
    return p


def main(
    argv: list[str] | None = None,
) -> None:
    args = build_parser().parse_args(argv)
    device = pick_device(args.device)

    pairs = pair_inputs(args.content, args.style)
    if args.output and len(pairs) > 1:
        raise SystemExit("--output works for a single pair only; use --out-dir for batches")

    # The frozen backbone is built once and reused across all pairs.
    extractor = build_extractor(args.backbone).to(device)
    content_layers, style_layers = DEFAULTS[args.backbone]
    cfg = TransferConfig(
        content_layers=content_layers,
        style_layers=style_layers,
        content_weight=args.content_weight,
        style_weight=args.style_weight,
        tv_weight=args.tv_weight,
        steps=args.steps,
        optimizer=args.optimizer,
        learning_rate=args.lr,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for content_path, style_path in pairs:
        content = load_image(content_path, args.image_size, device)
        style = load_image(style_path, args.image_size, device)
        if args.output and len(pairs) == 1:
            out_path = Path(args.output)
        else:
            stem = f"{Path(content_path).stem}__{Path(style_path).stem}.jpg"
            out_path = out_dir / stem

        print(f"\n{content_path} + {style_path} -> {out_path}")

        def log(
            step: int,
            _image: torch.Tensor,
            losses_dict: dict[str, float],
        ) -> None:
            print(
                f"  step {step:>3} | " + " | ".join(f"{k}={v:.4g}" for k, v in losses_dict.items())
            )

        result = run_style_transfer(content, style, extractor, cfg, callback=log)
        save_image(result.image, out_path, args.image_size)

    print(f"\ndone ({len(pairs)} image{'s' if len(pairs) != 1 else ''})")


if __name__ == "__main__":
    main()
