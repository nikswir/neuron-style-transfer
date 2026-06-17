"""Tests for image <-> tensor conversions."""

from __future__ import annotations

import torch
import numpy as np

from PIL import Image

from style_transfer.transforms import load_image, save_image, build_transforms

########################################
#             Conversions              #
########################################


def test_to_tensor_shape_and_batch_dim():
    to_tensor, _ = build_transforms(image_size=64)
    img = Image.new("RGB", (200, 100), (120, 30, 200))
    t = to_tensor(img)
    assert t.shape == (1, 3, 64, 64)


def test_roundtrip_recovers_image():
    to_tensor, to_image = build_transforms(image_size=32)
    original = Image.fromarray(
        np.random.RandomState(0).randint(0, 256, (32, 32, 3), dtype=np.uint8)
    )
    t = to_tensor(original)
    recovered = to_image(t)
    assert recovered.size == (32, 32)
    a = np.asarray(original).astype(np.int16)
    b = np.asarray(recovered).astype(np.int16)
    # Normalize -> de-normalize is exact up to 8-bit quantization.
    assert np.abs(a - b).max() <= 2


########################################
#               Clamping               #
########################################


def test_to_image_clamps_out_of_range():
    _, to_image = build_transforms(image_size=8)
    out = to_image(torch.full((1, 3, 8, 8), 100.0))
    assert np.asarray(out).max() <= 255


def test_to_image_clamps_both_bounds_exactly():
    # ToPILImage scales by 255 and casts to uint8 *without* saturating, so an
    # unclamped out-of-range value wraps around. The clamp to [0, 1] is what
    # guarantees saturation. A normalized value of +5 de-normalizes to >1 and a
    # value of -5 to <0; with the clamp these must hit exactly 255 and 0.
    _, to_image = build_transforms(image_size=8)
    high = np.asarray(to_image(torch.full((1, 3, 8, 8), 5.0)))
    low = np.asarray(to_image(torch.full((1, 3, 8, 8), -5.0)))
    assert high.max() == 255
    assert high.min() == 255
    assert low.max() == 0
    assert low.min() == 0


########################################
#               File I/O               #
########################################


def test_load_and_save_image_roundtrip_via_disk(tmp_path):
    # Exercises the public file I/O used by the CLI and experiments.
    src = tmp_path / "src.png"
    Image.fromarray(np.random.RandomState(1).randint(0, 256, (40, 40, 3), dtype=np.uint8)).save(src)

    tensor = load_image(src, image_size=32)
    assert tensor.shape == (1, 3, 32, 32)

    dst = tmp_path / "out.jpg"
    save_image(tensor, dst, image_size=32)
    assert dst.exists()
    assert Image.open(dst).size == (32, 32)


def test_load_image_converts_non_rgb_to_three_channels(tmp_path):
    # A 1-channel ("L") source must be converted to RGB -> 3 channels.
    src = tmp_path / "gray.png"
    Image.fromarray(np.full((20, 20), 128, dtype=np.uint8), mode="L").save(src)
    tensor = load_image(src, image_size=16)
    assert tensor.shape == (1, 3, 16, 16)
