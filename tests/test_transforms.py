"""Tests for image <-> tensor conversions."""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image

from style_transfer.transforms import build_transforms, load_image, save_image


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


def test_to_image_clamps_out_of_range():
    _, to_image = build_transforms(image_size=8)
    out = to_image(torch.full((1, 3, 8, 8), 100.0))
    assert np.asarray(out).max() <= 255


def test_load_and_save_image_roundtrip_via_disk(tmp_path):
    # Exercises the public file I/O used by the CLI and experiments.
    src = tmp_path / "src.png"
    Image.fromarray(np.random.RandomState(1).randint(0, 256, (40, 40, 3), dtype=np.uint8)).save(src)

    tensor = load_image(src, image_size=32)
    assert tensor.shape == (1, 3, 32, 32)

    dst = tmp_path / "out.jpg"
    save_image(tensor, dst, image_size=32)
    assert dst.exists() and Image.open(dst).size == (32, 32)
