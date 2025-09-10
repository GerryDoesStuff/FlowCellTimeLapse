import numpy as np
import cv2
from pathlib import Path
import sys
import logging
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from app.core.processing import analyze_sequence


def create_images(tmp_path):
    paths = []
    # first image with a small square
    img0 = np.zeros((50, 50), dtype=np.uint8)
    cv2.rectangle(img0, (10, 10), (20, 20), 255, -1)
    cv2.imwrite(str(tmp_path / "img_0.png"), img0)
    paths.append(tmp_path / "img_0.png")
    # second image blank
    img1 = np.zeros((50, 50), dtype=np.uint8)
    cv2.imwrite(str(tmp_path / "img_1.png"), img1)
    paths.append(tmp_path / "img_1.png")
    return paths


def test_warns_and_skips_ecc_mask(tmp_path, caplog):
    paths = create_images(tmp_path)

    reg_cfg = {
        "model": "translation",
        "max_iters": 10,
        "gauss_blur_sigma": 0,
        "clahe_clip": 0,
        "clahe_grid": 8,
        "use_masked_ecc": False,
        "method": "ECC",
        "eps": 1e-6,
        "growth_factor": 1.0,
        "initial_radius": 0,
    }

    seg_cfg = {
        "method": "manual",
        "manual_thresh": 1,
        "invert": False,
        "morph_open_radius": 0,
        "morph_close_radius": 0,
        "remove_objects_smaller_px": 0,
        "remove_holes_smaller_px": 0,
    }

    app_cfg = {"direction": "first-to-last", "save_intermediates": False}

    from app.core import processing

    def fake_register(ref, mov, model="affine", **kwargs):
        h, w = ref.shape
        valid = np.ones((h, w), dtype=np.uint8)
        return True, np.eye(3, dtype=np.float32), mov.copy(), valid

    processing.register_ecc = fake_register

    out_dir = tmp_path / "out"
    with caplog.at_level(logging.WARNING):
        df = analyze_sequence(paths, reg_cfg, seg_cfg, app_cfg, out_dir)

    assert any("segmentation mask is empty" in rec.message for rec in caplog.records)
    empty_mask_path = out_dir / "binary" / "0001_bw_mov_empty.png"
    assert empty_mask_path.exists()
    row = df[df["frame_index"] == 1].iloc[0]
    assert row["area_mov_px"] == 0


def test_all_masks_empty_error(tmp_path):
    paths = []
    img = np.zeros((50, 50), dtype=np.uint8)
    for i in range(2):
        cv2.imwrite(str(tmp_path / f"img_{i}.png"), img)
        paths.append(tmp_path / f"img_{i}.png")

    reg_cfg = {
        "model": "translation",
        "max_iters": 10,
        "gauss_blur_sigma": 0,
        "clahe_clip": 0,
        "clahe_grid": 8,
        "use_masked_ecc": False,
        "method": "ECC",
        "eps": 1e-6,
        "growth_factor": 1.0,
        "initial_radius": 0,
    }

    seg_cfg = {
        "method": "manual",
        "manual_thresh": 1,
        "invert": False,
        "morph_open_radius": 0,
        "morph_close_radius": 0,
        "remove_objects_smaller_px": 0,
        "remove_holes_smaller_px": 0,
    }

    app_cfg = {"direction": "first-to-last", "save_intermediates": False}

    from app.core import processing

    def fake_register(ref, mov, model="affine", **kwargs):
        h, w = ref.shape
        valid = np.ones((h, w), dtype=np.uint8)
        return True, np.eye(3, dtype=np.float32), mov.copy(), valid

    processing.register_ecc = fake_register

    out_dir = tmp_path / "out"
    with pytest.raises(ValueError, match="All segmentation masks were empty"):
        analyze_sequence(paths, reg_cfg, seg_cfg, app_cfg, out_dir)
    assert not (out_dir / "summary.csv").exists()
