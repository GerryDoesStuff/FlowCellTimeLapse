import numpy as np
import cv2
from pathlib import Path
import sys
import logging

sys.path.append(str(Path(__file__).resolve().parents[1]))

from app.core.processing import analyze_sequence


def create_blank_images(tmp_path, n=2):
    paths = []
    for i in range(n):
        img = np.zeros((100, 100), dtype=np.uint8)
        cv2.imwrite(str(tmp_path / f"img_{i}.png"), img)
        paths.append(tmp_path / f"img_{i}.png")
    return paths


def test_warns_and_preserves_mask(tmp_path, caplog):
    paths = create_blank_images(tmp_path, n=2)

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
        "manual_thresh": 0,
        "invert": True,
        "morph_open_radius": 0,
        "morph_close_radius": 0,
        "remove_objects_smaller_px": 0,
        "remove_holes_smaller_px": 0,
    }

    app_cfg = {"direction": "first-to-last", "save_intermediates": False}

    from app.core import processing

    def fake_register(ref, mov, model="affine", **kwargs):
        h, w = ref.shape
        valid = np.zeros((h, w), dtype=np.uint8)
        valid[0, 0] = 1
        return True, np.eye(3, dtype=np.float32), mov.copy(), valid

    processing.register_ecc = fake_register

    out_dir = tmp_path / "out"
    with caplog.at_level(logging.WARNING):
        df = analyze_sequence(paths, reg_cfg, seg_cfg, app_cfg, out_dir)

    assert any("overlap area" in rec.message for rec in caplog.records)
    row = df[df["frame_index"] == 1].iloc[0]
    assert row["overlap_w"] == 1
    assert row["overlap_h"] == 1
