import numpy as np
import cv2
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from app.core.processing import analyze_sequence


def create_blank_images(tmp_path, n=3):
    paths = []
    for i in range(n):
        img = np.zeros((100, 100), dtype=np.uint8)
        cv2.imwrite(str(tmp_path / f"img_{i}.png"), img)
        paths.append(tmp_path / f"img_{i}.png")
    return paths


def test_per_frame_crop_rectangles(tmp_path):
    paths = create_blank_images(tmp_path, n=3)

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

    app_cfg = {"direction": "first-to-last", "save_diagnostics": False}

    from app.core import processing, segmentation

    mask1 = np.zeros((100, 100), dtype=np.uint8)
    mask1[20:80, 10:60] = 255  # w=50, h=60
    mask2 = np.zeros((100, 100), dtype=np.uint8)
    mask2[0:70, 30:100] = 255  # w=70, h=70
    masks = [mask1, mask2]

    def fake_register(ref, mov, model="affine", **kwargs):
        valid = masks.pop(0)
        return True, np.eye(3, dtype=np.float32), mov.copy(), valid

    def fake_segment(img, **kwargs):
        return np.ones_like(img, dtype=np.uint8)

    processing.register_ecc = fake_register
    segmentation.segment = fake_segment

    out_dir = tmp_path / "out"
    df = analyze_sequence(paths, reg_cfg, seg_cfg, app_cfg, out_dir)

    row1 = df[df["frame_index"] == 1].iloc[0]
    row2 = df[df["frame_index"] == 2].iloc[0]

    assert int(row1["overlap_w"]) == 30
    assert int(row1["overlap_h"]) == 50
    assert int(row2["overlap_w"]) == 30
    assert int(row2["overlap_h"]) == 50
    assert row1["segmentation_method"] == seg_cfg["method"]
    assert row2["difference_method"] == app_cfg.get("difference_method", "abs")
    assert int(row1["area_overlap_px"]) == int(row1["overlap_px"])
    assert int(row2["area_overlap_px"]) == int(row2["overlap_px"])

