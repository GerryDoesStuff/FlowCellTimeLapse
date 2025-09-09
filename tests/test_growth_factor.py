import numpy as np
import cv2
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from app.core.processing import analyze_sequence


def create_blank_images(tmp_path, n=3):
    paths = []
    for i in range(n):
        img = np.zeros((20, 20), dtype=np.uint8)
        cv2.imwrite(str(tmp_path / f"img_{i}.png"), img)
        paths.append(tmp_path / f"img_{i}.png")
    return paths


def run(paths, growth):
    reg_cfg = {
        "model": "translation",
        "max_iters": 1,
        "gauss_blur_sigma": 0,
        "clahe_clip": 0,
        "clahe_grid": 8,
        "use_masked_ecc": False,
        "method": "ECC",
        "eps": 1e-6,
        "growth_factor": growth,
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
    out_dir = paths[0].parent / f"out_{growth}"
    return analyze_sequence(paths, reg_cfg, seg_cfg, app_cfg, out_dir)


def test_growth_factor_influences_window(tmp_path):
    paths = create_blank_images(tmp_path, n=3)
    df1 = run(paths, 1.0)
    df2 = run(paths, 0.5)
    row1 = df1.loc[df1["frame_index"] == 2].iloc[0]
    row2 = df2.loc[df2["frame_index"] == 2].iloc[0]
    w1, h1 = int(row1["overlap_w"]), int(row1["overlap_h"])
    w2, h2 = int(row2["overlap_w"]), int(row2["overlap_h"])
    assert w2 < w1
    assert h2 < h1
