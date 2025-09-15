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
        cv2.circle(img, (50, 50), 20, 255, -1)
        cv2.line(img, (0, 0), (99, 99), 128, 2)
        cv2.line(img, (99, 0), (0, 99), 128, 2)
        cv2.imwrite(str(tmp_path / f"img_{i}.png"), img)
        paths.append(tmp_path / f"img_{i}.png")
    return paths


def run(paths, monkeypatch):
    reg_cfg = {
        "model": "translation",
        "max_iters": 10,
        "gauss_blur_sigma": 0,
        "clahe_clip": 0,
        "clahe_grid": 8,
        "use_masked_ecc": False,
        "method": "ECC",
        "eps": 1e-6,
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
    from app.core import processing

    def fake_register(ref, mov, model="affine", **kwargs):
        h, w = ref.shape
        return True, np.eye(3, dtype=np.float32), mov.copy(), np.ones((h, w), dtype=np.uint8)

    monkeypatch.setattr(processing, "register_ecc", fake_register)
    out_dir = paths[0].parent / "out"
    return analyze_sequence(paths, reg_cfg, seg_cfg, app_cfg, out_dir)


def test_summary_metadata(tmp_path, monkeypatch):
    paths = create_blank_images(tmp_path, n=3)
    df = run(paths, monkeypatch)
    assert {"segmentation_method", "difference_method", "area_overlap_px"}.issubset(df.columns)
    assert (df["segmentation_method"] == "manual").all()
    assert (df["difference_method"] == "abs").all()
    assert (df["area_overlap_px"] == df["overlap_px"]).all()
