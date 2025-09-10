import numpy as np
import cv2
from pathlib import Path
import sys
import logging

# Ensure the application package is importable when tests are run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from app.core.processing import analyze_sequence


def create_dummy_images(tmp_path, n=3):
    paths = []
    for i in range(n):
        img = np.ones((10, 10), dtype=np.uint8) * i
        cv2.imwrite(str(tmp_path / f"img_{i}.png"), img)
        paths.append(tmp_path / f"img_{i}.png")
    return paths


def run_analyze(paths, direction, caplog=None):
    reg_cfg = {
        "model": "affine",
        "max_iters": 1,
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
    app_cfg = {
        "direction": direction,
        "save_intermediates": False,
    }
    out_dir = paths[0].parent / "out"
    if caplog:
        with caplog.at_level(logging.INFO):
            df = analyze_sequence(paths, reg_cfg, seg_cfg, app_cfg, out_dir)
    else:
        df = analyze_sequence(paths, reg_cfg, seg_cfg, app_cfg, out_dir)
    return int(df.loc[df["is_reference"], "frame_index"].iloc[0])


def test_last_to_first(tmp_path, caplog):
    paths = create_dummy_images(tmp_path)
    ref = run_analyze(paths, "last-to-first", caplog)
    assert ref == len(paths) - 1
    assert "direction=last-to-first" in caplog.text
    assert f"reference_frame_index={len(paths) - 1}" in caplog.text


def test_first_to_last(tmp_path, caplog):
    paths = create_dummy_images(tmp_path)
    ref = run_analyze(paths, "first-to-last", caplog)
    assert ref == 0
    assert "direction=first-to-last" in caplog.text
    assert "reference_frame_index=0" in caplog.text


def test_last_to_first_registration_order(tmp_path, monkeypatch):
    paths = create_dummy_images(tmp_path)

    calls = []

    from app.core import processing

    def fake_register_ecc(ref_gray, g_norm, model="affine", max_iters=1000, eps=1e-6, mask=None):
        calls.append(int(g_norm[0, 0]))
        return True, np.eye(2, 3, dtype=np.float32), g_norm, np.ones_like(g_norm, dtype=np.uint8)

    monkeypatch.setattr(processing, "register_ecc", fake_register_ecc)
    monkeypatch.setattr(processing, "preprocess", lambda g, *a, **k: g)
    monkeypatch.setattr(processing, "segment", lambda *a, **k: np.ones_like(a[0], dtype=np.uint8))

    reg_cfg = {
        "model": "affine",
        "max_iters": 1,
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
    app_cfg = {
        "direction": "last-to-first",
        "save_intermediates": False,
        "normalize": False,
    }
    out_dir = paths[0].parent / "out"
    df = analyze_sequence(paths, reg_cfg, seg_cfg, app_cfg, out_dir)

    assert df.loc[df["is_reference"], "frame_index"].iloc[0] == len(paths) - 1
    assert len(calls) == len(paths) - 1
    assert (len(paths) - 1) not in calls

