import numpy as np
import cv2
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from app.core.processing import analyze_sequence
from app.core import processing, segmentation


def create_shifted_images(tmp_path, shifts):
    base = np.zeros((50, 50), dtype=np.uint8)
    cv2.circle(base, (25, 25), 10, 255, -1)
    paths = []
    for i, dx in enumerate(shifts):
        M = np.array([[1, 0, dx], [0, 1, 0]], dtype=np.float32)
        img = cv2.warpAffine(base, M, (50, 50))
        path = tmp_path / f"img_{i}.png"
        cv2.imwrite(str(path), img)
        paths.append(path)
    return paths


def test_synthetic_registration_alignment(tmp_path):
    paths = create_shifted_images(tmp_path, [0, 5, 10])

    def fake_register(ref, mov, model="affine", **kwargs):
        M = np.array([[1, 0, -5], [0, 1, 0]], dtype=np.float32)
        h, w = ref.shape
        warped = cv2.warpAffine(mov, M, (w, h))
        mask = cv2.warpAffine(np.ones_like(mov, dtype=np.uint8), M, (w, h))
        mask = (mask > 0).astype(np.uint8)
        return True, M, warped, mask

    processing.register_ecc = fake_register
    segmentation.segment = lambda img, **kwargs: np.ones_like(img, dtype=np.uint8)

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

    app_cfg = {"direction": "first-to-last", "save_intermediates": True}

    out_dir = tmp_path / "out"
    analyze_sequence(paths, reg_cfg, seg_cfg, app_cfg, out_dir)

    reg_dir = out_dir / "registered"
    prev0 = cv2.imread(str(reg_dir / "0000_prev.png"), cv2.IMREAD_GRAYSCALE)
    mov1 = cv2.imread(str(reg_dir / "0001_mov.png"), cv2.IMREAD_GRAYSCALE)
    assert np.array_equal(prev0, mov1)

    prev1 = cv2.imread(str(reg_dir / "0001_prev.png"), cv2.IMREAD_GRAYSCALE)
    mov2 = cv2.imread(str(reg_dir / "0002_mov.png"), cv2.IMREAD_GRAYSCALE)
    assert np.array_equal(prev1, mov2)
