import numpy as np
import cv2
from pathlib import Path
import sys

# Ensure application package importable when tests run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from app.core.processing import analyze_sequence
from app.core import processing


def create_frames(tmp_path):
    img = np.zeros((32, 32), dtype=np.uint8)
    path0 = tmp_path / "f0.png"
    path1 = tmp_path / "f1.png"
    cv2.imwrite(str(path0), img)
    cv2.imwrite(str(path1), img)
    return [path0, path1]


def setup(monkeypatch):
    def fake_register(ref, mov, model="affine", **kwargs):
        h, w = ref.shape
        mask = np.ones((h, w), dtype=np.uint8)
        return True, np.eye(3, dtype=np.float32), mov, mask

    monkeypatch.setattr(processing, "register_ecc", fake_register)
    monkeypatch.setattr(processing, "segment", lambda img, **kwargs: np.ones_like(img, dtype=np.uint8))

    green = np.zeros((32, 32), dtype=np.uint8)
    magenta = np.zeros((32, 32), dtype=np.uint8)
    green[0:5, 0:5] = 1
    magenta[5:10, 0:5] = 1

    def fake_detect(gm, prev_seg, curr_seg, app_cfg, *, direction, **kwargs):
        return green.copy(), magenta.copy()

    monkeypatch.setattr(processing, "_detect_green_magenta", fake_detect)
    return green, magenta


def run_and_load(paths, direction, tmp_path, green, magenta):
    reg_cfg = {"initial_radius": 0, "gauss_blur_sigma": 0, "clahe_clip": 0, "clahe_grid": 8, "use_masked_ecc": False}
    seg_cfg = {}
    app_cfg = {"direction": direction, "save_masks": True}
    out_dir = tmp_path / f"out_{direction}"
    analyze_sequence(paths, reg_cfg, seg_cfg, app_cfg, out_dir)
    prev_idx = 0 if direction == "first-to-last" else 1
    g_path = out_dir / "diff" / "green" / f"{prev_idx:04d}_bw_green.png"
    m_path = out_dir / "diff" / "magenta" / f"{prev_idx:04d}_bw_magenta.png"
    g = cv2.imread(str(g_path), cv2.IMREAD_GRAYSCALE)
    m = cv2.imread(str(m_path), cv2.IMREAD_GRAYSCALE)
    return g, m


def test_green_magenta_saved_masks(tmp_path, monkeypatch):
    paths = create_frames(tmp_path)
    green, magenta = setup(monkeypatch)
    for direction in ["first-to-last", "last-to-first"]:
        g_img, m_img = run_and_load(paths, direction, tmp_path, green, magenta)
        assert np.array_equal(g_img, green * 255)
        assert np.array_equal(m_img, magenta * 255)
