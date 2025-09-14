import numpy as np
import cv2
from pathlib import Path
import sys

# Ensure application package importable when tests run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from app.core.processing import analyze_sequence, overlay_outline
from app.core import processing


def create_frames(tmp_path):
    img0 = np.zeros((32, 32), dtype=np.uint8)
    cv2.rectangle(img0, (5, 5), (15, 15), 255, -1)
    path0 = tmp_path / "frame0.png"
    cv2.imwrite(str(path0), img0)

    img1 = np.zeros_like(img0)
    cv2.rectangle(img1, (10, 5), (20, 15), 255, -1)
    path1 = tmp_path / "frame1.png"
    cv2.imwrite(str(path1), img1)
    return [path0, path1]


def test_gm_composite_opacity_and_saturation(tmp_path, monkeypatch):
    paths = create_frames(tmp_path)

    def fake_register(ref, mov, model="affine", **kwargs):
        h, w = ref.shape
        mask = np.ones((h, w), dtype=np.uint8)
        return True, np.eye(3, dtype=np.float32), mov, mask

    monkeypatch.setattr(processing, "register_ecc", fake_register)
    monkeypatch.setattr(
        processing, "segment", lambda img, **kwargs: (img > 127).astype(np.uint8)
    )

    captured = {}

    real_detect = processing._detect_green_magenta

    def capture_detect(gm_comp, prev_seg, curr_seg, app_cfg, direction):
        captured["gm_composite"] = gm_comp.copy()
        captured["app_cfg"] = dict(app_cfg)
        return real_detect(gm_comp, prev_seg, curr_seg, app_cfg, direction=direction)

    monkeypatch.setattr(processing, "_detect_green_magenta", capture_detect)

    reg_cfg = {
        "initial_radius": 0,
        "gauss_blur_sigma": 0,
        "clahe_clip": 0,
        "use_clahe": False,
    }
    seg_cfg = {}
    app_cfg = {
        "direction": "first-to-last",
        "save_intermediates": True,
        "save_gm_composite": True,
        "overlay_mov_color": (255, 255, 0),
        "gm_opacity": 60,
        "gm_saturation": 1.4,
    }

    out_dir = tmp_path / "out"
    analyze_sequence(paths, reg_cfg, seg_cfg, app_cfg, out_dir)

    alpha = app_cfg["gm_opacity"] / 100.0
    prev = cv2.imread(str(paths[0]), cv2.IMREAD_GRAYSCALE)
    curr = cv2.imread(str(paths[1]), cv2.IMREAD_GRAYSCALE)
    expected_gm = np.stack(
        [
            (curr * alpha).astype(np.uint8),
            (prev * (1 - alpha)).astype(np.uint8),
            (curr * alpha).astype(np.uint8),
        ],
        axis=-1,
    )
    assert np.array_equal(captured["gm_composite"], expected_gm)
    assert captured["app_cfg"]["gm_saturation"] == app_cfg["gm_saturation"]

    saved_gm = cv2.imread(str(out_dir / "diff" / "gm" / "0001_gm.png"))
    lab = cv2.cvtColor(expected_gm, cv2.COLOR_BGR2LAB).astype(np.int16)
    a = lab[..., 1].astype(np.int16) - 128
    a = np.clip(a * app_cfg["gm_saturation"], -255, 255) + 128
    lab[..., 1] = a.astype(np.uint8)
    expected_disp = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
    assert np.array_equal(saved_gm, expected_disp)

    seg_mask = (curr > 127).astype(np.uint8)
    expected_overlay = overlay_outline(curr, mask=seg_mask, color=app_cfg["overlay_mov_color"])
    overlay_img = cv2.imread(str(out_dir / "overlay" / "0001_overlay_mov.png"))
    assert np.array_equal(overlay_img, expected_overlay)
