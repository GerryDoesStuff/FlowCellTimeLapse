from pathlib import Path
import numpy as np
import cv2
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))
import app.core.registration as reg


def test_orb_ecc_uses_orb_init(monkeypatch):
    W_orb = np.array([[1, 0, 2], [0, 1, 3], [0, 0, 1]], dtype=np.float32)

    def fake_register_orb(ref, mov, model="affine", orb_features=4000, match_ratio=0.75,
                          fallback_model="affine", *, min_keypoints=8, min_matches=8,
                          use_ecc_fallback=True):
        return True, W_orb.copy(), mov, np.ones_like(mov, dtype=np.uint8), False, 12, 34

    captured = {}

    def fake_findTransformECC(tpl, img, W, mode, criteria, inputMask=None, gaussFiltSize=5):
        captured["init"] = W.copy()
        W_ref = W.copy()
        W_ref[0, 2] += 1
        W_ref[1, 2] += 1
        return 1.0, W_ref

    monkeypatch.setattr(reg, "register_orb", fake_register_orb)
    monkeypatch.setattr(cv2, "findTransformECC", fake_findTransformECC)
    monkeypatch.setattr(cv2, "warpPerspective", lambda img, H, dsize, flags=0: img)
    monkeypatch.setattr(cv2, "warpAffine", lambda img, M, dsize, flags=0: img)

    ref = np.zeros((10, 10), dtype=np.uint8)
    mov = np.zeros((10, 10), dtype=np.uint8)

    success, W_refined, _, _, n1, n2 = reg.register_orb_ecc(ref, mov, model="homography", max_iters=10, eps=1e-4)

    assert success
    # ECC should receive the inverse of the ORB matrix
    assert np.allclose(captured["init"], np.linalg.inv(W_orb))
    # The returned matrix should be the inverse of the refined ECC matrix
    expected = W_orb.copy()
    expected[0, 2] -= 1
    expected[1, 2] -= 1
    assert np.allclose(W_refined, expected)
    assert n1 == 12 and n2 == 34


def test_orb_ecc_aligns_shift_direction(monkeypatch):
    dx = 5
    ref = np.zeros((40, 40), dtype=np.uint8)
    cv2.circle(ref, (20, 20), 5, 255, -1)
    mov = np.zeros_like(ref)
    cv2.circle(mov, (20 + dx, 20), 5, 255, -1)

    def fake_register_orb(ref_img, mov_img, **kwargs):
        return True, np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32), mov_img, np.ones_like(mov_img, dtype=np.uint8), False, 20, 20

    monkeypatch.setattr(reg, "register_orb", fake_register_orb)

    success, W, warped, _ , _, _ = reg.register_orb_ecc(ref, mov, model="translation", max_iters=50, eps=1e-4)

    assert success
    # Returned warp should map moving -> reference, undoing the +dx shift
    assert np.allclose(W, np.array([[1, 0, -dx], [0, 1, 0]], dtype=np.float32), atol=0.5)
    # Warped image should align with reference location
    cy, cx = np.argwhere(warped > 0).mean(axis=0)
    assert abs(cx - 20) < 0.5 and abs(cy - 20) < 0.5
