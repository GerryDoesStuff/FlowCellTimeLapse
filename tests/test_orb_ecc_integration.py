from pathlib import Path
import numpy as np
import cv2
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))
import app.core.registration as reg


def test_orb_ecc_uses_orb_init(monkeypatch):
    W_orb = np.array([[1, 0, 2], [0, 1, 3], [0, 0, 1]], dtype=np.float32)

    def fake_register_orb(ref, mov, model="affine", orb_features=4000, match_ratio=0.75):
        return True, W_orb.copy(), mov, np.ones_like(mov, dtype=np.uint8), False

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

    success, W_refined, _, _ = reg.register_orb_ecc(ref, mov, model="homography", max_iters=10, eps=1e-4)

    assert success
    assert np.allclose(captured["init"], W_orb)
    expected = W_orb.copy()
    expected[0, 2] += 1
    expected[1, 2] += 1
    assert np.allclose(W_refined, expected)
