from pathlib import Path
import numpy as np
import cv2
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))
import app.core.registration as reg

def test_register_ecc_handles_nan(monkeypatch):
    def fake_findTransformECC(tpl, img, W, mode, criteria, inputMask=None, gaussFiltSize=5):
        W_nan = np.full_like(W, np.nan, dtype=np.float32)
        return 1.0, W_nan

    monkeypatch.setattr(cv2, "findTransformECC", fake_findTransformECC)

    ref = np.tile(np.arange(10, dtype=np.uint8), (10, 1))
    mov = ref.copy()

    success, W, warped, mask = reg.register_ecc(ref, mov, model="affine")

    assert not success
    assert np.array_equal(W, np.eye(2, 3, dtype=np.float32))
    assert np.array_equal(warped, mov)
    assert np.array_equal(mask, np.zeros_like(mov, dtype=np.uint8))
