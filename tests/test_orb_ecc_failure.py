from pathlib import Path
import numpy as np
import cv2
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))
import app.core.registration as reg


def test_orb_ecc_handles_cv2_error(monkeypatch):
    def fake_register_orb(ref, mov, **kwargs):
        return True, np.eye(2, 3, dtype=np.float32), mov, np.zeros_like(ref, dtype=np.uint8), False, 0, 0

    def fake_findTransformECC(*args, **kwargs):
        raise cv2.error("fake")

    monkeypatch.setattr(reg, "register_orb", fake_register_orb)
    monkeypatch.setattr(cv2, "findTransformECC", fake_findTransformECC)

    ref = np.zeros((10, 10), dtype=np.uint8)
    mov = ref.copy()

    success, W, warped, mask, _, _ = reg.register_orb_ecc(ref, mov, model="translation")

    assert not success
    assert np.allclose(W, np.eye(2, 3, dtype=np.float32))
    assert np.array_equal(warped, mov)
    assert np.count_nonzero(mask) == 0


def test_orb_ecc_handles_nan_warp(monkeypatch):
    def fake_register_orb(ref, mov, **kwargs):
        return True, np.eye(2, 3, dtype=np.float32), mov, np.ones_like(ref, dtype=np.uint8), False, 0, 0

    def nan_findTransformECC(tpl, img, W, mode, criteria, inputMask=None, gaussFiltSize=5):
        return 1.0, np.full_like(W, np.nan, dtype=np.float32)

    monkeypatch.setattr(reg, "register_orb", fake_register_orb)
    monkeypatch.setattr(cv2, "findTransformECC", nan_findTransformECC)

    ref = np.zeros((10, 10), dtype=np.uint8)
    mov = ref.copy()

    success, W, warped, mask, _, _ = reg.register_orb_ecc(ref, mov, model="translation")

    assert not success
    assert np.allclose(W, np.eye(2, 3, dtype=np.float32))
    assert np.array_equal(warped, mov)
    assert np.count_nonzero(mask) == 0


def test_orb_ecc_rejects_extreme_translation(monkeypatch):
    called = {"ecc": False}

    def fake_register_orb(ref, mov, **kwargs):
        return False, np.eye(2, 3, dtype=np.float32), mov, np.zeros_like(ref, dtype=np.uint8), False, 0, 0

    def fake_findTransformECC(*args, **kwargs):
        called["ecc"] = True
        return 1.0, np.eye(2, 3, dtype=np.float32)

    monkeypatch.setattr(reg, "register_orb", fake_register_orb)
    monkeypatch.setattr(cv2, "findTransformECC", fake_findTransformECC)

    ref = np.zeros((10, 10), dtype=np.uint8)
    mov = ref.copy()

    success, W, warped, mask, _, _ = reg.register_orb_ecc(ref, mov, model="translation")

    assert not success
    assert np.allclose(W, np.eye(2, 3, dtype=np.float32))
    assert np.array_equal(warped, mov)
    assert np.count_nonzero(mask) == 0
    assert not called["ecc"]
