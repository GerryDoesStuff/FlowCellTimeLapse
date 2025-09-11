import sys
from pathlib import Path
import numpy as np
import cv2

sys.path.append(str(Path(__file__).resolve().parents[1]))
import app.core.registration as reg


def dummy_keypoints(n):
    return [cv2.KeyPoint(float(i), float(i), 1) for i in range(n)]


def setup_orb(monkeypatch):
    class DummyORB:
        def detectAndCompute(self, img, mask):
            kps = dummy_keypoints(10)
            desc = np.zeros((10, 32), dtype=np.uint8)
            return kps, desc
    monkeypatch.setattr(cv2, "ORB_create", lambda n: DummyORB())

    class DummyMatcher:
        def knnMatch(self, d1, d2, k=2):
            class Match:
                def __init__(self, dist, q=0, t=0):
                    self.distance = dist
                    self.queryIdx = q
                    self.trainIdx = t
            m = Match(5)
            n = Match(10)
            return [(m, n)] * 10
    monkeypatch.setattr(cv2, "BFMatcher", lambda norm, crossCheck: DummyMatcher())


def test_oversized_translation_fallback(monkeypatch):
    setup_orb(monkeypatch)

    def fake_estimate(dst, src, method=cv2.RANSAC, ransacReprojThreshold=3.0):
        return np.array([[1.0, 0.0, 100.0], [0.0, 1.0, 100.0]], dtype=np.float32), None
    monkeypatch.setattr(cv2, "estimateAffine2D", fake_estimate)

    called = {}

    def fake_register_ecc(ref, mov, model="affine"):
        called["yes"] = True
        return True, np.eye(2, 3, dtype=np.float32), mov, np.ones_like(ref, dtype=np.uint8)

    monkeypatch.setattr(reg, "register_ecc", fake_register_ecc)

    ref = np.zeros((5, 5), dtype=np.uint8)
    mov = np.zeros((5, 5), dtype=np.uint8)

    success, W, warped, valid_mask, fb, n1, n2 = reg.register_orb(ref, mov, model="affine")

    assert called.get("yes")
    assert success and fb
    assert W.shape == (2, 3)


def test_oversized_translation_no_fallback(monkeypatch):
    setup_orb(monkeypatch)
    monkeypatch.setattr(
        cv2,
        "estimateAffine2D",
        lambda dst, src, method=cv2.RANSAC, ransacReprojThreshold=3.0: (
            np.array([[1.0, 0.0, 100.0], [0.0, 1.0, 100.0]], dtype=np.float32),
            None,
        ),
    )

    ref = np.zeros((5, 5), dtype=np.uint8)
    mov = np.zeros((5, 5), dtype=np.uint8)

    success, W, warped, valid_mask, fb, n1, n2 = reg.register_orb(
        ref, mov, model="affine", use_ecc_fallback=False
    )

    assert not success and not fb
    assert np.allclose(W, np.eye(2, 3, dtype=np.float32))
    assert np.array_equal(warped, mov)
    assert np.count_nonzero(valid_mask) == 0
