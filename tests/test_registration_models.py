import sys
from pathlib import Path
import numpy as np
import cv2

sys.path.append(str(Path(__file__).resolve().parents[1]))
from app.core.registration import register_orb


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

    monkeypatch.setattr(cv2, "warpAffine", lambda img, M, dsize, flags=0: img)
    monkeypatch.setattr(cv2, "warpPerspective", lambda img, M, dsize, flags=0: img)
    monkeypatch.setattr(cv2, "findHomography", lambda dst, src, method, ransac: (np.eye(3, dtype=np.float32), np.ones((10,1), dtype=np.uint8)))

    def fake_estimate(dst, src, method=cv2.RANSAC, ransacReprojThreshold=3.0):
        return np.array([[0.5, 0.1, 2.0], [-0.1, 0.5, 3.0]], dtype=np.float32), None
    monkeypatch.setattr(cv2, "estimateAffine2D", fake_estimate)


def test_register_orb_models(monkeypatch):
    setup_orb(monkeypatch)
    ref = np.zeros((5, 5), dtype=np.uint8)
    mov = np.zeros((5, 5), dtype=np.uint8)

    _, H, _, _, fb = register_orb(ref, mov, model="homography")
    assert H.shape == (3, 3) and not fb

    _, A, _, _, fb = register_orb(ref, mov, model="affine")
    assert A.shape == (2, 3) and not fb

    _, E, _, _, fb = register_orb(ref, mov, model="euclidean")
    assert E.shape == (2, 3) and not fb
    R = E[:, :2]
    assert np.allclose(R.T @ R, np.eye(2), atol=1e-6)

    _, T, _, _, fb = register_orb(ref, mov, model="translation")
    assert T.shape == (2, 3) and not fb
    assert np.allclose(T[:, :2], np.eye(2), atol=1e-6)


def test_orb_homography_fallback(monkeypatch):
    setup_orb(monkeypatch)
    # Force findHomography to return a near-singular matrix
    monkeypatch.setattr(cv2, "findHomography", lambda dst, src, method, ransac: (np.zeros((3,3), dtype=np.float32), np.ones((10,1), dtype=np.uint8)))
    import app.core.registration as reg
    monkeypatch.setattr(reg, "register_ecc", lambda ref, mov, model='homography': (True, np.eye(3, dtype=np.float32), mov, np.ones_like(ref, dtype=np.uint8)))
    ref = np.zeros((5,5), dtype=np.uint8)
    mov = np.zeros((5,5), dtype=np.uint8)
    success, _, _, _, fb = reg.register_orb(ref, mov, model="homography")
    assert success and fb


def test_register_orb_fallback_model(monkeypatch):
    import app.core.registration as reg

    class DummyORB:
        def detectAndCompute(self, img, mask):
            return [], None

    monkeypatch.setattr(cv2, "ORB_create", lambda n: DummyORB())

    captured = {}

    def fake_register_ecc(ref, mov, model="affine"):
        captured["model"] = model
        return True, np.eye(3, dtype=np.float32), mov, np.ones_like(ref, dtype=np.uint8)

    monkeypatch.setattr(reg, "register_ecc", fake_register_ecc)

    ref = np.zeros((5, 5), dtype=np.uint8)
    mov = np.zeros((5, 5), dtype=np.uint8)

    reg.register_orb(ref, mov, model="homography", fallback_model="translation")

    assert captured["model"] == "translation"
