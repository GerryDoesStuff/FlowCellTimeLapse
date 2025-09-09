import sys
from pathlib import Path
import numpy as np
import cv2

sys.path.append(str(Path(__file__).resolve().parents[1]))
from app.core.registration import register_orb


def dummy_keypoints(n):
    return [cv2.KeyPoint(float(i), float(i), 1) for i in range(n)]


def test_orb_parameters_affect_registration(monkeypatch):
    class DummyORB:
        def detectAndCompute(self, img, mask):
            kps = dummy_keypoints(10)
            desc = np.zeros((10, 32), dtype=np.uint8)
            return kps, desc

    def fake_orb_create(n):
        return DummyORB()

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

    def fake_bfmatcher(norm, crossCheck):
        return DummyMatcher()

    monkeypatch.setattr(cv2, "ORB_create", fake_orb_create)
    monkeypatch.setattr(cv2, "BFMatcher", fake_bfmatcher)
    monkeypatch.setattr(cv2, "findHomography", lambda dst, src, method, ransac: (np.eye(3, dtype=np.float32), None))
    monkeypatch.setattr(cv2, "warpPerspective", lambda img, H, dsize, flags=0: img)
    monkeypatch.setattr(cv2, "warpAffine", lambda img, M, dsize, flags=0: img)

    ref = np.zeros((5, 5), dtype=np.uint8)
    mov = np.zeros((5, 5), dtype=np.uint8)
    H_good, _, _ = register_orb(ref, mov, orb_features=500, match_ratio=0.7)
    assert H_good.shape == (3, 3)
    H_bad, _, _ = register_orb(ref, mov, orb_features=500, match_ratio=0.4)
    assert H_bad.shape == (2, 3)
