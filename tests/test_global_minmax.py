import numpy as np
import cv2
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from app.core.io_utils import compute_global_minmax, imread_gray


def test_compute_global_minmax(tmp_path):
    imgs = [
        np.full((5, 5), 10, dtype=np.uint16),
        np.full((5, 5), 100, dtype=np.uint16),
        np.full((5, 5), 1000, dtype=np.uint16),
    ]
    paths = []
    for i, img in enumerate(imgs):
        p = tmp_path / f"img_{i}.png"
        cv2.imwrite(str(p), img)
        paths.append(p)
    lo, hi = compute_global_minmax(paths)
    assert (lo, hi) == (10, 1000)
    # Ensure imread_gray uses these values for normalization
    g_min = imread_gray(paths[0], normalize=True, scale_minmax=(lo, hi))
    g_max = imread_gray(paths[-1], normalize=True, scale_minmax=(lo, hi))
    assert g_min.dtype == np.uint8 and g_min.min() == 0
    assert g_max.max() == 255
