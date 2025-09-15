import numpy as np
import cv2
from pathlib import Path
import sys

# Ensure application package importable when tests run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from app.core.processing import _detect_green_magenta


def test_percentile_sparse_threshold():
    gm = np.zeros((5, 5, 3), dtype=np.uint8)
    # Magenta differences with varying intensities
    gm[1, 0] = (10, 0, 10)
    gm[1, 1] = (20, 0, 20)
    gm[1, 2] = (30, 0, 30)
    # Green differences with higher intensities
    gm[3, 2] = (0, 100, 0)
    gm[3, 3] = (0, 200, 0)
    gm[3, 4] = (0, 250, 0)

    prev_seg = np.ones((5, 5), dtype=np.uint8)
    curr_seg = np.ones((5, 5), dtype=np.uint8)
    app_cfg = {
        "gm_thresh_method": "percentile",
        "gm_thresh_percentile_magenta": 50,
        "gm_thresh_percentile_green": 50,
    }

    green, magenta = _detect_green_magenta(
        gm, prev_seg, curr_seg, app_cfg, direction="first-to-last"
    )

    assert magenta.sum() == 1
    assert green.sum() == 1
    assert magenta[1, 2] == 1
    assert green[3, 4] == 1
    assert magenta[1, 0] == 0 and magenta[1, 1] == 0
    assert green[3, 2] == 0 and green[3, 3] == 0

