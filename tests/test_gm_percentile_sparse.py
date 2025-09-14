import numpy as np
import cv2
from pathlib import Path
import sys

# Ensure application package importable when tests run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from app.core.processing import _detect_green_magenta


def test_percentile_sparse_threshold():
    gm = np.zeros((8, 8, 3), dtype=np.uint8)
    # High-intensity magenta and green differences
    gm[1, 2] = (255, 0, 255)
    gm[3, 4] = (0, 255, 0)
    # Low-intensity noise that should be ignored
    gm[0, 1] = (30, 0, 30)
    gm[2, 0] = (0, 30, 0)

    prev_seg = np.ones((8, 8), dtype=np.uint8)
    curr_seg = np.ones((8, 8), dtype=np.uint8)
    app_cfg = {"gm_thresh_method": "percentile", "gm_thresh_percentile": 50}

    green, magenta = _detect_green_magenta(
        gm, prev_seg, curr_seg, app_cfg, direction="first-to-last"
    )

    assert magenta.sum() == 1
    assert green.sum() == 1
    assert magenta[1, 2] == 1
    assert green[3, 4] == 1
    assert magenta[0, 1] == 0 and green[2, 0] == 0
