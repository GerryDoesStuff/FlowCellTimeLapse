import numpy as np
import sys
from pathlib import Path

# Ensure the application package is on the import path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from app.core.segmentation import segment


def test_manual_segmentation_threshold_changes():
    img = np.array(
        [[10, 60, 200],
         [100, 150, 250],
         [30, 180, 220]],
        dtype=np.uint8,
    )

    seg_low = segment(
        img,
        method="manual",
        invert=False,
        manual_thresh=100,
        morph_open_radius=0,
        morph_close_radius=0,
        remove_objects_smaller_px=0,
        remove_holes_smaller_px=0,
    )

    seg_high = segment(
        img,
        method="manual",
        invert=False,
        manual_thresh=200,
        morph_open_radius=0,
        morph_close_radius=0,
        remove_objects_smaller_px=0,
        remove_holes_smaller_px=0,
    )

    expected_low = np.array(
        [[0, 0, 1],
         [1, 1, 1],
         [0, 1, 1]],
        dtype=np.uint8,
    )

    expected_high = np.array(
        [[0, 0, 1],
         [0, 0, 1],
         [0, 0, 1]],
        dtype=np.uint8,
    )

    assert np.array_equal(seg_low, expected_low)
    assert np.array_equal(seg_high, expected_high)
    assert not np.array_equal(seg_low, seg_high)
