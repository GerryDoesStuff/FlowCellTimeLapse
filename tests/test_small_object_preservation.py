import numpy as np
import sys
from pathlib import Path

# Ensure the application package is on the import path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from app.core.segmentation import segment


def test_small_objects_preserved_with_defaults():
    img = np.zeros((20, 20), dtype=np.uint8)
    img[5:7, 5:10] = 255  # 10 px object

    seg = segment(
        img,
        method="manual",
        invert=False,
        manual_thresh=100,
        morph_open_radius=0,
        morph_close_radius=0,
    )

    assert seg.sum() == 10


def test_skip_outline_preserves_small_bright_objects():
    img = np.zeros((20, 20), dtype=np.uint8)
    img[5:10, 5:10] = 255  # 25 px square

    seg_default = segment(
        img,
        method="otsu",
        invert=False,
        auto_skip_outline=False,
        morph_open_radius=0,
        morph_close_radius=0,
    )

    seg_skip = segment(
        img,
        method="otsu",
        invert=False,
        skip_outline=True,
        morph_open_radius=0,
        morph_close_radius=0,
    )

    assert seg_default.sum() == 0
    assert seg_skip.sum() == 25


def test_difference_mode_auto_skips_outline():
    img = np.zeros((20, 20), dtype=np.uint8)
    img[5:10, 5:10] = 255  # 25 px square

    seg_diff = segment(
        img,
        method="otsu",
        invert=False,
        use_diff=True,
        morph_open_radius=0,
        morph_close_radius=0,
    )

    seg_skip = segment(
        img,
        method="otsu",
        invert=False,
        skip_outline=True,
        morph_open_radius=0,
        morph_close_radius=0,
    )

    assert np.array_equal(seg_diff, seg_skip)
