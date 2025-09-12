import numpy as np
import sys
from pathlib import Path
from skimage import filters

# Ensure the application package is on the import path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from app.core.segmentation import segment


def test_yen_segmentation_low_contrast():
    """Yen thresholding on a low-contrast image."""
    img = np.full((5, 5), 120, dtype=np.uint8)
    img[2:, 2:] = 130

    seg = segment(
        img,
        method="yen",
        invert=False,
        skip_outline=True,
        morph_open_radius=0,
        morph_close_radius=0,
        remove_objects_smaller_px=0,
        remove_holes_smaller_px=0,
    )

    t = filters.threshold_yen(img)
    expected = (img >= t).astype(np.uint8)

    assert np.array_equal(seg, expected)
