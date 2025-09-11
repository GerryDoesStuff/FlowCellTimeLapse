import numpy as np
from app.core.difference import compute_difference


def test_difference_methods_shape_and_range():
    ref = np.zeros((32, 32), dtype=np.uint8)
    mov = ref.copy()
    mov[8:24, 8:24] = 255
    for method in ["abs", "lab", "edges"]:
        diff = compute_difference(ref, mov, method=method)
        assert diff.shape == ref.shape
        assert diff.dtype == np.uint8
        assert diff.min() >= 0 and diff.max() <= 255
