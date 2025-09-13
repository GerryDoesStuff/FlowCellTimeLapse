# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]
### Added
- Support for additional thresholding methods (`multi_otsu`, `li`, `yen`) in `segment` and the UI combo box.
  - `tests/test_segmentation_yen.py` covers Yen thresholding on low-contrast images.
  - `tests/test_segmentation_multi_otsu.py` checks Multi-Otsu segmentation with two classes.
  - `tests/test_segmentation_li.py` verifies Li thresholding.
- Usage notes: Yen excels on low-contrast backgrounds, while Multi-Otsu is suited for images with distinct histogram peaks.

### Changed
- Green–magenta composite now blends frames using `overlay_opacity` to weight the current frame.

