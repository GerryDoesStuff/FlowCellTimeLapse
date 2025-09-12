# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]
### Added
- Support for `yen` and `multi_otsu` thresholding methods in `segment`.
  - `tests/test_segmentation_yen.py` covers Yen thresholding on low-contrast images.
  - `tests/test_segmentation_multi_otsu.py` checks Multi-Otsu segmentation with two classes.
- Usage notes: Yen excels on low-contrast backgrounds, while Multi-Otsu is suited for images with distinct histogram peaks.

