# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]
### Added
- Support for additional thresholding methods (`multi_otsu`, `li`, `yen`) in `segment` and the UI combo box.
  - `tests/test_segmentation_yen.py` covers Yen thresholding on low-contrast images.
  - `tests/test_segmentation_multi_otsu.py` checks Multi-Otsu segmentation with two classes.
  - `tests/test_segmentation_li.py` verifies Li thresholding.
- Usage notes: Yen excels on low-contrast backgrounds, while Multi-Otsu is suited for images with distinct histogram peaks.
- Gain/loss preview adds `gm_saturation` control and persists `gm_opacity`/
  `gm_saturation` in saved presets and settings.
- Collapsible **Gain/Loss Detection** section that becomes available only after a
  successful segmentation preview.

### Changed
- Green–magenta composite now blends frames using `gm_opacity` to weight the current frame.

