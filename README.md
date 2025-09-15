# Yeast Flowcell Accumulation — PyQt App

PyQt6 desktop app for registration → segmentation → subtraction → evaluation of time-lapse images
to quantify yeast attachment over time. Multi-core by default; optional CUDA acceleration when
OpenCV with CUDA is available.

## Quick start
```bash
python -m venv .venv && source .venv/bin/activate  # (Linux/macOS)
# or .venv\Scripts\activate (Windows)

pip install -r requirements.txt
python app/main.py
```

### GPU (optional)
If your OpenCV is CUDA-enabled, the app will automatically use it for some ops (warp, threshold, absdiff).
Otherwise it runs on CPU with multi-processing. To install a CUDA wheel, consider prebuilt packages from
`opencv-python` ecosystem or build from source with CUDA. The code checks `cv2.cuda` presence.

### Run Analysis
After selecting an image folder and setting parameters, click **Run Analysis** to process the
time-lapse sequence. Output controls provide an option:

- **Save diagnostic outputs** – write optional masks (gain/loss, overlap/union, segmentation overlays, GM composites).

### Intermediate outputs
The pipeline always writes core results to `registered/mov/` and the `diff/` subdirectories
`raw/`, `bw/`, `green/`, and `magenta/`. Enabling **Save diagnostic outputs** adds
further artifacts such as `diff/new/`, `diff/lost/`, `diff/gain/`, `diff/loss/`,
`diff/overlap/`, `diff/union/`, `diff/gm/`, and segmentation masks in `seg/`.

### Gain/Loss Detection

After a successful segmentation preview, a collapsible **Gain/Loss Detection**
section becomes available below the segmentation controls. It exposes the
thresholding, morphology and saturation settings for detecting newly appeared
(magenta) or disappeared (green) regions and includes a preview button. The
section and its preview button remain disabled until segmentation completes, and
any new folder selection or parameter tweak collapses and disables it again.

The separation between the magenta and green channels is determined in LAB
color space using an adaptive threshold on the "a" channel. The composite
itself blends the current frame with the previous according to
`gm_opacity` (percentage of the current frame, default `50`). A saturation
boost (`gm_saturation`, default `1.0`) scales the chroma channel before
thresholding to emphasize subtle differences. By default an Otsu threshold
(`gm_thresh_method="otsu"`) is used, but a percentile (`gm_thresh_percentile`,
default `99.0`) can be selected instead. To remove speckles and recover full
structures the masks are optionally processed with morphological closing
(`gm_close_kernel`, default `3`) and dilation (`gm_dilate_kernel`, default
`0`).

### Project layout
- `app/main.py` — app entry, sets up MainWindow.
- `app/ui/main_window.py` — PyQt UI and interactions.
- `app/models/config.py` — dataclasses for parameters; JSON presets; QSettings persistence.
- `app/core/io_utils.py` — file discovery, timestamp spacing, safe I/O.
- `app/core/registration.py` — ECC / ORB(+RANSAC) on CPU; CUDA path when available. ECC assumes sufficient texture; nearly uniform frames can cause the optimizer to diverge, yielding NaN transforms that are handled by falling back to the identity matrix.
- `app/core/segmentation.py` — outline-focused segmentation (black-hat + Otsu, Multi-Otsu, Li, Yen, adaptive, local, or manual thresholding), morphology; `skip_outline` option bypasses the prefilter for low-contrast images.
- `app/core/background.py` — on-the-fly background estimation (temporal median of early frames, fallback to blur).
- `app/core/processing.py` — end-to-end per-frame pipeline; overlap cropping; metrics aggregation.
- `app/core/multiproc.py` — ProcessPool execution with chunking; CPU core detection.
- `app/workers/pipeline_worker.py` — background worker orchestration.
- `requirements.txt` — dependencies.

### Low-contrast frames
The `segment` function accepts a `use_clahe` flag that applies Contrast Limited Adaptive
Histogram Equalization before thresholding. This can recover features in nearly
uniform frames where default segmentation yields an empty mask.

```python
from app.core.segmentation import segment

mask_plain = segment(low_contrast_frame)
mask_clahe = segment(low_contrast_frame, use_clahe=True)
```

With CLAHE enabled (`mask_clahe`), faint cell boundaries become visible compared to the
plain result (`mask_plain`).
