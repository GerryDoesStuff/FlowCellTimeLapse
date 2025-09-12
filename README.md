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

### Intermediate outputs
When `save_intermediates` is enabled, the pipeline saves additional artifacts alongside the final results.
For every pair of frames a raw difference (`{frame}_diff.png`) is written to `diff/raw/` and its
thresholded mask (`{frame}_bw_diff.png`) to `diff/bw/`. The binary mask is also duplicated in the
`binary/` directory. These files are the same difference maps shown in the UI when using the
**Preview Difference** button.

- `diff/new/` — binary masks highlighting regions that newly appear, used for evaluation.
- `diff/lost/` — binary masks highlighting regions that disappear, used for evaluation.

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
