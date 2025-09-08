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

### Project layout
- `app/main.py` — app entry, sets up MainWindow.
- `app/ui/main_window.py` — PyQt UI and interactions.
- `app/models/config.py` — dataclasses for parameters; JSON presets; QSettings persistence.
- `app/core/io_utils.py` — file discovery, timestamp spacing, safe I/O.
- `app/core/registration.py` — ECC / ORB(+RANSAC) on CPU; CUDA path when available.
- `app/core/segmentation.py` — outline-focused segmentation (black-hat + Otsu/adaptive), morphology.
- `app/core/background.py` — on-the-fly background estimation (temporal median of early frames, fallback to blur).
- `app/core/processing.py` — end-to-end per-frame pipeline; overlap cropping; metrics aggregation.
- `app/core/multiproc.py` — ProcessPool execution with chunking; CPU core detection.
- `app/workers/pipeline_worker.py` — background worker orchestration.
- `requirements.txt` — dependencies.
