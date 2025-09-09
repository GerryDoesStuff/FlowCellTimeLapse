from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Literal, Optional, Dict, Any
import json
from PyQt6.QtCore import QSettings

RegistrationModel = Literal["translation","euclidean","affine","homography"]
SegMethod = Literal["otsu","adaptive","local","manual"]

@dataclass
class RegParams:
    method: str = "ECC"   # "ECC" or "ORB"
    model: RegistrationModel = "affine"
    max_iters: int = 1000
    eps: float = 1e-6
    gauss_blur_sigma: float = 1.0
    clahe_clip: float = 2.0
    clahe_grid: int = 8
    initial_radius: int = 20
    growth_factor: float = 1.0
    use_masked_ecc: bool = True
    orb_features: int = 4000
    match_ratio: float = 0.75

@dataclass
class SegParams:
    method: SegMethod = "otsu"
    manual_thresh: int = 128
    adaptive_block: int = 51
    adaptive_C: int = 5
    local_block: int = 51
    remove_holes_smaller_px: int = 64
    remove_objects_smaller_px: int = 64
    morph_open_radius: int = 2
    morph_close_radius: int = 2
    invert: bool = True  # cells darker

@dataclass
class AppParams:
    px_size_um: float = 1.0
    minutes_between_frames: float = 1.0  # default; can be overridden by timestamps
    direction: str = "last-to-first"  # "last-to-first" | "first-to-last"
    show_ref_overlay: bool = True
    show_mov_overlay: bool = True
    overlay_opacity: int = 50
    save_jpg_quality: int = 95
    save_png: bool = False
    save_intermediates: bool = True
    use_difference_for_seg: bool = False
    use_file_timestamps: bool = True
    presets_path: Optional[str] = None
    last_folder: str | None = None

def save_preset(path: str, reg: RegParams, seg: SegParams, app: AppParams) -> None:
    data = {"reg": asdict(reg), "seg": asdict(seg), "app": asdict(app)}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

def load_preset(path: str) -> tuple[RegParams, SegParams, AppParams]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return RegParams(**data["reg"]), SegParams(**data["seg"]), AppParams(**data["app"])

def save_settings(reg: RegParams, seg: SegParams, app: AppParams) -> None:
    s = QSettings("YeastLab", "FlowcellPyQt")
    s.setValue("reg", json.dumps(asdict(reg)))
    s.setValue("seg", json.dumps(asdict(seg)))
    s.setValue("app", json.dumps(asdict(app)))
    s.sync()

def load_settings() -> tuple[RegParams, SegParams, AppParams]:
    s = QSettings("YeastLab", "FlowcellPyQt")
    reg = s.value("reg"); seg = s.value("seg"); app = s.value("app")
    def parse(v, cls, default):
        if v is None: return default
        try: return cls(**json.loads(v))
        except Exception: return default
    return (
        parse(reg, RegParams, RegParams()),
        parse(seg, SegParams, SegParams()),
        parse(app, AppParams, AppParams()),
    )
