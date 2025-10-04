from __future__ import annotations

import io
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List

import numpy as np
import pandas as pd

from ..models.config import UvVisParams

logger = logging.getLogger(__name__)


@dataclass
class UvVisSpectrum:
    """Container for a single UV-Vis spectrum."""

    wavelengths: np.ndarray
    absorbance: np.ndarray
    raw_absorbance: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class UvVisDataset:
    """Collection of spectra and references for a UV-Vis experiment."""

    params: UvVisParams
    spectra: List[UvVisSpectrum] = field(default_factory=list)
    blank: UvVisSpectrum | None = None
    dark: UvVisSpectrum | None = None


SUPPORTED_EXTENSIONS = {".csv", ".tsv", ".txt", ".dat", ".jdx", ".dx"}


def load_uvvis_dataset(params: UvVisParams) -> UvVisDataset:
    """Load spectra and references, applying corrections defined by ``params``."""

    blank_path = _resolve_reference_path(params.blank_reference, params)
    dark_path = _resolve_reference_path(params.dark_reference, params)
    exclude = {p.resolve() for p in [blank_path, dark_path] if p is not None}

    files = _resolve_data_files(params, exclude_paths=exclude)
    blank_spec = _load_reference_from_path(blank_path)
    dark_spec = _load_reference_from_path(dark_path)

    spectra: List[UvVisSpectrum] = []
    base_wavelengths: np.ndarray | None = None
    blank_values: np.ndarray | None = None
    dark_values: np.ndarray | None = None

    for file_path in files:
        spectrum = _read_spectrum_file(file_path)
        spectrum.metadata.setdefault("source", str(file_path))
        spectrum.raw_absorbance = spectrum.absorbance.copy()

        if base_wavelengths is None:
            base_wavelengths = spectrum.wavelengths
            if blank_spec is not None:
                blank_spec = _resample_to(blank_spec, base_wavelengths)
                blank_values = blank_spec.absorbance.copy()
            if dark_spec is not None:
                dark_spec = _resample_to(dark_spec, base_wavelengths)
                dark_values = dark_spec.absorbance.copy()
        else:
            spectrum = _resample_to(spectrum, base_wavelengths)

        corrected = spectrum.raw_absorbance.copy()
        if dark_values is not None:
            corrected = corrected - dark_values
        if blank_values is not None:
            corrected = corrected - blank_values

        if params.apply_baseline_correction and len(corrected) > 0:
            corrected = _apply_baseline_correction(base_wavelengths, corrected, params)
        if params.apply_smoothing and params.smoothing_window > 1:
            corrected = _apply_smoothing(corrected, params.smoothing_window)

        spectrum.absorbance = corrected
        spectra.append(spectrum)

    dataset = UvVisDataset(
        params=params,
        spectra=spectra,
        blank=blank_spec,
        dark=dark_spec,
    )
    return dataset


def compute_uvvis_metrics(
    dataset: UvVisDataset,
    params: UvVisParams | None = None,
) -> dict[str, Any]:
    """Compute peaks, areas, and ratios for each spectrum in ``dataset``."""

    params = params or dataset.params
    results: List[dict[str, Any]] = []
    for spectrum in dataset.spectra:
        entry: dict[str, Any] = {"metadata": spectrum.metadata.copy()}
        wavelengths = spectrum.wavelengths
        values = spectrum.absorbance
        if params.enable_peak_metrics:
            peaks = _find_peaks(wavelengths, values, params)
            entry["peaks"] = peaks
        if params.enable_area_metrics:
            entry["area"] = float(np.trapezoid(values, wavelengths))
        if params.enable_ratio_metrics:
            ratio_a, ratio_b = params.as_ratio_tuple()
            ratios = _compute_ratios(wavelengths, values, ratio_a, ratio_b)
            entry["ratios"] = ratios
        results.append(entry)

    return {
        "params": params,
        "spectra": results,
    }


def _resolve_data_files(
    params: UvVisParams,
    *,
    exclude_paths: set[Path] | None = None,
) -> List[Path]:
    candidates: List[Path] = []
    seen: set[Path] = set()

    base_dir = Path(params.data_directory) if params.data_directory else None
    excluded = {p.resolve() for p in exclude_paths} if exclude_paths else set()
    for item in params.data_files:
        if not item:
            continue
        path = Path(item)
        if not path.is_absolute() and base_dir is not None:
            candidate = base_dir / path
            if candidate.exists():
                path = candidate
        if not path.exists():
            raise FileNotFoundError(f"UV-Vis spectrum not found: {path}")
        resolved = path.resolve()
        if resolved in excluded:
            continue
        if resolved not in seen:
            candidates.append(path)
            seen.add(resolved)

    if base_dir and base_dir.is_dir():
        for child in sorted(base_dir.iterdir()):
            if child.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue
            resolved_child = child.resolve()
            if resolved_child in excluded or resolved_child in seen:
                continue
            candidates.append(child)
            seen.add(resolved_child)

    return candidates


def _resolve_reference_path(path_str: str | None, params: UvVisParams) -> Path | None:
    if not path_str:
        return None
    base_dir = Path(params.data_directory) if params.data_directory else None
    path = Path(path_str)
    if not path.is_absolute() and base_dir is not None:
        candidate = base_dir / path
        if candidate.exists():
            path = candidate
    if not path.exists():
        raise FileNotFoundError(f"UV-Vis reference not found: {path}")
    return path


def _load_reference_from_path(path: Path | None) -> UvVisSpectrum | None:
    if path is None:
        return None
    spectrum = _read_spectrum_file(path)
    spectrum.metadata.setdefault("source", str(path))
    spectrum.raw_absorbance = spectrum.absorbance.copy()
    return spectrum


def _read_spectrum_file(path: Path) -> UvVisSpectrum:
    if path.suffix.lower() in {".jdx", ".dx"}:
        return _read_jcamp(path)
    return _read_text_spectrum(path)


def _read_text_spectrum(path: Path) -> UvVisSpectrum:
    with path.open("r", encoding="utf-8", errors="ignore") as fh:
        lines = fh.readlines()

    metadata = {"source": str(path)}
    data_lines: List[str] = []
    data_started = False
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        meta = _parse_metadata_line(stripped)
        if meta:
            metadata.update(meta)
            continue
        if not data_started and _looks_like_data(stripped):
            data_started = True
        if data_started:
            data_lines.append(stripped)

    if not data_lines:
        raise ValueError(f"No spectral data found in {path}")

    df = pd.read_csv(
        io.StringIO("\n".join(data_lines)),
        comment="#",
        sep=None,
        engine="python",
        header=None,
    )
    df = df.apply(pd.to_numeric, errors="coerce")
    numeric_cols = [col for col in df.columns if df[col].notna().sum()]
    if len(numeric_cols) < 2:
        raise ValueError(f"Unable to parse spectral columns in {path}")
    x = df[numeric_cols[0]].to_numpy(dtype=float)
    y = df[numeric_cols[1]].to_numpy(dtype=float)
    mask = ~np.isnan(x) & ~np.isnan(y)
    wavelengths = x[mask]
    absorbance = y[mask]
    order = np.argsort(wavelengths)
    wavelengths = wavelengths[order]
    absorbance = absorbance[order]
    return UvVisSpectrum(wavelengths=wavelengths, absorbance=absorbance, metadata=metadata)


def _read_jcamp(path: Path) -> UvVisSpectrum:
    metadata = {"source": str(path), "format": "JCAMP"}
    wavelengths: list[float] = []
    absorbance: list[float] = []
    xy_mode: str | None = None
    xfactor = 1.0
    yfactor = 1.0
    delta_x = 1.0

    with path.open("r", encoding="utf-8", errors="ignore") as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("##"):
                upper = line.upper()
                if upper.startswith("##XYDATA="):
                    xy_mode = "XYDATA"
                    metadata["XYDATA"] = line.split("=", 1)[1].strip()
                    continue
                if upper.startswith("##XYPOINTS="):
                    xy_mode = "XYPOINTS"
                    metadata["XYPOINTS"] = line.split("=", 1)[1].strip()
                    continue
                if "=" in line[2:]:
                    key, value = line[2:].split("=", 1)
                    key = key.strip()
                    value = value.strip()
                    metadata[key] = value
                    key_upper = key.upper()
                    if key_upper == "XFACTOR":
                        try:
                            xfactor = float(value)
                        except ValueError:
                            pass
                    elif key_upper == "YFACTOR":
                        try:
                            yfactor = float(value)
                        except ValueError:
                            pass
                    elif key_upper in {"DELTAX", "XINCREMENT", "XSTEP"}:
                        try:
                            delta_x = float(value)
                        except ValueError:
                            pass
                continue

            numbers = _extract_numbers(line)
            if not numbers:
                continue
            if xy_mode == "XYDATA" and len(numbers) > 1:
                start_x = numbers[0] * xfactor
                for idx, val in enumerate(numbers[1:]):
                    wavelengths.append(start_x + idx * delta_x * xfactor)
                    absorbance.append(val * yfactor)
            else:
                for i in range(0, len(numbers) - 1, 2):
                    wavelengths.append(numbers[i] * xfactor)
                    absorbance.append(numbers[i + 1] * yfactor)

    if not wavelengths:
        raise ValueError(f"No spectral data found in {path}")

    arr_x = np.asarray(wavelengths, dtype=float)
    arr_y = np.asarray(absorbance, dtype=float)
    order = np.argsort(arr_x)
    arr_x = arr_x[order]
    arr_y = arr_y[order]
    return UvVisSpectrum(wavelengths=arr_x, absorbance=arr_y, metadata=metadata)


def _parse_metadata_line(line: str) -> dict[str, str] | None:
    if any(line.startswith(prefix) for prefix in ("#", ";")):
        return None
    for sep in ("=", ":"):
        if sep in line:
            key, value = line.split(sep, 1)
            key = key.strip()
            value = value.strip()
            if key and not _looks_like_number(key):
                return {key: value}
    return None


def _looks_like_data(line: str) -> bool:
    tokens = line.replace(",", " ").split()
    if len(tokens) < 2:
        return False
    return _looks_like_number(tokens[0]) and _looks_like_number(tokens[1])


def _looks_like_number(text: str) -> bool:
    try:
        float(text)
        return True
    except ValueError:
        return False


def _extract_numbers(line: str) -> List[float]:
    numbers: List[float] = []
    for token in line.replace(",", " ").split():
        try:
            numbers.append(float(token))
        except ValueError:
            continue
    return numbers


def _resample_to(spectrum: UvVisSpectrum, target_wavelengths: np.ndarray) -> UvVisSpectrum:
    if np.array_equal(spectrum.wavelengths, target_wavelengths):
        return spectrum
    orig_wavelengths = spectrum.wavelengths
    orig_absorbance = spectrum.absorbance
    raw = spectrum.raw_absorbance.copy() if spectrum.raw_absorbance is not None else None
    values = np.interp(target_wavelengths, orig_wavelengths, orig_absorbance)
    spectrum.wavelengths = target_wavelengths.copy()
    spectrum.absorbance = values
    if raw is not None:
        spectrum.raw_absorbance = np.interp(
            target_wavelengths,
            orig_wavelengths,
            raw,
        )
    return spectrum


def _apply_baseline_correction(
    wavelengths: np.ndarray,
    values: np.ndarray,
    params: UvVisParams,
) -> np.ndarray:
    order = max(0, min(params.baseline_poly_order, len(wavelengths) - 1))
    if order <= 0:
        baseline = np.full_like(values, np.nanmean(values))
    else:
        try:
            coeffs = np.polyfit(wavelengths, values, order)
            baseline = np.polyval(coeffs, wavelengths)
        except Exception as exc:
            logger.debug("Baseline fit failed (%s); skipping", exc)
            return values
    return values - baseline


def _apply_smoothing(values: np.ndarray, window: int) -> np.ndarray:
    window = max(1, int(window))
    if window % 2 == 0:
        window += 1
    if window <= 1:
        return values
    pad = window // 2
    padded = np.pad(values, (pad, pad), mode="edge")
    kernel = np.ones(window, dtype=float) / window
    smoothed = np.convolve(padded, kernel, mode="valid")
    return smoothed[: len(values)]


def _find_peaks(
    wavelengths: np.ndarray,
    values: np.ndarray,
    params: UvVisParams,
) -> List[dict[str, float]]:
    if len(values) < 3:
        return []
    gradients = np.diff(values)
    signs = np.sign(gradients)
    zero_crossings = np.diff(signs)
    candidate_indices = np.where(zero_crossings < 0)[0] + 1
    peaks: List[dict[str, float]] = []
    prominence = max(params.peak_prominence, 0.0)
    for idx in candidate_indices:
        left = max(0, idx - 1)
        right = min(len(values) - 1, idx + 1)
        peak_height = values[idx]
        local_min = min(values[left], values[right])
        if peak_height - local_min >= prominence:
            peaks.append(
                {
                    "wavelength": float(wavelengths[idx]),
                    "absorbance": float(peak_height),
                }
            )
    peaks.sort(key=lambda p: p["absorbance"], reverse=True)
    if params.top_n_peaks > 0:
        peaks = peaks[: params.top_n_peaks]
    return peaks


def _compute_ratios(
    wavelengths: np.ndarray,
    values: np.ndarray,
    wave_a: float,
    wave_b: float,
) -> dict[str, float]:
    if wave_a == wave_b:
        return {"ratio": 1.0, "a": wave_a, "b": wave_b}
    a_val = float(np.interp(wave_a, wavelengths, values))
    b_val = float(np.interp(wave_b, wavelengths, values))
    ratio = a_val / b_val if b_val != 0 else np.nan
    return {
        "ratio": ratio,
        "a_wavelength": wave_a,
        "a_value": a_val,
        "b_wavelength": wave_b,
        "b_value": b_val,
    }

