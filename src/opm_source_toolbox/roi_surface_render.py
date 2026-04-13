"""Render atlas ROI values onto cortical surfaces and build simple montage views."""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Mapping, Optional, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from nibabel.freesurfer.io import read_annot
from nilearn import plotting, surface
from PIL import Image, ImageDraw

from .core import DEFAULT_ATLAS_PARC, ensure_dir, resolve_atlas_annotation_spec


_VIEW_LAYOUT = (
    ("left", "lateral", "lh"),
    ("right", "lateral", "rh"),
    ("left", "dorsal", "lh"),
    ("right", "dorsal", "rh"),
)


@dataclass
class SurfaceRenderConfig:
    """Store atlas, surface, and color settings for ROI-to-surface rendering."""

    atlas_parc: str = DEFAULT_ATLAS_PARC
    atlas_subject: Optional[str] = None
    atlas_subjects_dir: Optional[str] = None
    surface: str = "inflated"
    cmap: str = "inferno"
    color_mode: str = "auto"
    vmin: Optional[float] = None
    vmax: Optional[float] = None
    threshold: float = 1e-12
    bg_binary_low: float = 0.44
    bg_binary_high: float = 0.56

    def validate(self) -> None:
        """Validate the surface rendering configuration."""

        if self.color_mode not in {"auto", "positive", "symmetric"}:
            raise ValueError(f"Unsupported color_mode: {self.color_mode}")


def _decode_annot_names(names: Sequence[bytes | str]) -> list[str]:
    """Decode nibabel annotation label names into plain Python strings."""

    decoded = []
    for name in names:
        if isinstance(name, bytes):
            decoded.append(name.decode("utf-8"))
        else:
            decoded.append(str(name))
    return decoded


def _load_atlas_surface_spec(config: SurfaceRenderConfig) -> dict[str, object]:
    """Resolve mesh and annotation files for the requested atlas surface."""

    config.validate()
    if config.atlas_subjects_dir:
        subjects_root = config.atlas_subjects_dir
    else:
        subjects_root = "/raid/toolbox/freesurfer/subjects"
    spec = resolve_atlas_annotation_spec(
        subject_dir=subjects_root,
        fs_subject=config.atlas_subject or "fsaverage",
        atlas_parc=config.atlas_parc,
        atlas_subject=config.atlas_subject,
        atlas_subjects_dir=config.atlas_subjects_dir,
    )
    atlas_subject_name = str(spec["source_subject"])
    atlas_root = str(spec["atlas_subjects_dir"])
    surf_root = os.path.join(atlas_root, atlas_subject_name, "surf")
    if not os.path.isdir(surf_root):
        raise FileNotFoundError(f"Atlas surface directory not found: {surf_root}")
    return {
        "atlas_subject": atlas_subject_name,
        "subjects_dir": atlas_root,
        "surf_root": surf_root,
        "annot_paths": list(spec["atlas_paths"]),
    }


def _surface_value_map_from_annot(
    annot_path: str,
    hemi: str,
    value_by_name: Mapping[str, float],
) -> np.ndarray:
    """Expand ROI label values into a vertex-wise surface array for one hemisphere."""

    labels, _ctab, names = read_annot(annot_path)
    names_decoded = _decode_annot_names(names)
    values = np.zeros(labels.shape, dtype=float)
    hemi_suffix = f"-{hemi}"
    for label_idx, name in enumerate(names_decoded):
        value = value_by_name.get(name)
        if value is None:
            value = value_by_name.get(f"{name}{hemi_suffix}")
        if value is None or not np.isfinite(value):
            continue
        values[labels == int(label_idx)] = float(value)
    return values


def _load_binarized_bg_map(
    bg_map_path: str,
    low_value: float,
    high_value: float,
) -> np.ndarray:
    """Convert a continuous sulcal depth map into a simple light/dark background."""

    bg_map = np.asarray(surface.load_surf_data(bg_map_path), dtype=float)
    if bg_map.ndim != 1:
        raise ValueError(f"Expected 1D surface background map, got shape {bg_map.shape}")

    finite = np.isfinite(bg_map)
    if not np.any(finite):
        return np.full(bg_map.shape, 0.5, dtype=float)

    finite_values = bg_map[finite]
    if float(np.nanmin(finite_values)) < 0.0 and float(np.nanmax(finite_values)) > 0.0:
        threshold = 0.0
    else:
        threshold = float(np.nanmedian(finite_values))

    binary_map = np.full(bg_map.shape, 0.5, dtype=float)
    binary_map[finite & (bg_map <= threshold)] = float(low_value)
    binary_map[finite & (bg_map > threshold)] = float(high_value)
    return binary_map


def _resolve_color_limits(
    surface_maps: Mapping[str, np.ndarray],
    config: SurfaceRenderConfig,
) -> tuple[float, float, bool]:
    """Choose colorbar limits that fit the provided ROI value distribution."""

    finite_chunks = []
    for values in surface_maps.values():
        arr = np.asarray(values, dtype=float).ravel()
        arr = arr[np.isfinite(arr)]
        if arr.size:
            finite_chunks.append(arr)
    if not finite_chunks:
        return 0.0, 1.0, False

    finite = np.concatenate(finite_chunks, axis=0)
    finite = finite[~np.isclose(finite, 0.0)]
    if finite.size == 0:
        return 0.0, 1.0, False

    color_mode = config.color_mode
    if color_mode == "auto":
        color_mode = "symmetric" if np.nanmin(finite) < 0.0 else "positive"

    if config.vmin is not None and config.vmax is not None:
        return float(config.vmin), float(config.vmax), color_mode == "symmetric"

    if color_mode == "symmetric":
        vmax = float(np.nanpercentile(np.abs(finite), 99.0))
        if not np.isfinite(vmax) or vmax <= 0.0:
            vmax = 1.0
        vmin = -vmax if config.vmin is None else float(config.vmin)
        vmax = vmax if config.vmax is None else float(config.vmax)
        return float(vmin), float(vmax), True

    positive = finite[finite > 0.0]
    if positive.size == 0:
        vmax = 1.0
    else:
        vmax = float(np.nanpercentile(positive, 99.0))
        if not np.isfinite(vmax) or vmax <= 0.0:
            vmax = 1.0
    vmin = 0.0 if config.vmin is None else float(config.vmin)
    vmax = vmax if config.vmax is None else float(config.vmax)
    return float(vmin), float(vmax), False


def _render_tile_png(
    mesh_path: str,
    bg_map: np.ndarray,
    surface_values: np.ndarray,
    hemi: str,
    view: str,
    title: str,
    out_path: str,
    config: SurfaceRenderConfig,
    vmin: float,
    vmax: float,
    symmetric_cbar: bool,
) -> None:
    """Render one hemisphere/view tile to PNG."""

    display = plotting.plot_surf_stat_map(
        surf_mesh=mesh_path,
        stat_map=np.asarray(surface_values, dtype=float),
        bg_map=np.asarray(bg_map, dtype=float),
        hemi=hemi,
        view=view,
        engine="matplotlib",
        cmap=config.cmap,
        colorbar=True,
        bg_on_data=True,
        symmetric_cbar=bool(symmetric_cbar),
        threshold=float(config.threshold),
        vmin=float(vmin),
        vmax=float(vmax),
        title=title,
        output_file=out_path,
    )
    if hasattr(display, "close"):
        display.close()
    plt.close("all")


def _build_montage(tile_paths: Sequence[str], out_path: str, header_text: str) -> None:
    """Assemble the four default surface views into one 2x2 montage PNG."""

    tiles = [Image.open(path).convert("RGB") for path in tile_paths]
    tile_w = max(tile.width for tile in tiles)
    tile_h = max(tile.height for tile in tiles)
    header_h = 70
    canvas = Image.new("RGB", (tile_w * 2, header_h + (tile_h * 2)), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((20, 18), header_text, fill="black")
    positions = (
        (0, header_h),
        (tile_w, header_h),
        (0, header_h + tile_h),
        (tile_w, header_h + tile_h),
    )
    for tile, pos in zip(tiles, positions):
        canvas.paste(tile.resize((tile_w, tile_h)), pos)
        tile.close()
    canvas.save(out_path)


def load_roi_value_map_csv(
    csv_path: str,
    roi_name_col: str = "roi_name",
    value_col: str = "value",
) -> dict[str, float]:
    """Load a simple ROI-name/value CSV into a dictionary."""

    df = pd.read_csv(csv_path)
    if roi_name_col not in df.columns:
        raise KeyError(f"Missing ROI name column '{roi_name_col}' in {csv_path}")
    if value_col not in df.columns:
        raise KeyError(f"Missing value column '{value_col}' in {csv_path}")
    values = pd.to_numeric(df[value_col], errors="coerce")
    out: dict[str, float] = {}
    for roi_name, value in zip(df[roi_name_col], values):
        if pd.isna(value):
            continue
        out[str(roi_name)] = float(value)
    if not out:
        raise RuntimeError(f"No finite ROI values loaded from {csv_path}")
    return out


def render_roi_vector_to_surface(
    roi_names: Sequence[str],
    values: Sequence[float],
    out_dir: str,
    stem: str,
    title: str,
    config: Optional[SurfaceRenderConfig] = None,
) -> dict[str, object]:
    """Render one ROI vector onto fsaverage-style surfaces and write a montage."""

    if len(roi_names) != len(values):
        raise ValueError(
            f"ROI name/value length mismatch: {len(roi_names)} names, {len(values)} values"
        )

    value_by_name: dict[str, float] = {}
    for roi_name, value in zip(roi_names, values):
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(numeric):
            continue
        value_by_name[str(roi_name)] = numeric
    if not value_by_name:
        raise RuntimeError("No finite ROI values were supplied for rendering")

    return render_roi_value_map_to_surface(
        value_by_name=value_by_name,
        out_dir=out_dir,
        stem=stem,
        title=title,
        config=config,
    )


def render_roi_value_map_to_surface(
    value_by_name: Mapping[str, float],
    out_dir: str,
    stem: str,
    title: str,
    config: Optional[SurfaceRenderConfig] = None,
) -> dict[str, object]:
    """Render an ROI-name/value mapping onto the atlas surface and save PNG outputs."""

    config = config or SurfaceRenderConfig()
    config.validate()
    ensure_dir(out_dir)

    atlas_spec = _load_atlas_surface_spec(config)
    annot_paths = {
        "lh": next(
            path for path in atlas_spec["annot_paths"] if os.path.basename(path).startswith("lh.")
        ),
        "rh": next(
            path for path in atlas_spec["annot_paths"] if os.path.basename(path).startswith("rh.")
        ),
    }
    mesh_paths = {
        hemi: os.path.join(str(atlas_spec["surf_root"]), f"{hemi}.{config.surface}")
        for hemi in ("lh", "rh")
    }
    bg_maps = {
        hemi: _load_binarized_bg_map(
            os.path.join(str(atlas_spec["surf_root"]), f"{hemi}.sulc"),
            low_value=float(config.bg_binary_low),
            high_value=float(config.bg_binary_high),
        )
        for hemi in ("lh", "rh")
    }
    surface_maps = {
        hemi: _surface_value_map_from_annot(
            annot_paths[hemi],
            hemi=hemi,
            value_by_name=value_by_name,
        )
        for hemi in ("lh", "rh")
    }
    vmin, vmax, symmetric_cbar = _resolve_color_limits(surface_maps, config)

    tile_paths: list[str] = []
    for hemi_name, view, hemi in _VIEW_LAYOUT:
        tile_path = os.path.join(out_dir, f"{stem}_{hemi_name}_{view}.png")
        _render_tile_png(
            mesh_path=mesh_paths[hemi],
            bg_map=bg_maps[hemi],
            surface_values=surface_maps[hemi],
            hemi=hemi_name,
            view=view,
            title=f"{hemi.upper()} {view}",
            out_path=tile_path,
            config=config,
            vmin=vmin,
            vmax=vmax,
            symmetric_cbar=symmetric_cbar,
        )
        tile_paths.append(tile_path)

    montage_path = os.path.join(out_dir, f"{stem}_montage.png")
    header_text = (
        f"{title} | {config.atlas_parc} | {config.surface} | "
        f"vmin={vmin:.2f} vmax={vmax:.2f}"
    )
    _build_montage(tile_paths=tile_paths, out_path=montage_path, header_text=header_text)
    return {
        "tile_paths": tile_paths,
        "montage_path": montage_path,
        "vmin": vmin,
        "vmax": vmax,
        "symmetric_cbar": symmetric_cbar,
        "atlas_subject": atlas_spec["atlas_subject"],
    }
