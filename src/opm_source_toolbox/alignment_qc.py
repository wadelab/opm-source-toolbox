"""Render simple sensor-versus-head alignment QC montages for source-imaging workflows."""

from __future__ import annotations

from dataclasses import dataclass
import os
import re
from typing import Optional, Sequence
import warnings

import mne
from mne.io.constants import FIFF
from mne.surface import get_head_surf
from mne.transforms import apply_trans, invert_transform
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import pyvista as pv

from .core import ensure_dir, load_run_info


_SAFE_STEM_RE = re.compile(r"[^A-Za-z0-9._-]+")
_DEFAULT_VIEWS = ("oblique", "side", "top")


@dataclass
class AlignmentQcResult:
    """Describe the PNG outputs written for one alignment QC montage."""

    stem: str
    montage_path: str
    view_paths: dict[str, str]
    title: str


def _sanitize_stem(value: str) -> str:
    """Normalize a label into a filename-safe stem."""

    cleaned = _SAFE_STEM_RE.sub("_", str(value).strip())
    cleaned = cleaned.strip("._")
    return cleaned or "alignment"


def _load_font(size: int) -> ImageFont.ImageFont:
    """Load a reasonable sans-serif font for montage labels."""

    for name in ("DejaVuSans.ttf", "Arial.ttf"):
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _to_pyvista_faces(triangles: np.ndarray) -> np.ndarray:
    """Convert an MNE triangles array into PyVista face encoding."""

    triangles = np.asarray(triangles, dtype=np.int64)
    prefix = np.full((triangles.shape[0], 1), 3, dtype=np.int64)
    return np.hstack([prefix, triangles]).ravel()


def _load_info_for_alignment(
    geometry_info_file: str,
    ch_names: Optional[Sequence[str]] = None,
) -> mne.Info:
    """Load a MEG-only Info object suitable for sensor/head alignment rendering."""

    if ch_names is not None:
        return load_run_info(geometry_info_file, ch_names)

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="This filename .* does not conform to MNE naming conventions.*",
            category=RuntimeWarning,
        )
        info = None
        try:
            info = mne.io.read_info(geometry_info_file, verbose=False)
        except Exception:
            info = None

        if info is None:
            try:
                raw = mne.io.read_raw_fif(geometry_info_file, preload=False, verbose=False)
                info = raw.info.copy()
            except Exception:
                info = None

        if info is None:
            try:
                epochs = mne.read_epochs(geometry_info_file, preload=False, verbose=False)
                info = epochs.info.copy()
            except Exception:
                info = None

        if info is None:
            try:
                evoked = mne.read_evokeds(geometry_info_file, condition=0, verbose=False)
                info = evoked.info.copy()
            except Exception:
                info = None

    if info is None:
        raise RuntimeError(f"Could not read measurement info from FIF file: {geometry_info_file}")

    picks = mne.pick_types(
        info,
        meg=True,
        eeg=False,
        stim=False,
        eog=False,
        ecg=False,
        emg=False,
        misc=False,
        ref_meg=False,
        exclude="bads",
    )
    if len(picks) == 0:
        raise RuntimeError(f"No MEG channels were found in {geometry_info_file}")
    return mne.pick_info(info.copy(), picks.tolist())


def _sensor_positions_in_mri(info: mne.Info, trans_path: str) -> np.ndarray:
    """Transform MEG sensor positions from device/head space into MRI coordinates."""

    if info.get("dev_head_t") is None:
        raise RuntimeError("Info does not contain dev_head_t, so sensor alignment cannot be rendered")

    trans = mne.read_trans(trans_path)
    if trans["from"] == FIFF.FIFFV_COORD_MRI and trans["to"] == FIFF.FIFFV_COORD_HEAD:
        trans = invert_transform(trans)
    if trans["from"] != FIFF.FIFFV_COORD_HEAD or trans["to"] != FIFF.FIFFV_COORD_MRI:
        raise ValueError(
            f"Unsupported transform frames in {trans_path}: from={trans['from']} to={trans['to']}"
        )

    dev_to_head = info["dev_head_t"]["trans"]
    head_to_mri = trans["trans"]
    rr_device = np.asarray([ch["loc"][:3] for ch in info["chs"]], dtype=float)
    rr_head = apply_trans(dev_to_head, rr_device)
    rr_mri = apply_trans(head_to_mri, rr_head)
    return np.asarray(rr_mri, dtype=float)


def render_alignment_screenshot(
    subject_dir: str,
    fs_subject: str,
    info: mne.Info,
    trans_path: str,
    out_path: str,
    view: str = "oblique",
) -> None:
    """Render one off-screen sensor/head alignment screenshot to PNG."""

    pv.OFF_SCREEN = True
    head = get_head_surf(fs_subject, source="head", subjects_dir=subject_dir, verbose=False)
    sensors_mri = _sensor_positions_in_mri(info, trans_path)
    head_mesh = pv.PolyData(head["rr"], _to_pyvista_faces(np.asarray(head["tris"], dtype=int)))

    plotter = pv.Plotter(off_screen=True, window_size=(900, 900))
    plotter.set_background("white")
    plotter.add_mesh(head_mesh, color="#d9d9d9", opacity=0.35, smooth_shading=True)
    plotter.add_points(
        sensors_mri,
        color="#c73b2c",
        point_size=8,
        render_points_as_spheres=True,
    )
    plotter.show_axes()

    if view == "top":
        plotter.view_xy()
        plotter.camera.azimuth += 180.0
    elif view == "side":
        plotter.view_yz()
        plotter.camera.azimuth += 180.0
    elif view == "front":
        plotter.view_xz()
        plotter.camera.azimuth += 180.0
    elif view == "oblique":
        plotter.camera_position = "iso"
        plotter.camera.azimuth += 25.0
        plotter.camera.elevation += 10.0
    else:
        raise ValueError(f"Unsupported alignment QC view: {view}")

    plotter.show(screenshot=out_path, auto_close=True)


def _build_alignment_montage(
    tile_paths: Sequence[str],
    out_path: str,
    title: str,
    view_labels: Sequence[str],
    image_size: int,
) -> None:
    """Assemble several alignment screenshots into one labeled montage PNG."""

    tiles = [Image.open(path).convert("RGB").resize((image_size, image_size)) for path in tile_paths]
    n_tiles = len(tiles)
    gutter = 12
    label_h = 34
    title_h = 42
    canvas = Image.new(
        "RGB",
        (gutter + n_tiles * (image_size + gutter), title_h + label_h + image_size + gutter * 2),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    title_font = _load_font(22)
    body_font = _load_font(16)
    draw.text((gutter, 10), title, fill="black", font=title_font)

    for idx, (tile, label) in enumerate(zip(tiles, view_labels)):
        x = gutter + idx * (image_size + gutter)
        draw.text((x + 8, title_h), label, fill="#333333", font=body_font)
        canvas.paste(tile, (x, title_h + label_h))
        tile.close()

    canvas.save(out_path)


def render_alignment_qc_bundle(
    subject_dir: str,
    fs_subject: str,
    geometry_info_file: str,
    trans_path: str,
    out_dir: str,
    stem: str,
    *,
    ch_names: Optional[Sequence[str]] = None,
    title: Optional[str] = None,
    views: Sequence[str] = _DEFAULT_VIEWS,
    image_size: int = 320,
) -> AlignmentQcResult:
    """Render a multi-view alignment QC bundle and return the written file paths."""

    ensure_dir(out_dir)
    info = _load_info_for_alignment(geometry_info_file, ch_names=ch_names)
    safe_stem = _sanitize_stem(stem)
    title = title or safe_stem

    view_paths: dict[str, str] = {}
    for view in views:
        out_path = os.path.join(out_dir, f"{safe_stem}_{view}.png")
        render_alignment_screenshot(
            subject_dir=subject_dir,
            fs_subject=fs_subject,
            info=info,
            trans_path=trans_path,
            out_path=out_path,
            view=str(view),
        )
        view_paths[str(view)] = out_path

    montage_path = os.path.join(out_dir, f"{safe_stem}_montage.png")
    _build_alignment_montage(
        tile_paths=[view_paths[str(view)] for view in views],
        out_path=montage_path,
        title=title,
        view_labels=[str(view).capitalize() for view in views],
        image_size=int(image_size),
    )
    return AlignmentQcResult(
        stem=safe_stem,
        montage_path=montage_path,
        view_paths=view_paths,
        title=title,
    )
