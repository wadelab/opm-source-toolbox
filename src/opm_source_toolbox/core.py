"""Shared generic helpers for source projection, atlas handling, and ROI export."""

from __future__ import annotations

import contextlib
import glob
import json
import os
import re
import tempfile
import warnings
from typing import Iterable, Iterator, List, Optional, Sequence, Tuple

import mne
import nibabel.freesurfer.io as fsio
import numpy as np
import pandas as pd
from mne.beamformer import apply_lcmv, make_lcmv
from mne.bem import _bem_find_surface
from mne.forward._make_forward import _prep_meg_channels
from mne.io.constants import FIFF
from mne.surface import _CheckInside
from mne.transforms import apply_trans, invert_transform


DEFAULT_ATLAS_PARC = "Schaefer2018_200Parcels_7Networks_order"
DEFAULT_SOMATOSENSORY_PATTERNS = (
    "sommot",
    "somatomotor",
    "somatosensory",
    "postcentral",
    "precentral",
    "paracentral",
    "ba1",
    "ba2",
    "ba3",
    "ba4",
)
EXCLUDED_ATLAS_LABEL_PATTERNS = (
    "unknown",
    "medial_wall",
    "medial wall",
    "background+freesurfer_defined_medial_wall",
)


def _user_atlas_subjects_dir() -> str:
    cache_root = os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache")
    return os.path.join(cache_root, "opm_source_toolbox", "subjects")


def _channel_key(name: str) -> str:
    name = str(name).strip()
    if "_" in name:
        name = name.split("_", 1)[0]
    return name


def load_matrix_csv(path: str, name_col: str) -> Tuple[List[str], np.ndarray]:
    df = pd.read_csv(path)
    time_cols = sorted(col for col in df.columns if col.startswith("t"))
    names = df[name_col].astype(str).tolist()
    data = df[time_cols].to_numpy(dtype=float)
    return names, data


def write_matrix_csv(path: str, name_col: str, names: Sequence[str], data: np.ndarray) -> None:
    data = np.asarray(data, dtype=float)
    cols = {f"t{idx:03d}": data[:, idx] for idx in range(data.shape[1])}
    df = pd.DataFrame({name_col: list(names), **cols})
    df.to_csv(path, index=False)


def load_matrix_npz(path: str, name_col: str) -> Tuple[List[str], np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        stored_name_col = str(payload["name_col"].item())
        if stored_name_col != name_col:
            raise KeyError(
                f"NPZ matrix {path} stores '{stored_name_col}' instead of '{name_col}'"
            )
        names = payload["names"].astype(str).tolist()
        data = np.asarray(payload["data"], dtype=float)
    if data.ndim != 2:
        raise ValueError(f"Expected a 2D matrix in {path}, got shape {data.shape}")
    if len(names) != data.shape[0]:
        raise ValueError(
            f"Matrix/name mismatch in {path}: {len(names)} names for {data.shape[0]} rows"
        )
    return names, data


def write_matrix_npz(path: str, name_col: str, names: Sequence[str], data: np.ndarray) -> None:
    data = np.asarray(data, dtype=float)
    np.savez_compressed(
        path,
        name_col=np.asarray(name_col),
        names=np.asarray(list(names), dtype=str),
        data=data,
    )


def resolve_subject_anatomy(
    coreg_dir: str,
    subject: str,
    fs_subject: str,
) -> Tuple[str, str]:
    subject_dir = os.path.join(coreg_dir, subject)
    fs_dir = os.path.join(subject_dir, fs_subject)
    if not os.path.isdir(subject_dir):
        raise FileNotFoundError(f"Missing co-registration directory for {subject}: {subject_dir}")
    if not os.path.isdir(fs_dir):
        raise FileNotFoundError(f"Missing FreeSurfer directory for {subject}: {fs_dir}")
    return subject_dir, fs_dir


def resolve_trans_path(subject_dir: str, source_file: str) -> str:
    base = os.path.basename(source_file)
    if base.startswith("preprocessed_"):
        base = base[len("preprocessed_") :]

    direct = os.path.join(subject_dir, base.replace(".fif", "_trans.fif"))
    if os.path.exists(direct):
        return direct

    match = re.search(r"_Run(\d+)", base)
    if match:
        run_number = int(match.group(1))
        matches = sorted(glob.glob(os.path.join(subject_dir, f"*Run{run_number:02d}_trans.fif")))
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise RuntimeError(
                f"Multiple trans files matched run {run_number} for {source_file}: {matches}"
            )

    subject_level_matches = sorted(glob.glob(os.path.join(subject_dir, "*_trans.fif")))
    if len(subject_level_matches) == 1:
        return subject_level_matches[0]
    if len(subject_level_matches) > 1:
        raise RuntimeError(
            f"Could not resolve run-specific trans for {source_file}, and multiple subject-level "
            f"trans files exist: {subject_level_matches}"
        )

    raise FileNotFoundError(f"Could not resolve trans file for source file {source_file}")


def resolve_geometry_info_file(subject_dir: str, source_file: str) -> str:
    base = os.path.basename(source_file)
    if base.startswith("preprocessed_"):
        base = base[len("preprocessed_") :]

    candidate = os.path.join(subject_dir, base)
    if os.path.exists(candidate):
        return candidate
    return source_file


def _find_channel_picks(all_ch_names: Sequence[str], ch_names: Sequence[str]) -> List[int]:
    name_to_idx = {name: idx for idx, name in enumerate(all_ch_names)}
    if all(name in name_to_idx for name in ch_names):
        return [name_to_idx[name] for name in ch_names]

    key_to_idx: dict[str, int] = {}
    for idx, name in enumerate(all_ch_names):
        key = _channel_key(name)
        if key not in key_to_idx:
            key_to_idx[key] = idx

    missing = [name for name in ch_names if _channel_key(name) not in key_to_idx]
    if missing:
        raise RuntimeError(f"Failed to match exported channels back to FIF header: {missing[:8]}")

    return [key_to_idx[_channel_key(name)] for name in ch_names]


def load_run_info(source_file: str, ch_names: Sequence[str]) -> mne.Info:
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="This filename .* does not conform to MNE naming conventions.*",
            category=RuntimeWarning,
        )
        info = None
        try:
            # Prefer the lightweight info reader because it works for FIF containers
            # beyond Raw files and avoids loading any data samples.
            info = mne.io.read_info(source_file, verbose=False)
        except Exception:
            info = None

        if info is None:
            try:
                raw = mne.io.read_raw_fif(source_file, preload=False, verbose=False)
                info = raw.info.copy()
            except Exception:
                info = None

        if info is None:
            try:
                epochs = mne.read_epochs(source_file, preload=False, verbose=False)
                info = epochs.info.copy()
            except Exception:
                info = None

        if info is None:
            try:
                evoked = mne.read_evokeds(source_file, condition=0, verbose=False)
                info = evoked.info.copy()
            except Exception:
                info = None

    if info is None:
        raise RuntimeError(f"Could not read measurement info from FIF file: {source_file}")

    picks = _find_channel_picks(info["ch_names"], ch_names)
    return mne.pick_info(info.copy(), picks)


def build_condition_noise_covariance(
    X: np.ndarray,
    info: mne.Info,
    quiet_conditions: Sequence[int] = (2, 3),
    ridge_fraction: float = 1e-6,
    fallback_kind: str = "adhoc",
) -> mne.Covariance:
    quiet_idx = _select_condition_indices(X.shape[0], quiet_conditions)
    return _build_condition_covariance(
        X=X,
        info=info,
        condition_indices=quiet_idx,
        ridge_fraction=ridge_fraction,
        fallback_kind=fallback_kind,
        failure_context="quiet conditions",
    )


def build_identity_noise_covariance(
    info: mne.Info,
    scale: float = 1.0,
) -> mne.Covariance:
    n_channels = len(info["ch_names"])
    cov = np.eye(n_channels, dtype=float) * float(scale)
    return mne.Covariance(
        cov,
        list(info["ch_names"]),
        list(info["bads"]),
        info["projs"],
        1,
    )


def build_empirical_covariance_from_data(
    data: np.ndarray,
    info: mne.Info,
    ridge_fraction: float = 1e-6,
    fallback_kind: str = "raise",
) -> mne.Covariance:
    data = np.asarray(data, dtype=float)
    if data.ndim == 2:
        data = data[np.newaxis, :, :]
    elif data.ndim != 3:
        raise ValueError(
            f"Expected data with shape (n_channels, n_times) or "
            f"(n_items, n_channels, n_times), got {data.shape}"
        )

    return _build_condition_covariance(
        X=data,
        info=info,
        condition_indices=list(range(int(data.shape[0]))),
        ridge_fraction=ridge_fraction,
        fallback_kind=fallback_kind,
        failure_context="provided sensor data",
    )


def _select_condition_indices(
    n_conditions: int,
    condition_codes: Optional[Sequence[int]],
) -> List[int]:
    if condition_codes is None:
        return list(range(int(n_conditions)))
    wanted = {int(cond) for cond in condition_codes}
    return [idx for idx in range(int(n_conditions)) if ((idx % 5) + 1) in wanted]


def _build_condition_covariance(
    X: np.ndarray,
    info: mne.Info,
    condition_indices: Sequence[int],
    ridge_fraction: float,
    fallback_kind: str,
    failure_context: str,
) -> mne.Covariance:
    if condition_indices:
        cond_data = X[list(condition_indices)].transpose(1, 0, 2).reshape(X.shape[1], -1)
        valid = np.all(np.isfinite(cond_data), axis=0)
        cond_data = cond_data[:, valid]
        if cond_data.shape[1] > X.shape[1]:
            cov = np.cov(cond_data)
            if np.all(np.isfinite(cov)):
                trace = float(np.trace(cov))
                if trace > 0.0:
                    cov = cov + np.eye(cov.shape[0]) * (trace / cov.shape[0]) * ridge_fraction
                return mne.Covariance(
                    cov,
                    list(info["ch_names"]),
                    list(info["bads"]),
                    info["projs"],
                    max(1, cond_data.shape[1] - 1),
                )

    if fallback_kind == "adhoc":
        return mne.make_ad_hoc_cov(info, verbose=False)
    if fallback_kind == "raise":
        raise RuntimeError(f"Failed to compute empirical covariance from {failure_context}")
    raise ValueError(f"Unsupported covariance fallback kind: {fallback_kind}")


def build_condition_data_covariance(
    X: np.ndarray,
    info: mne.Info,
    active_conditions: Optional[Sequence[int]] = None,
    ridge_fraction: float = 1e-6,
) -> mne.Covariance:
    active_idx = _select_condition_indices(X.shape[0], active_conditions)
    return _build_condition_covariance(
        X=X,
        info=info,
        condition_indices=active_idx,
        ridge_fraction=ridge_fraction,
        fallback_kind="raise",
        failure_context="selected data conditions",
    )


def setup_subject_source_space(
    subject_dir: str,
    fs_subject: str,
    spacing: str,
) -> mne.SourceSpaces:
    return mne.setup_source_space(
        fs_subject,
        spacing=spacing,
        add_dist=False,
        subjects_dir=subject_dir,
        verbose=False,
    )


def find_bem_solution(subject_dir: str, fs_subject: str) -> Optional[str]:
    patterns = (
        os.path.join(subject_dir, "*-bem-sol.fif"),
        os.path.join(subject_dir, fs_subject, "bem", "*-bem-sol.fif"),
    )
    matches: List[str] = []
    for pattern in patterns:
        matches.extend(glob.glob(pattern))
    matches = sorted(set(matches))
    if not matches:
        return None
    return matches[0]


def resolve_head_to_mri_transform(trans_path: str) -> mne.transforms.Transform:
    trans = mne.read_trans(trans_path)
    if trans["from"] == FIFF.FIFFV_COORD_HEAD and trans["to"] == FIFF.FIFFV_COORD_MRI:
        return trans
    if trans["from"] == FIFF.FIFFV_COORD_MRI and trans["to"] == FIFF.FIFFV_COORD_HEAD:
        return invert_transform(trans)
    raise ValueError(
        f"Unsupported transform frames in {trans_path}: from={trans['from']} to={trans['to']}"
    )


def find_bem_surface_paths(subject_dir: str, fs_subject: str) -> dict[str, str]:
    bem_dir = os.path.join(subject_dir, fs_subject, "bem")
    return {
        "bem_dir": bem_dir,
        "inner_skull": os.path.join(bem_dir, "inner_skull.surf"),
        "outer_skull": os.path.join(bem_dir, "outer_skull.surf"),
        "outer_skin": os.path.join(bem_dir, "outer_skin.surf"),
        "head": os.path.join(bem_dir, f"{fs_subject}-head.fif"),
        "bem_surfaces": os.path.join(bem_dir, f"{fs_subject}-1layer-bem.fif"),
        "bem_solution": os.path.join(bem_dir, f"{fs_subject}-1layer-bem-sol.fif"),
    }


def find_meg_sensors_inside_inner_skull(
    info: mne.Info,
    trans_path: str,
    subject_dir: str,
    fs_subject: str,
) -> List[str]:
    bem_path = find_bem_solution(subject_dir, fs_subject)
    if not bem_path:
        return []

    bem = mne.read_bem_solution(bem_path, verbose=False)
    if bem.get("is_sphere", False):
        return []

    sensors = _prep_meg_channels(info)
    meg_loc_head = np.asarray([coil["r0"] for coil in sensors["defs"]], dtype=float)
    meg_loc_mri = apply_trans(resolve_head_to_mri_transform(trans_path), meg_loc_head)
    inside = _CheckInside(_bem_find_surface(bem, "inner_skull"))(meg_loc_mri)
    return [name for name, is_inside in zip(sensors["ch_names"], inside) if is_inside]


def default_atlas_subjects_dir(
    atlas_subjects_dir: Optional[str] = None,
    *,
    create: bool = False,
) -> Optional[str]:
    if create:
        target = (
            atlas_subjects_dir
            or os.environ.get("OPM_SOURCE_ATLAS_SUBJECTS_DIR")
            or _user_atlas_subjects_dir()
        )
        os.makedirs(target, exist_ok=True)
        return target

    candidates = [
        atlas_subjects_dir,
        os.environ.get("OPM_SOURCE_ATLAS_SUBJECTS_DIR"),
        os.environ.get("SUBJECTS_DIR"),
    ]
    freesurfer_home = os.environ.get("FREESURFER_HOME")
    if freesurfer_home:
        candidates.append(os.path.join(freesurfer_home, "subjects"))
    candidates.append(_user_atlas_subjects_dir())
    candidates.append("/raid/toolbox/freesurfer/subjects")

    for candidate in candidates:
        if candidate and os.path.isdir(candidate):
            return candidate
    return None


def _default_atlas_subjects_dir(atlas_subjects_dir: Optional[str]) -> Optional[str]:
    return default_atlas_subjects_dir(atlas_subjects_dir, create=False)


def _find_packaged_annotation_paths(atlas_parc: str) -> List[str]:
    atlas_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "schaefer")
    paths = [
        os.path.join(atlas_dir, f"lh.{atlas_parc}.annot"),
        os.path.join(atlas_dir, f"rh.{atlas_parc}.annot"),
    ]
    if all(os.path.exists(path) for path in paths):
        return paths
    return []


def _infer_atlas_subject_from_annotation(annotation_path: str) -> Optional[str]:
    try:
        labels, _ctab, _names = fsio.read_annot(annotation_path)
    except Exception:
        return None

    subject_by_n_vertices = {
        163842: "fsaverage",
        40962: "fsaverage6",
        10242: "fsaverage5",
        2562: "fsaverage4",
        642: "fsaverage3",
    }
    return subject_by_n_vertices.get(int(len(labels)))


def resolve_atlas_annotation_spec(
    subject_dir: str,
    fs_subject: str,
    atlas_parc: str,
    atlas_subject: Optional[str],
    atlas_subjects_dir: Optional[str],
) -> dict:
    source_subject = atlas_subject or fs_subject
    label_root = subject_dir if source_subject == fs_subject else _default_atlas_subjects_dir(atlas_subjects_dir)
    label_dir = os.path.join(label_root, source_subject, "label") if label_root and source_subject != fs_subject else os.path.join(subject_dir, fs_subject, "label")
    direct_paths = [
        os.path.join(label_dir, f"lh.{atlas_parc}.annot"),
        os.path.join(label_dir, f"rh.{atlas_parc}.annot"),
    ]
    if all(os.path.exists(path) for path in direct_paths):
        return {
            "source_subject": source_subject,
            "atlas_paths": direct_paths,
            "atlas_subjects_dir": label_root if source_subject != fs_subject else subject_dir,
            "uses_packaged_annotations": False,
        }

    packaged_paths = _find_packaged_annotation_paths(atlas_parc)
    if packaged_paths:
        packaged_subject = atlas_subject or _infer_atlas_subject_from_annotation(packaged_paths[0])
        packaged_subjects_dir = _default_atlas_subjects_dir(atlas_subjects_dir)
        if (
            packaged_subject
            and packaged_subjects_dir
            and os.path.isdir(os.path.join(packaged_subjects_dir, packaged_subject))
        ):
            return {
                "source_subject": packaged_subject,
                "atlas_paths": packaged_paths,
                "atlas_subjects_dir": packaged_subjects_dir,
                "uses_packaged_annotations": True,
            }

    return {
        "source_subject": source_subject,
        "atlas_paths": direct_paths,
        "atlas_subjects_dir": label_root if source_subject != fs_subject else subject_dir,
        "uses_packaged_annotations": False,
    }


def resolve_atlas_annotation_paths(
    subject_dir: str,
    fs_subject: str,
    atlas_parc: str,
    atlas_subject: Optional[str],
    atlas_subjects_dir: Optional[str],
) -> Tuple[str, List[str]]:
    spec = resolve_atlas_annotation_spec(
        subject_dir=subject_dir,
        fs_subject=fs_subject,
        atlas_parc=atlas_parc,
        atlas_subject=atlas_subject,
        atlas_subjects_dir=atlas_subjects_dir,
    )
    return str(spec["source_subject"]), list(spec["atlas_paths"])


def build_conductor_model(
    subject_dir: str,
    fs_subject: str,
    kind: str,
    sphere_origin: Sequence[float],
    sphere_head_radius: float,
):
    if kind not in {"auto", "bem", "sphere"}:
        raise ValueError(f"Unsupported conductor kind: {kind}")

    bem_path = find_bem_solution(subject_dir, fs_subject)
    if bem_path and kind in {"auto", "bem"}:
        return bem_path, "bem"
    if kind == "bem":
        raise FileNotFoundError(
            f"Requested BEM conductor, but no *-bem-sol.fif was found under {subject_dir}"
        )

    sphere = mne.make_sphere_model(
        r0=tuple(float(x) for x in sphere_origin),
        head_radius=float(sphere_head_radius),
        verbose=False,
    )
    return sphere, "sphere"


@contextlib.contextmanager
def _merged_subjects_dir(
    subject_dir: str,
    fs_subject: str,
    atlas_subject: Optional[str],
    atlas_subjects_dir: Optional[str],
    atlas_annotation_paths: Optional[Sequence[str]] = None,
    use_annotation_overlay: bool = False,
) -> Iterator[str]:
    if atlas_subject in (None, fs_subject) and not use_annotation_overlay:
        yield subject_dir
        return

    atlas_subjects_dir = _default_atlas_subjects_dir(atlas_subjects_dir)
    if not atlas_subjects_dir:
        raise RuntimeError(
            "Morphing atlas labels requires --atlas-subjects-dir or SUBJECTS_DIR"
        )

    atlas_path = os.path.join(atlas_subjects_dir, atlas_subject)
    fs_path = os.path.join(subject_dir, fs_subject)
    if not os.path.isdir(atlas_path):
        raise FileNotFoundError(f"Atlas subject not found: {atlas_path}")
    if not os.path.isdir(fs_path):
        raise FileNotFoundError(f"Subject FreeSurfer directory not found: {fs_path}")

    with tempfile.TemporaryDirectory(prefix="opm-source-subjects-", dir="/tmp") as tmpdir:
        os.symlink(fs_path, os.path.join(tmpdir, fs_subject))
        atlas_tmp = os.path.join(tmpdir, atlas_subject)
        if use_annotation_overlay:
            os.makedirs(atlas_tmp, exist_ok=True)
            for name in os.listdir(atlas_path):
                src = os.path.join(atlas_path, name)
                dst = os.path.join(atlas_tmp, name)
                if name == "label":
                    os.makedirs(dst, exist_ok=True)
                    for label_name in os.listdir(src):
                        label_src = os.path.join(src, label_name)
                        label_dst = os.path.join(dst, label_name)
                        if not os.path.lexists(label_dst):
                            os.symlink(label_src, label_dst)
                else:
                    os.symlink(src, dst)
            label_tmp = os.path.join(atlas_tmp, "label")
            os.makedirs(label_tmp, exist_ok=True)
            for annotation_path in atlas_annotation_paths or ():
                dst = os.path.join(label_tmp, os.path.basename(annotation_path))
                if os.path.lexists(dst):
                    os.remove(dst)
                os.symlink(os.path.abspath(annotation_path), dst)
        else:
            os.symlink(atlas_path, atlas_tmp)
        yield tmpdir


def load_atlas_labels(
    subject_dir: str,
    fs_subject: str,
    atlas_parc: str,
    atlas_subject: Optional[str],
    atlas_subjects_dir: Optional[str],
) -> List[mne.Label]:
    atlas_spec = resolve_atlas_annotation_spec(
        subject_dir=subject_dir,
        fs_subject=fs_subject,
        atlas_parc=atlas_parc,
        atlas_subject=atlas_subject,
        atlas_subjects_dir=atlas_subjects_dir,
    )
    source_subject = str(atlas_spec["source_subject"])
    with _merged_subjects_dir(
        subject_dir,
        fs_subject,
        source_subject,
        atlas_spec["atlas_subjects_dir"],
        atlas_annotation_paths=atlas_spec["atlas_paths"],
        use_annotation_overlay=bool(atlas_spec["uses_packaged_annotations"]),
    ) as work_dir:
        try:
            labels = mne.read_labels_from_annot(
                source_subject,
                parc=atlas_parc,
                hemi="both",
                subjects_dir=work_dir,
                sort=True,
                verbose=False,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load atlas '{atlas_parc}' for subject '{source_subject}'. "
                "Make sure the annotation exists locally, or provide a compatible "
                "atlas subject for morphing."
            ) from exc

        labels = [
            label
            for label in labels
            if not any(pattern in label.name.lower() for pattern in EXCLUDED_ATLAS_LABEL_PATTERNS)
        ]
        if source_subject != fs_subject:
            labels = mne.morph_labels(
                labels,
                subject_to=fs_subject,
                subject_from=source_subject,
                subjects_dir=work_dir,
                verbose=False,
            )
        return labels


def build_run_inverse_operator(
    source_file: str,
    trans_path: str,
    ch_names: Sequence[str],
    src: mne.SourceSpaces,
    noise_cov: mne.Covariance,
    data_cov: Optional[mne.Covariance],
    subject_dir: str,
    fs_subject: str,
    inverse_kind: str,
    loose: float,
    depth: float,
    conductor_kind: str,
    sphere_origin: Sequence[float],
    sphere_head_radius: float,
    beamformer_reg: float = 0.05,
    beamformer_pick_ori: Optional[str] = "normal",
    beamformer_weight_norm: Optional[str] = "unit-noise-gain-invariant",
    beamformer_depth: Optional[float] = None,
) -> Tuple[mne.Info, dict]:
    geometry_info_file = resolve_geometry_info_file(subject_dir, source_file)
    info = load_run_info(geometry_info_file, ch_names)
    conductor, resolved_conductor = build_conductor_model(
        subject_dir=subject_dir,
        fs_subject=fs_subject,
        kind=conductor_kind,
        sphere_origin=sphere_origin,
        sphere_head_radius=sphere_head_radius,
    )
    fwd = mne.make_forward_solution(
        info,
        trans=trans_path,
        src=src,
        bem=conductor,
        meg=True,
        eeg=False,
        n_jobs=1,
        verbose=False,
    )

    if inverse_kind == "mne":
        inverse_operator = mne.minimum_norm.make_inverse_operator(
            info,
            fwd,
            noise_cov,
            loose=float(loose),
            depth=float(depth),
            fixed=False,
            verbose=False,
        )
        return info, {
            "kind": "mne",
            "inverse_operator": inverse_operator,
            "forward": fwd,
            "conductor_kind": resolved_conductor,
        }

    if inverse_kind == "lcmv":
        if data_cov is None:
            raise ValueError("LCMV beamformer requires an empirical data covariance")
        beamformer_fwd = mne.convert_forward_solution(
            fwd,
            surf_ori=True,
            force_fixed=False,
            copy=True,
            use_cps=True,
            verbose=False,
        )
        filters = make_lcmv(
            info,
            beamformer_fwd,
            data_cov=data_cov,
            reg=float(beamformer_reg),
            noise_cov=noise_cov,
            pick_ori=coerce_mne_pick_ori(beamformer_pick_ori),
            rank="info",
            weight_norm=_coerce_optional_keyword(beamformer_weight_norm),
            reduce_rank=bool(resolved_conductor == "sphere"),
            depth=None if beamformer_depth is None else float(beamformer_depth),
            inversion="matrix",
            verbose=False,
        )
        return info, {
            "kind": "lcmv",
            "filters": filters,
            "forward": beamformer_fwd,
            "conductor_kind": resolved_conductor,
            "beamformer_pick_ori": coerce_mne_pick_ori(beamformer_pick_ori),
            "beamformer_weight_norm": _coerce_optional_keyword(beamformer_weight_norm),
            "beamformer_reg": float(beamformer_reg),
            "beamformer_depth": None if beamformer_depth is None else float(beamformer_depth),
            "beamformer_reduce_rank": bool(resolved_conductor == "sphere"),
        }

    raise ValueError(f"Unsupported inverse kind: {inverse_kind}")


def _coerce_optional_keyword(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    normalized = str(value).strip().lower()
    if not normalized or normalized == "none":
        return None
    return normalized


def coerce_mne_pick_ori(mne_pick_ori: Optional[str]) -> Optional[str]:
    return _coerce_optional_keyword(mne_pick_ori)


def extract_condition_roi_timecourses(
    data: np.ndarray,
    info: mne.Info,
    src: mne.SourceSpaces,
    labels: Sequence[mne.Label],
    inverse_payload: dict,
    mne_method: str,
    mne_pick_ori: Optional[str],
    lambda2: float,
    label_mode: str,
) -> np.ndarray:
    evoked = mne.EvokedArray(np.asarray(data, dtype=float), info, tmin=0.0, verbose=False)

    if inverse_payload["kind"] == "mne":
        stc = mne.minimum_norm.apply_inverse(
            evoked,
            inverse_payload["inverse_operator"],
            lambda2=float(lambda2),
            method=str(mne_method),
            pick_ori=coerce_mne_pick_ori(mne_pick_ori),
            verbose=False,
        )
    elif inverse_payload["kind"] == "lcmv":
        stc = apply_lcmv(evoked, inverse_payload["filters"], verbose=False)
    else:
        raise ValueError(f"Unsupported inverse payload kind: {inverse_payload['kind']}")

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="source space does not contain any vertices for .*",
            category=RuntimeWarning,
        )
        roi_timecourses = mne.extract_label_time_course(
            stc,
            labels,
            src,
            mode=label_mode,
            allow_empty=True,
            verbose=False,
        )
    return np.asarray(roi_timecourses, dtype=float)


def select_names(
    names: Sequence[str],
    include_patterns: Optional[Sequence[str]] = None,
    exclude_patterns: Optional[Sequence[str]] = None,
) -> List[int]:
    include = [pattern.lower() for pattern in (include_patterns or []) if pattern]
    exclude = [pattern.lower() for pattern in (exclude_patterns or []) if pattern]
    idxs: List[int] = []
    for idx, name in enumerate(names):
        lowered = str(name).lower()
        if include and not any(pattern in lowered for pattern in include):
            continue
        if exclude and any(pattern in lowered for pattern in exclude):
            continue
        idxs.append(idx)
    return idxs


def summarize_selected_names(
    names: Sequence[str],
    include_patterns: Optional[Sequence[str]],
    exclude_patterns: Optional[Sequence[str]] = None,
) -> List[str]:
    idxs = select_names(names, include_patterns=include_patterns, exclude_patterns=exclude_patterns)
    return [names[idx] for idx in idxs]


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def json_dump(path: str, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def as_list(values: Optional[Iterable[str]]) -> List[str]:
    if values is None:
        return []
    return [str(value) for value in values]
