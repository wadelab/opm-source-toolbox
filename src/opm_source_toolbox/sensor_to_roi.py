"""Manifest-driven sensor-space to atlas-ROI conversion.

This module is the generic source-imaging layer behind the reusable workflow:

    sensor matrix or FIF data + trans + subject anatomy + atlas -> ROI time series

It is intentionally agnostic to trigger logic and experiment structure. Upstream code
is responsible for deciding how sensor data should be selected; this module only
normalizes those inputs into a channels-by-time matrix, projects that matrix into
source space, and exports ROI-by-time outputs plus provenance.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import json
import os
import re
from typing import Any, Optional, Sequence
import warnings

import mne
import numpy as np

from .core import (
    DEFAULT_ATLAS_PARC,
    build_empirical_covariance_from_data,
    build_identity_noise_covariance,
    build_run_inverse_operator,
    ensure_dir,
    extract_condition_roi_timecourses,
    find_meg_sensors_inside_inner_skull,
    json_dump,
    load_atlas_labels,
    load_matrix_csv,
    load_run_info,
    setup_subject_source_space,
    write_matrix_csv,
    write_matrix_npz,
)


_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9._-]+")
_SUPPORTED_OUTPUT_FORMATS = {"csv", "npz"}


@dataclass
class SensorMatrixSpec:
    """Describe one sensor data item plus the geometry files needed to project it."""

    name: str
    trans_path: str
    matrix_path: Optional[str] = None
    sfreq_hz: Optional[float] = None
    time_start_s: Optional[float] = None
    fif_path: Optional[str] = None
    fif_kind: Optional[str] = None
    source_file: Optional[str] = None
    ch_name_col: str = "ch_name"
    geometry_info_file: Optional[str] = None
    epoch_index: Optional[int] = None
    epochs_average: bool = False
    evoked_index: Optional[int] = None
    evoked_comment: Optional[str] = None
    tmin_s: Optional[float] = None
    tmax_s: Optional[float] = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SubjectProjectionSpec:
    """Group all source-projection inputs for one subject."""

    subject: str
    subject_dir: str
    fs_subject: str = "FS"
    items: list[SensorMatrixSpec] = field(default_factory=list)


@dataclass
class SourceProjectionConfig:
    """Store atlas, inverse, conductor, and covariance settings for the conversion."""

    atlas_parc: str = DEFAULT_ATLAS_PARC
    atlas_subject: Optional[str] = None
    atlas_subjects_dir: Optional[str] = None
    source_spacing: str = "ico3"
    inverse_kind: str = "mne"
    mne_method: str = "MNE"
    mne_pick_ori: Optional[str] = "normal"
    label_mode: str = "pca_flip"
    estimate_covariance: bool = False
    covariance_scope: str = "per_item"
    covariance_fallback: str = "identity"
    snr: float = 3.0
    loose: float = 0.2
    depth: float = 0.8
    beamformer_reg: float = 0.05
    beamformer_pick_ori: Optional[str] = "normal"
    beamformer_weight_norm: Optional[str] = "unit-noise-gain-invariant"
    beamformer_depth: Optional[float] = None
    conductor_kind: str = "auto"
    sphere_origin: tuple[float, float, float] = (0.0, 0.0, 0.0)
    sphere_head_radius: float = 0.07

    def validate(self) -> None:
        """Validate that the configuration is internally consistent."""

        # Fail fast on unsupported modes before any expensive source-model setup starts.
        if self.inverse_kind not in {"mne", "lcmv"}:
            raise ValueError(f"Unsupported inverse kind: {self.inverse_kind}")
        if self.covariance_scope not in {"per_item", "per_subject"}:
            raise ValueError(f"Unsupported covariance scope: {self.covariance_scope}")
        if self.covariance_fallback not in {"identity", "adhoc"}:
            raise ValueError(f"Unsupported covariance fallback: {self.covariance_fallback}")
        # The generic LCMV path depends on an empirical covariance rather than a
        # synthetic fallback, so reject that combination here.
        if self.inverse_kind == "lcmv" and not self.estimate_covariance:
            raise ValueError(
                "LCMV requires estimate_covariance=True in the generic sensor-to-ROI workflow"
            )


@dataclass
class RoiProjectionResult:
    """Capture one projected ROI time-series result plus provenance details."""

    item_name: str
    roi_names: list[str]
    data: np.ndarray
    source_file: str
    geometry_info_file: str
    trans_path: str
    n_input_channels: int
    n_used_channels: int
    dropped_sensor_channels: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class _LoadedSensorItem:
    """Hold a validated sensor matrix together with its resolved geometry file."""

    spec: SensorMatrixSpec
    ch_names: list[str]
    data: np.ndarray
    geometry_info_file: str
    input_kind: str
    input_path: str
    sfreq_hz: Optional[float] = None
    time_start_s: Optional[float] = None


def _first_present(payload: dict[str, Any], *keys: str, default: Any = None) -> Any:
    """Return the first non-None value found among several candidate keys."""

    # Manifest fields can arrive under a few aliases; use the first populated one.
    for key in keys:
        value = payload.get(key)
        if value is not None:
            return value
    return default


def _sanitize_output_stem(name: str) -> str:
    """Convert an item name into a filename-safe output stem."""

    # Item names often become output filenames, so normalize them once here.
    cleaned = _SAFE_NAME_RE.sub("_", str(name).strip())
    cleaned = cleaned.strip("._")
    return cleaned or "item"


def _item_name_from_path(path: str) -> str:
    """Derive a default item name from a matrix filepath."""

    base = os.path.basename(str(path))
    return os.path.splitext(base)[0] or "item"


def _normalize_output_format(output_format: str) -> str:
    normalized = str(output_format).strip().lower()
    if normalized not in _SUPPORTED_OUTPUT_FORMATS:
        allowed = ", ".join(sorted(_SUPPORTED_OUTPUT_FORMATS))
        raise ValueError(
            f"Unsupported output format '{output_format}'; expected one of: {allowed}"
        )
    return normalized


def _coerce_positive_float(value: Any, *, field_name: str, context: str) -> float:
    try:
        coerced = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context} has invalid '{field_name}': {value!r}") from exc
    if not np.isfinite(coerced) or coerced <= 0.0:
        raise ValueError(f"{context} has invalid '{field_name}': {value!r}")
    return coerced


def _coerce_finite_float(value: Any, *, field_name: str, context: str) -> float:
    try:
        coerced = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context} has invalid '{field_name}': {value!r}") from exc
    if not np.isfinite(coerced):
        raise ValueError(f"{context} has invalid '{field_name}': {value!r}")
    return coerced


def _coerce_bool(value: Any, *, default: bool = False) -> bool:
    """Normalize a manifest boolean flag from JSON or string input."""

    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Could not parse boolean value: {value!r}")


def _meg_picks(info: mne.Info) -> list[int]:
    """Return MEG channel picks while excluding trigger and auxiliary channels."""

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
    ).tolist()
    if not picks:
        raise ValueError("No MEG channels were found in the requested sensor item")
    return picks


def _resolve_fif_kind(spec: SensorMatrixSpec) -> str:
    """Determine which MNE container type should be read for a FIF-backed item."""

    if not spec.fif_path:
        raise ValueError("Cannot resolve FIF kind without a fif_path")
    if spec.fif_kind:
        normalized = str(spec.fif_kind).strip().lower()
        if normalized not in {"raw", "epochs", "evoked"}:
            raise ValueError(f"Unsupported fif_kind for {spec.name}: {spec.fif_kind}")
        return normalized

    detected = str(mne.what(spec.fif_path)).strip().lower()
    if detected in {"raw", "epochs", "evoked"}:
        return detected
    raise ValueError(
        f"Could not infer a supported FIF type for {spec.fif_path}; "
        "pass fif_kind explicitly as raw, epochs, or evoked"
    )


def _validate_crop_bounds(times: np.ndarray, spec: SensorMatrixSpec) -> None:
    """Check that requested time cropping lies within the available data span."""

    if times.size == 0:
        raise ValueError(f"Sensor item {spec.name} does not contain any time samples")
    time_min = float(times[0])
    time_max = float(times[-1])
    if spec.tmin_s is not None and float(spec.tmin_s) > time_max:
        raise ValueError(
            f"Requested tmin_s={spec.tmin_s} is outside the data range "
            f"[{time_min:g}, {time_max:g}] for {spec.name}"
        )
    if spec.tmax_s is not None and float(spec.tmax_s) < time_min:
        raise ValueError(
            f"Requested tmax_s={spec.tmax_s} is outside the data range "
            f"[{time_min:g}, {time_max:g}] for {spec.name}"
        )
    if (
        spec.tmin_s is not None
        and spec.tmax_s is not None
        and float(spec.tmin_s) > float(spec.tmax_s)
    ):
        raise ValueError(f"Requested tmin_s > tmax_s for {spec.name}")


def _crop_data_by_times(
    data: np.ndarray,
    times: np.ndarray,
    spec: SensorMatrixSpec,
) -> tuple[np.ndarray, np.ndarray]:
    """Crop a channels-by-time matrix to the requested time window."""

    data = np.asarray(data, dtype=float)
    times = np.asarray(times, dtype=float)
    if spec.tmin_s is None and spec.tmax_s is None:
        return data, times

    _validate_crop_bounds(times, spec)
    mask = np.ones(times.shape, dtype=bool)
    if spec.tmin_s is not None:
        mask &= times >= float(spec.tmin_s)
    if spec.tmax_s is not None:
        mask &= times <= float(spec.tmax_s)
    if not np.any(mask):
        raise ValueError(f"Requested crop left no time samples for {spec.name}")
    return data[:, mask], times[mask]


def _read_raw_matrix_from_fif(spec: SensorMatrixSpec) -> tuple[list[str], np.ndarray, float, float]:
    """Read a MEG channels-by-time matrix directly from a Raw FIF file."""

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="This filename .* does not conform to MNE naming conventions.*",
            category=RuntimeWarning,
        )
        raw = mne.io.read_raw_fif(str(spec.fif_path), preload=False, verbose=False)

    picks = _meg_picks(raw.info)
    ch_names = [raw.ch_names[idx] for idx in picks]
    data = raw.get_data(picks=picks)
    data, times = _crop_data_by_times(data, raw.times, spec)
    return ch_names, data, float(raw.info["sfreq"]), float(times[0])


def _read_epochs_matrix_from_fif(
    spec: SensorMatrixSpec,
) -> tuple[list[str], np.ndarray, float, float]:
    """Read one MEG channels-by-time matrix from an Epochs FIF file."""

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="This filename .* does not conform to MNE naming conventions.*",
            category=RuntimeWarning,
        )
        epochs = mne.read_epochs(str(spec.fif_path), preload=False, verbose=False)

    picks = _meg_picks(epochs.info)
    ch_names = [epochs.ch_names[idx] for idx in picks]
    if spec.epochs_average:
        # Averaging here lets the source stage consume an Epochs file without deciding
        # anything about triggers beyond "collapse all contained epochs".
        data = epochs.get_data(picks=picks).mean(axis=0)
    elif spec.epoch_index is not None:
        epoch_index = int(spec.epoch_index)
        if epoch_index < 0 or epoch_index >= len(epochs):
            raise IndexError(
                f"epoch_index={epoch_index} is out of range for {spec.name} ({len(epochs)} epochs)"
            )
        data = epochs[epoch_index].get_data(picks=picks)[0]
    elif len(epochs) == 1:
        data = epochs.get_data(picks=picks)[0]
    else:
        raise ValueError(
            f"Epochs FIF {spec.fif_path} contains {len(epochs)} epochs for {spec.name}; "
            "pass epoch_index or epochs_average in the manifest"
        )
    data, times = _crop_data_by_times(data, epochs.times, spec)
    return ch_names, data, float(epochs.info["sfreq"]), float(times[0])


def _read_evoked_matrix_from_fif(
    spec: SensorMatrixSpec,
) -> tuple[list[str], np.ndarray, float, float]:
    """Read a MEG channels-by-time matrix from an Evoked FIF file."""

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="This filename .* does not conform to MNE naming conventions.*",
            category=RuntimeWarning,
        )
        if spec.evoked_comment is not None:
            evoked = mne.read_evokeds(
                str(spec.fif_path),
                condition=str(spec.evoked_comment),
                verbose=False,
            )
        else:
            evokeds = mne.read_evokeds(str(spec.fif_path), condition=None, verbose=False)
            if spec.evoked_index is not None:
                evoked_index = int(spec.evoked_index)
                if evoked_index < 0 or evoked_index >= len(evokeds):
                    raise IndexError(
                        f"evoked_index={evoked_index} is out of range for {spec.name} "
                        f"({len(evokeds)} evoked entries)"
                    )
                evoked = evokeds[evoked_index]
            elif len(evokeds) == 1:
                evoked = evokeds[0]
            else:
                raise ValueError(
                    f"Evoked FIF {spec.fif_path} contains {len(evokeds)} entries for {spec.name}; "
                    "pass evoked_index or evoked_comment in the manifest"
                )

    picks = _meg_picks(evoked.info)
    ch_names = [evoked.ch_names[idx] for idx in picks]
    data, times = _crop_data_by_times(evoked.data[picks], evoked.times, spec)
    return ch_names, data, float(evoked.info["sfreq"]), float(times[0])


def load_subject_specs_from_manifest(manifest_path: str) -> list[SubjectProjectionSpec]:
    """Parse a manifest JSON file into subject and item projection specs."""

    with open(manifest_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    # Top-level keys can act as defaults for each subject block so one manifest can
    # describe many subjects without repeating shared settings.
    global_defaults = dict(payload) if isinstance(payload, dict) else {}
    subject_payloads = global_defaults.pop("subjects", None)
    # If there is no explicit "subjects" list, treat the whole file as one subject block.
    if subject_payloads is None:
        subject_payloads = [payload]

    if not isinstance(subject_payloads, list) or not subject_payloads:
        raise ValueError("Manifest must contain one or more subject blocks")

    specs: list[SubjectProjectionSpec] = []
    for subject_idx, subject_payload in enumerate(subject_payloads):
        if not isinstance(subject_payload, dict):
            raise ValueError(f"Subject block {subject_idx} is not a JSON object")

        # Resolve the subject identity first so any later validation errors can point to
        # a human-meaningful label rather than just a list index.
        # Accept a few key aliases so manifests from other pipelines can be reused
        # without being rewritten into one exact naming scheme.
        subject_name = _first_present(
            subject_payload,
            "subject",
            "subject_id",
            default=_first_present(global_defaults, "subject", "subject_id"),
        )
        if not subject_name:
            raise ValueError(f"Subject block {subject_idx} is missing 'subject'")

        subject_dir = _first_present(
            subject_payload,
            "subject_dir",
            default=_first_present(global_defaults, "subject_dir"),
        )
        if not subject_dir:
            raise ValueError(f"Subject block {subject_name} is missing 'subject_dir'")

        # These subject-level defaults let one block describe many items without
        # repeating the same anatomy and geometry settings over and over.
        fs_subject = str(
            _first_present(subject_payload, "fs_subject", default=global_defaults.get("fs_subject", "FS"))
        )
        ch_name_col = str(
            _first_present(subject_payload, "ch_name_col", default=global_defaults.get("ch_name_col", "ch_name"))
        )
        default_source_file = _first_present(
            subject_payload,
            "source_file",
            default=_first_present(global_defaults, "source_file"),
        )
        default_fif_path = _first_present(
            subject_payload,
            "fif_path",
            "input_fif",
            "data_fif",
            default=_first_present(global_defaults, "fif_path", "input_fif", "data_fif"),
        )
        default_fif_kind = _first_present(
            subject_payload,
            "fif_kind",
            "input_kind",
            default=_first_present(global_defaults, "fif_kind", "input_kind"),
        )
        default_trans_path = _first_present(
            subject_payload,
            "trans_path",
            default=_first_present(global_defaults, "trans_path"),
        )
        default_geometry_info_file = _first_present(
            subject_payload,
            "geometry_info_file",
            "info_file",
            default=_first_present(global_defaults, "geometry_info_file", "info_file"),
        )
        default_epoch_index = _first_present(
            subject_payload,
            "epoch_index",
            default=_first_present(global_defaults, "epoch_index"),
        )
        default_epochs_average = _coerce_bool(
            _first_present(
                subject_payload,
                "epochs_average",
                "average_epochs",
                default=_first_present(global_defaults, "epochs_average", "average_epochs"),
            ),
            default=False,
        )
        default_evoked_index = _first_present(
            subject_payload,
            "evoked_index",
            default=_first_present(global_defaults, "evoked_index"),
        )
        default_evoked_comment = _first_present(
            subject_payload,
            "evoked_comment",
            "evoked_name",
            default=_first_present(global_defaults, "evoked_comment", "evoked_name"),
        )
        default_tmin_s = _first_present(
            subject_payload,
            "tmin_s",
            "tmin",
            default=_first_present(global_defaults, "tmin_s", "tmin"),
        )
        default_tmax_s = _first_present(
            subject_payload,
            "tmax_s",
            "tmax",
            default=_first_present(global_defaults, "tmax_s", "tmax"),
        )
        default_sfreq_hz = _first_present(
            subject_payload,
            "sfreq_hz",
            "sfreq",
            default=_first_present(global_defaults, "sfreq_hz", "sfreq"),
        )
        default_time_start_s = _first_present(
            subject_payload,
            "time_start_s",
            default=_first_present(global_defaults, "time_start_s"),
        )

        # Different callers may think of these as items, matrices, or runs; here they
        # all normalize to "sensor matrices to project into source space".
        items_payload = (
            subject_payload.get("items")
            or subject_payload.get("matrices")
            or subject_payload.get("runs")
        )
        if not isinstance(items_payload, list) or not items_payload:
            raise ValueError(f"Subject block {subject_name} has no items")

        items: list[SensorMatrixSpec] = []
        for item_idx, item_payload in enumerate(items_payload):
            if not isinstance(item_payload, dict):
                raise ValueError(f"Item {item_idx} for subject {subject_name} is not a JSON object")

            matrix_path = _first_present(
                item_payload,
                "matrix_path",
                "csv_path",
                "csv",
            )
            fif_path = _first_present(
                item_payload,
                "fif_path",
                "input_fif",
                "data_fif",
                default=default_fif_path,
            )

            # Each item can now come from either an explicit channels-by-time CSV or a
            # FIF container that will be converted into that matrix on the fly.
            if bool(matrix_path) == bool(fif_path):
                raise ValueError(
                    f"Item {item_idx} for subject {subject_name} must provide exactly one of "
                    "'matrix_path' or 'fif_path'"
                )

            source_file = _first_present(
                item_payload,
                "source_file",
                default=default_source_file,
            )
            if not source_file and fif_path:
                # When reading directly from a FIF object, the same file usually also
                # supplies the measurement info needed for forward modeling.
                source_file = fif_path
            if matrix_path and not source_file:
                raise ValueError(
                    f"Item {item_idx} for subject {subject_name} is missing 'source_file'"
                )

            # The trans file links sensor coordinates into MRI space, so require it once
            # all inheritance has been resolved.
            trans_path = _first_present(
                item_payload,
                "trans_path",
                default=default_trans_path,
            )
            if not trans_path:
                raise ValueError(
                    f"Item {item_idx} for subject {subject_name} is missing 'trans_path'"
                )

            geometry_info_file = _first_present(
                item_payload,
                "geometry_info_file",
                "info_file",
                default=default_geometry_info_file,
            )
            fif_kind = _first_present(
                item_payload,
                "fif_kind",
                "input_kind",
                default=default_fif_kind,
            )
            epoch_index = _first_present(
                item_payload,
                "epoch_index",
                default=default_epoch_index,
            )
            epochs_average = _coerce_bool(
                _first_present(
                    item_payload,
                    "epochs_average",
                    "average_epochs",
                    default=default_epochs_average,
                ),
                default=default_epochs_average,
            )
            evoked_index = _first_present(
                item_payload,
                "evoked_index",
                default=default_evoked_index,
            )
            evoked_comment = _first_present(
                item_payload,
                "evoked_comment",
                "evoked_name",
                default=default_evoked_comment,
            )
            tmin_s = _first_present(
                item_payload,
                "tmin_s",
                "tmin",
                default=default_tmin_s,
            )
            tmax_s = _first_present(
                item_payload,
                "tmax_s",
                "tmax",
                default=default_tmax_s,
            )
            sfreq_hz = _first_present(
                item_payload,
                "sfreq_hz",
                "sfreq",
                default=default_sfreq_hz,
            )
            time_start_s = _first_present(
                item_payload,
                "time_start_s",
                default=default_time_start_s,
            )
            if matrix_path and sfreq_hz is None:
                raise ValueError(
                    f"Item {item_idx} for subject {subject_name} uses 'matrix_path' but "
                    "is missing 'sfreq_hz'"
                )
            if matrix_path and time_start_s is None:
                raise ValueError(
                    f"Item {item_idx} for subject {subject_name} uses 'matrix_path' but "
                    "is missing 'time_start_s'"
                )
            name = str(
                _first_present(
                    item_payload,
                    "name",
                    default=_item_name_from_path(str(matrix_path or fif_path)),
                )
            )
            # Preserve any caller-provided metadata so experiment-specific wrappers can
            # carry their own annotations through the generic export layer.
            metadata = item_payload.get("metadata") or {}
            if not isinstance(metadata, dict):
                raise ValueError(
                    f"Item {name} for subject {subject_name} has non-dict 'metadata'"
                )

            # Convert the loose JSON structure into a typed spec so downstream code can
            # stop thinking about manifest aliases and optional inheritance.
            items.append(
                SensorMatrixSpec(
                    name=name,
                    trans_path=str(trans_path),
                    matrix_path=None if matrix_path is None else str(matrix_path),
                    sfreq_hz=(
                        None
                        if sfreq_hz is None
                        else _coerce_positive_float(
                            sfreq_hz,
                            field_name="sfreq_hz",
                            context=f"Item {name} for subject {subject_name}",
                        )
                    ),
                    time_start_s=(
                        None
                        if time_start_s is None
                        else _coerce_finite_float(
                            time_start_s,
                            field_name="time_start_s",
                            context=f"Item {name} for subject {subject_name}",
                        )
                    ),
                    fif_path=None if fif_path is None else str(fif_path),
                    fif_kind=None if fif_kind is None else str(fif_kind),
                    source_file=None if source_file is None else str(source_file),
                    ch_name_col=str(item_payload.get("ch_name_col", ch_name_col)),
                    geometry_info_file=None if geometry_info_file is None else str(geometry_info_file),
                    epoch_index=None if epoch_index is None else int(epoch_index),
                    epochs_average=bool(epochs_average),
                    evoked_index=None if evoked_index is None else int(evoked_index),
                    evoked_comment=None if evoked_comment is None else str(evoked_comment),
                    tmin_s=None if tmin_s is None else float(tmin_s),
                    tmax_s=None if tmax_s is None else float(tmax_s),
                    metadata=dict(metadata),
                )
            )

        specs.append(
            SubjectProjectionSpec(
                subject=str(subject_name),
                subject_dir=str(subject_dir),
                fs_subject=fs_subject,
                items=items,
            )
        )

    return specs


def _load_sensor_item(spec: SensorMatrixSpec) -> _LoadedSensorItem:
    """Load one sensor item from either a CSV matrix or a FIF container."""

    # Every item still needs a trans file plus some FIF-based geometry source, even if
    # the actual sensor data arrives via a precomputed CSV matrix.
    geometry_info_file = spec.geometry_info_file or spec.source_file or spec.fif_path
    if geometry_info_file is None:
        raise ValueError(
            f"Sensor item {spec.name} is missing a geometry/source FIF file; "
            "pass source_file, geometry_info_file, or fif_path"
        )
    if not os.path.exists(geometry_info_file):
        raise FileNotFoundError(f"Missing geometry info file: {geometry_info_file}")
    if not os.path.exists(spec.trans_path):
        raise FileNotFoundError(f"Missing trans file: {spec.trans_path}")

    if spec.matrix_path is not None:
        # The CSV path preserves the original generic contract: an already-prepared
        # channels-by-time matrix that the source layer should project as-is.
        if not os.path.exists(spec.matrix_path):
            raise FileNotFoundError(f"Missing sensor matrix CSV: {spec.matrix_path}")
        if spec.sfreq_hz is None:
            raise ValueError(
                f"Sensor item {spec.name} uses matrix_path but is missing required sfreq_hz"
            )
        if spec.time_start_s is None:
            raise ValueError(
                f"Sensor item {spec.name} uses matrix_path but is missing required time_start_s"
            )
        ch_names, data = load_matrix_csv(spec.matrix_path, name_col=spec.ch_name_col)
        data = np.asarray(data, dtype=float)
        input_kind = "csv"
        input_path = str(spec.matrix_path)
        sfreq_hz = _coerce_positive_float(
            spec.sfreq_hz,
            field_name="sfreq_hz",
            context=f"Sensor item {spec.name}",
        )
        time_start_s = _coerce_finite_float(
            spec.time_start_s,
            field_name="time_start_s",
            context=f"Sensor item {spec.name}",
        )
    else:
        if spec.fif_path is None:
            raise ValueError(
                f"Sensor item {spec.name} does not define matrix_path or fif_path"
            )
        if not os.path.exists(spec.fif_path):
            raise FileNotFoundError(f"Missing input FIF file: {spec.fif_path}")

        # Direct FIF support removes the need for a manual CSV export when the caller
        # already knows which Raw/Epochs/Evoked object should be projected.
        fif_kind = _resolve_fif_kind(spec)
        if fif_kind == "raw":
            ch_names, data, sfreq_hz, time_start_s = _read_raw_matrix_from_fif(spec)
        elif fif_kind == "epochs":
            ch_names, data, sfreq_hz, time_start_s = _read_epochs_matrix_from_fif(spec)
        elif fif_kind == "evoked":
            ch_names, data, sfreq_hz, time_start_s = _read_evoked_matrix_from_fif(spec)
        else:
            raise ValueError(f"Unsupported FIF input kind for {spec.name}: {fif_kind}")

        spec = replace(
            spec,
            fif_kind=fif_kind,
            source_file=spec.source_file or str(spec.fif_path),
        )
        input_kind = fif_kind
        input_path = str(spec.fif_path)

    # By the time control reaches here, every input format has been normalized into the
    # same channels-by-time matrix expected by the source projection code below.
    if data.ndim != 2:
        raise ValueError(f"Expected a 2D sensor matrix for {spec.name}, got {data.shape}")
    if data.shape[0] != len(ch_names):
        raise ValueError(
            f"Sensor matrix row count does not match channel count for {spec.name}"
        )
    # A single time sample would make covariance-based processing degenerate.
    if data.shape[1] < 2:
        raise ValueError(f"Sensor matrix has too few timepoints for {spec.name}")

    return _LoadedSensorItem(
        spec=spec,
        ch_names=list(ch_names),
        data=np.asarray(data, dtype=float),
        geometry_info_file=str(geometry_info_file),
        input_kind=input_kind,
        input_path=input_path,
        sfreq_hz=sfreq_hz,
        time_start_s=time_start_s,
    )


def _build_fallback_covariance(info: mne.Info, fallback_kind: str) -> mne.Covariance:
    """Build the configured non-empirical covariance for minimum-norm export."""

    # These simple fallback covariances keep the generic workflow usable even when the
    # caller has not provided a baseline or other segment for empirical estimation.
    if fallback_kind == "identity":
        return build_identity_noise_covariance(info)
    if fallback_kind == "adhoc":
        return mne.make_ad_hoc_cov(info, verbose=False)
    raise ValueError(f"Unsupported covariance fallback: {fallback_kind}")


def _build_shared_empirical_covariance(loaded_items: Sequence[_LoadedSensorItem]) -> mne.Covariance:
    """Estimate one empirical covariance across all matrices for a subject."""

    if not loaded_items:
        raise ValueError("Cannot compute a shared covariance without any loaded items")

    # A subject-wide covariance is only meaningful if every item uses the same channel
    # ordering; otherwise the covariance matrix would mix different sensors together.
    ref_names = loaded_items[0].ch_names
    mismatched = [
        item.spec.name for item in loaded_items[1:] if list(item.ch_names) != list(ref_names)
    ]
    if mismatched:
        raise ValueError(
            "Per-subject empirical covariance requires identical channel ordering across items. "
            f"Mismatched items: {', '.join(mismatched)}"
        )

    # Rebuild the Info object in the same channel order as the exported matrices.
    info = load_run_info(loaded_items[0].geometry_info_file, ref_names)
    # Concatenate all items in time so the subject-level covariance sees every sample.
    concatenated = np.concatenate([item.data for item in loaded_items], axis=1)
    return build_empirical_covariance_from_data(
        concatenated,
        info=info,
        fallback_kind="raise",
    )


def _resolve_covariances(
    loaded_item: _LoadedSensorItem,
    config: SourceProjectionConfig,
    shared_empirical_cov: Optional[mne.Covariance] = None,
) -> tuple[mne.Info, mne.Covariance, Optional[mne.Covariance]]:
    """Choose the noise and data covariance objects for one loaded sensor item."""

    # Reconstruct Info from the geometry file so MNE sees the exact same channel order
    # as the exported sensor matrix.
    info = load_run_info(loaded_item.geometry_info_file, loaded_item.ch_names)

    if config.estimate_covariance:
        # Empirical covariance can be computed per item or once per subject and reused.
        empirical_cov = shared_empirical_cov or build_empirical_covariance_from_data(
            loaded_item.data,
            info=info,
            fallback_kind="raise",
        )
        if config.inverse_kind == "mne":
            # In the minimum-norm path, the empirical estimate acts as the noise covariance.
            return info, empirical_cov, None
        # In the current LCMV path, the same empirical estimate is reused for both noise
        # and data covariance unless a richer upstream workflow is added later.
        return info, empirical_cov, empirical_cov

    # If covariance estimation is disabled, only the minimum-norm branch has a defined
    # fallback behavior in this generic workflow.
    if config.inverse_kind == "lcmv":
        raise ValueError("LCMV requires estimate_covariance=True")

    # Without empirical estimation, fall back to a simple prior covariance so minimum-
    # norm export can still run on arbitrary matrices.
    return info, _build_fallback_covariance(info, config.covariance_fallback), None


def project_sensor_item_to_atlas_rois(
    loaded_item: _LoadedSensorItem,
    subject_dir: str,
    fs_subject: str,
    src: mne.SourceSpaces,
    labels: Sequence[mne.Label],
    config: SourceProjectionConfig,
    shared_empirical_cov: Optional[mne.Covariance] = None,
) -> RoiProjectionResult:
    """Project one loaded sensor matrix into atlas ROI time series."""

    # Resolve channel metadata and covariance inputs before touching the forward model.
    full_info, noise_cov_full, data_cov_full = _resolve_covariances(
        loaded_item=loaded_item,
        config=config,
        shared_empirical_cov=shared_empirical_cov,
    )

    # Start by assuming every exported channel can be used, then reduce that set only if
    # geometry checks force a retry.
    keep_idxs = list(range(len(loaded_item.ch_names)))
    run_ch_names = list(loaded_item.ch_names)
    run_noise_cov = noise_cov_full
    run_data_cov = data_cov_full
    dropped_channels: list[str] = []

    try:
        # Build the forward/inverse objects against this item's exact channel list.
        run_info, inverse_payload = build_run_inverse_operator(
            source_file=loaded_item.geometry_info_file,
            trans_path=loaded_item.spec.trans_path,
            ch_names=run_ch_names,
            src=src,
            noise_cov=run_noise_cov,
            data_cov=run_data_cov,
            subject_dir=subject_dir,
            fs_subject=fs_subject,
            inverse_kind=config.inverse_kind,
            loose=config.loose,
            depth=config.depth,
            conductor_kind=config.conductor_kind,
            sphere_origin=config.sphere_origin,
            sphere_head_radius=config.sphere_head_radius,
            beamformer_reg=config.beamformer_reg,
            beamformer_pick_ori=config.beamformer_pick_ori,
            beamformer_weight_norm=config.beamformer_weight_norm,
            beamformer_depth=config.beamformer_depth,
        )
    except RuntimeError as exc:
        if "inside the inner skull surface" not in str(exc):
            raise

        # This branch specifically handles a known geometry issue seen in some OPM
        # datasets; other runtime failures should still surface unchanged.
        # Some real datasets contain channels whose sensor locations sit inside the
        # inner-skull surface used for the forward model. When that happens, drop only
        # those channels and retry rather than failing the entire item.
        dropped_channels = find_meg_sensors_inside_inner_skull(
            info=full_info,
            trans_path=loaded_item.spec.trans_path,
            subject_dir=subject_dir,
            fs_subject=fs_subject,
        )
        if not dropped_channels:
            raise

        dropped_set = set(dropped_channels)
        keep_idxs = [
            idx for idx, ch_name in enumerate(loaded_item.ch_names) if ch_name not in dropped_set
        ]
        if not keep_idxs:
            raise RuntimeError(
                f"All sensors were excluded for {loaded_item.spec.name} after inside-skull filtering"
            )

        # The matrix, channel list, and covariance objects must all be sliced to the
        # same surviving channels before the inverse can be rebuilt.
        run_ch_names = [loaded_item.ch_names[idx] for idx in keep_idxs]
        run_noise_cov = mne.pick_channels_cov(
            noise_cov_full,
            include=run_ch_names,
            exclude=[],
            ordered=True,
        )
        run_data_cov = None
        if data_cov_full is not None:
            run_data_cov = mne.pick_channels_cov(
                data_cov_full,
                include=run_ch_names,
                exclude=[],
                ordered=True,
            )

        # Retry the inverse construction with the filtered channel set.
        run_info, inverse_payload = build_run_inverse_operator(
            source_file=loaded_item.geometry_info_file,
            trans_path=loaded_item.spec.trans_path,
            ch_names=run_ch_names,
            src=src,
            noise_cov=run_noise_cov,
            data_cov=run_data_cov,
            subject_dir=subject_dir,
            fs_subject=fs_subject,
            inverse_kind=config.inverse_kind,
            loose=config.loose,
            depth=config.depth,
            conductor_kind=config.conductor_kind,
            sphere_origin=config.sphere_origin,
            sphere_head_radius=config.sphere_head_radius,
            beamformer_reg=config.beamformer_reg,
            beamformer_pick_ori=config.beamformer_pick_ori,
            beamformer_weight_norm=config.beamformer_weight_norm,
            beamformer_depth=config.beamformer_depth,
        )

    # Once the inverse payload exists, apply it to the sensor matrix and collapse the
    # resulting source estimate into atlas label time series.
    roi_timecourses = extract_condition_roi_timecourses(
        data=loaded_item.data[keep_idxs],
        info=run_info,
        src=src,
        labels=labels,
        inverse_payload=inverse_payload,
        mne_method=config.mne_method,
        mne_pick_ori=config.mne_pick_ori,
        lambda2=1.0 / float(config.snr) ** 2,
        label_mode=config.label_mode,
    )

    # Package both the ROI matrix and the provenance needed to explain how it was made.
    return RoiProjectionResult(
        item_name=loaded_item.spec.name,
        roi_names=[label.name for label in labels],
        data=np.asarray(roi_timecourses, dtype=float),
        source_file=loaded_item.spec.source_file,
        geometry_info_file=loaded_item.geometry_info_file,
        trans_path=loaded_item.spec.trans_path,
        n_input_channels=len(loaded_item.ch_names),
        n_used_channels=len(keep_idxs),
        dropped_sensor_channels=dropped_channels,
        metadata=dict(loaded_item.spec.metadata),
    )


def export_subject_sensor_matrices(
    subject_spec: SubjectProjectionSpec,
    out_root: str,
    config: SourceProjectionConfig,
    output_format: str = "csv",
) -> str:
    """Export ROI time-series matrices and metadata for every item in one subject spec."""

    config.validate()
    output_format = _normalize_output_format(output_format)

    # Subject anatomy is expected in a standard FreeSurfer-style layout under subject_dir.
    subject_dir = os.path.abspath(subject_spec.subject_dir)
    fs_dir = os.path.join(subject_dir, subject_spec.fs_subject)
    if not os.path.isdir(subject_dir):
        raise FileNotFoundError(f"Missing subject_dir for {subject_spec.subject}: {subject_dir}")
    if not os.path.isdir(fs_dir):
        raise FileNotFoundError(
            f"Missing FreeSurfer anatomy for {subject_spec.subject}: {fs_dir}"
        )

    # Load every item up front so missing paths fail before source-space setup begins.
    loaded_items = [_load_sensor_item(spec) for spec in subject_spec.items]

    # Source space and atlas labels are subject-level resources, so compute them once
    # and reuse them across all matrices for the subject.
    src = setup_subject_source_space(
        subject_dir=subject_dir,
        fs_subject=subject_spec.fs_subject,
        spacing=config.source_spacing,
    )
    labels = load_atlas_labels(
        subject_dir=subject_dir,
        fs_subject=subject_spec.fs_subject,
        atlas_parc=config.atlas_parc,
        atlas_subject=config.atlas_subject,
        atlas_subjects_dir=config.atlas_subjects_dir,
    )

    shared_empirical_cov = None
    if config.estimate_covariance and config.covariance_scope == "per_subject":
        # This gives one covariance across all subject items when the caller wants a
        # shared estimate instead of separate per-item covariances.
        shared_empirical_cov = _build_shared_empirical_covariance(loaded_items)

    # Each subject gets its own export directory containing ROI matrices plus one
    # metadata record that describes the full conversion.
    subject_out = os.path.join(out_root, subject_spec.subject)
    ensure_dir(subject_out)

    used_stems: dict[str, int] = {}
    item_records: list[dict[str, Any]] = []
    roi_names = [label.name for label in labels]

    for loaded_item in loaded_items:
        # Process one matrix at a time so each output file has a direct provenance link
        # back to one sensor matrix and its geometry files.
        result = project_sensor_item_to_atlas_rois(
            loaded_item=loaded_item,
            subject_dir=subject_dir,
            fs_subject=subject_spec.fs_subject,
            src=src,
            labels=labels,
            config=config,
            shared_empirical_cov=shared_empirical_cov,
        )

        # Keep output filenames stable and shell-safe even if item names contain spaces
        # or punctuation.
        stem = _sanitize_output_stem(result.item_name)
        seen_n = used_stems.get(stem, 0)
        used_stems[stem] = seen_n + 1
        if seen_n:
            stem = f"{stem}_{seen_n + 1:02d}"
        out_path = os.path.join(subject_out, f"{stem}.{output_format}")
        if output_format == "csv":
            # Keep the original CSV layout as the default because downstream analysis
            # code already consumes it directly.
            write_matrix_csv(
                out_path,
                name_col="roi_name",
                names=result.roi_names,
                data=result.data,
            )
        else:
            write_matrix_npz(
                out_path,
                name_col="roi_name",
                names=result.roi_names,
                data=result.data,
            )

        # Keep enough metadata to audit how each exported matrix was produced later on.
        item_records.append(
            {
                "name": result.item_name,
                "output_file": os.path.basename(out_path),
                "output_format": output_format,
                "output_csv": os.path.basename(out_path) if output_format == "csv" else None,
                "input_kind": loaded_item.input_kind,
                "input_path": os.path.abspath(loaded_item.input_path),
                "matrix_path": (
                    None
                    if loaded_item.spec.matrix_path is None
                    else os.path.abspath(loaded_item.spec.matrix_path)
                ),
                "fif_path": (
                    None
                    if loaded_item.spec.fif_path is None
                    else os.path.abspath(loaded_item.spec.fif_path)
                ),
                "fif_kind": loaded_item.spec.fif_kind,
                "source_file": os.path.abspath(result.source_file),
                "geometry_info_file": os.path.abspath(result.geometry_info_file),
                "trans_path": os.path.abspath(result.trans_path),
                "sfreq_hz": loaded_item.sfreq_hz,
                "time_start_s": loaded_item.time_start_s,
                "n_timepoints": int(result.data.shape[1]),
                "n_input_channels": int(result.n_input_channels),
                "n_used_channels": int(result.n_used_channels),
                "dropped_sensor_channels": list(result.dropped_sensor_channels),
                "metadata": dict(result.metadata),
            }
        )

    # The per-subject metadata file is the provenance record for downstream analyses.
    # Keep it fairly explicit because this is usually the first file someone inspects
    # when checking what atlas/inverse/covariance settings produced the export.
    payload = {
        "subject": subject_spec.subject,
        "subject_dir": subject_dir,
        "fs_subject": subject_spec.fs_subject,
        "n_items": len(item_records),
        "n_rois": len(roi_names),
        "roi_names": roi_names,
        "atlas_parc": config.atlas_parc,
        "atlas_subject": config.atlas_subject or subject_spec.fs_subject,
        "source_spacing": config.source_spacing,
        "output_format": output_format,
        "inverse_kind": config.inverse_kind,
        "mne_method": config.mne_method,
        "mne_pick_ori": config.mne_pick_ori,
        "label_mode": config.label_mode,
        "estimate_covariance": bool(config.estimate_covariance),
        "covariance_scope": config.covariance_scope,
        "covariance_fallback": config.covariance_fallback,
        "beamformer_reg": float(config.beamformer_reg),
        "beamformer_pick_ori": config.beamformer_pick_ori,
        "beamformer_weight_norm": config.beamformer_weight_norm,
        "beamformer_depth": config.beamformer_depth,
        "conductor_kind": config.conductor_kind,
        "sphere_origin_m": [float(value) for value in config.sphere_origin],
        "sphere_head_radius_m": float(config.sphere_head_radius),
        "items": item_records,
    }
    json_dump(os.path.join(subject_out, "metadata.json"), payload)
    return subject_out


def export_manifest_to_rois(
    manifest_path: str,
    out_root: str,
    config: SourceProjectionConfig,
    output_format: str = "csv",
) -> list[str]:
    """Convert a whole manifest into per-subject ROI exports and return their paths."""

    # Manifest parsing is separate from export so other Python callers can bypass JSON
    # and construct SubjectProjectionSpec objects directly if they prefer.
    subject_specs = load_subject_specs_from_manifest(manifest_path)
    ensure_dir(out_root)
    out_paths: list[str] = []
    for subject_spec in subject_specs:
        # Return the subject output directories so callers can chain into downstream
        # analyses without rediscovering where files were written.
        out_paths.append(
            export_subject_sensor_matrices(
                subject_spec,
                out_root=out_root,
                config=config,
                output_format=output_format,
            )
        )
    return out_paths
