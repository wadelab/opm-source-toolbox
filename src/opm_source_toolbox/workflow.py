"""Reusable workflow helpers for tutorials and lightweight pipelines.

These helpers sit above the generic sensor-to-ROI core. They handle a few common
workflow tasks that show up in notebooks and small scripts:

- downloading tutorial/sample files
- extracting cached zip archives exactly once
- locating a FreeSurfer subject directory under a downloaded tree
- selecting a primary event source from annotations or stim channels
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
import shutil
from typing import Sequence
from urllib.request import urlopen
import zipfile

import mne
import numpy as np


DEFAULT_YORK_SAMPLE_FILES: dict[str, str] = {
    "Rxxxx_InHelmet.ply": "https://osf.io/emvf7/download",
    "Rxxxx_01_Outside.ply": "https://osf.io/vp9e7/download",
    "FS/surf/mriscalp.stl": "https://osf.io/d4exr/download",
    "VEP_DS-raw.fif": "https://osf.io/ztkyf/download",
}
DEFAULT_YORK_FS_SAMPLE_URL = "https://osf.io/8v35q/download"
DEFAULT_SAMPLE_FILES = DEFAULT_YORK_SAMPLE_FILES
DEFAULT_SAMPLE_FS_URL = DEFAULT_YORK_FS_SAMPLE_URL


@dataclass(frozen=True)
class EventDetectionResult:
    """Store the chosen event array, ID mapping, and provenance label."""

    events: np.ndarray
    event_id: dict[str, int]
    source: str


@dataclass(frozen=True)
class YorkSampleDataset:
    """Resolved paths for the York tutorial sample dataset layout."""

    sample_dir: Path
    downloads_dir: Path
    subjects_download_dir: Path
    subject_dir: Path
    subjects_dir: Path
    subject_name: str
    raw_fif: Path
    aligned_fif: Path
    aligned_trans: Path
    inside_mesh: Path
    outside_mesh: Path
    mri_scalp: Path
    bem_solution: Path | None


SampleDataset = YorkSampleDataset


def download_file(
    url: str,
    destination: str | Path,
    *,
    chunk_size: int = 1024 * 1024,
) -> Path:
    """Download a file unless a non-empty destination already exists."""

    destination_path = Path(destination)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    if destination_path.exists() and destination_path.stat().st_size > 0:
        return destination_path

    with urlopen(url) as response, destination_path.open("wb") as handle:
        while True:
            chunk = response.read(chunk_size)
            if not chunk:
                break
            handle.write(chunk)

    return destination_path


def extract_zip_once(
    zip_path: str | Path,
    destination: str | Path,
    *,
    marker_name: str | None = None,
) -> Path:
    """Extract a zip archive once, guarded by a marker file in the output dir."""

    zip_path = Path(zip_path)
    destination_path = Path(destination)
    marker = destination_path / (marker_name or f".extracted_{zip_path.stem}")
    if marker.exists():
        return destination_path

    destination_path.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(destination_path)
    marker.touch()
    return destination_path


def find_freesurfer_subject_dir(root: str | Path) -> Path:
    """Locate the first FreeSurfer subject directory under a root tree."""

    root_path = Path(root)
    candidates: list[Path] = []
    for orig_mgz in root_path.rglob("mri/orig.mgz"):
        subject_dir = orig_mgz.parents[1]
        if (subject_dir / "surf" / "lh.white").exists():
            candidates.append(subject_dir)

    if not candidates:
        raise FileNotFoundError(f"No FreeSurfer subject found under {root_path}")

    return sorted(set(candidates))[0]


def detect_primary_events(
    raw: mne.io.BaseRaw,
    *,
    ignore_annotation_prefixes: Sequence[str] = ("bad", "edge"),
) -> EventDetectionResult:
    """Choose a primary event source from annotations or stim channels.

    The heuristic matches the notebook workflow: prefer usable annotations, else
    fall back to the stim channel with the most frequent event code.
    """

    useful_annotations = [
        desc
        for desc in raw.annotations.description
        if desc and not desc.lower().startswith(tuple(ignore_annotation_prefixes))
    ]
    if useful_annotations:
        events, event_id = mne.events_from_annotations(raw, verbose=False)
        counts = Counter(events[:, 2])
        chosen_code = max(counts.items(), key=lambda item: item[1])[0]
        chosen_name = next(name for name, code in event_id.items() if code == chosen_code)
        return EventDetectionResult(
            events=events,
            event_id={chosen_name: chosen_code},
            source="annotations",
        )

    stim_picks = mne.pick_types(raw.info, stim=True)
    for pick in stim_picks:
        stim_channel = raw.ch_names[pick]
        try:
            events = mne.find_events(
                raw,
                stim_channel=stim_channel,
                shortest_event=1,
                verbose=False,
            )
        except Exception:
            continue
        if len(events):
            counts = Counter(events[:, 2])
            chosen_code = max(counts.items(), key=lambda item: item[1])[0]
            return EventDetectionResult(
                events=events,
                event_id={"visual": chosen_code},
                source=stim_channel,
            )

    raise RuntimeError("Could not find events from annotations or stim channels.")


def build_bem(
    subject_dir: str | Path,
    *,
    subject_name: str | None = None,
    subjects_dir: str | Path | None = None,
    conductivity: Sequence[float] = (0.3,),
    overwrite: bool = True,
) -> Path:
    """Build and write a BEM solution for one FreeSurfer subject."""

    subject_dir = Path(subject_dir)
    subject_name = subject_name or subject_dir.name
    subjects_dir = Path(subjects_dir) if subjects_dir is not None else subject_dir.parent
    bem_dir = subject_dir / "bem"
    bem_dir.mkdir(parents=True, exist_ok=True)

    bem_model = mne.make_bem_model(
        subject=subject_name,
        conductivity=tuple(conductivity),
        subjects_dir=str(subjects_dir),
        verbose=False,
    )
    bem_solution = mne.make_bem_solution(bem_model, verbose=False)
    bem_solution_path = bem_dir / f"{subject_name}-bem-sol.fif"
    mne.write_bem_solution(
        str(bem_solution_path),
        bem_solution,
        overwrite=overwrite,
        verbose=False,
    )
    return bem_solution_path


def prepare_york_sample_dataset(
    work_dir: str | Path,
    *,
    sample_files: dict[str, str] | None = None,
    fs_sample_url: str = DEFAULT_YORK_FS_SAMPLE_URL,
    fs_zip_name: str = "FS_Sample.zip",
    raw_name: str = "VEP_DS-raw.fif",
) -> YorkSampleDataset:
    """Download/resolve the York tutorial sample dataset and key file paths."""

    work_dir = Path(work_dir)
    sample_dir = work_dir / "sampleData"
    downloads_dir = work_dir / "downloads"
    subjects_download_dir = work_dir / "subjects_download"
    sample_dir.mkdir(parents=True, exist_ok=True)
    downloads_dir.mkdir(parents=True, exist_ok=True)

    file_map = sample_files or DEFAULT_YORK_SAMPLE_FILES
    for relative_path, url in file_map.items():
        download_file(url, sample_dir / relative_path)

    fs_zip_path = downloads_dir / fs_zip_name
    download_file(fs_sample_url, fs_zip_path)
    extract_zip_once(fs_zip_path, subjects_download_dir)

    subject_dir = find_freesurfer_subject_dir(subjects_download_dir)
    subjects_dir = subject_dir.parent
    subject_name = subject_dir.name

    raw_fif = sample_dir / raw_name
    inside_mesh = sample_dir / "Rxxxx_InHelmet.ply"
    outside_mesh = sample_dir / "Rxxxx_01_Outside.ply"
    mri_scalp = sample_dir / "FS" / "surf" / "mriscalp.stl"

    aligned_fif = sample_dir / raw_name.replace("-raw.fif", "-aligned_raw.fif")
    if aligned_fif == raw_fif:
        aligned_fif = sample_dir / f"{raw_fif.stem}_aligned.fif"
    if not aligned_fif.exists():
        shutil.copy2(raw_fif, aligned_fif)
    aligned_trans = aligned_fif.with_name(f"{aligned_fif.stem}_trans.fif")

    bem_candidates = sorted((subject_dir / "bem").glob("*-bem-sol.fif"))
    bem_solution = bem_candidates[0] if bem_candidates else None

    return YorkSampleDataset(
        sample_dir=sample_dir,
        downloads_dir=downloads_dir,
        subjects_download_dir=subjects_download_dir,
        subject_dir=subject_dir,
        subjects_dir=subjects_dir,
        subject_name=subject_name,
        raw_fif=raw_fif,
        aligned_fif=aligned_fif,
        aligned_trans=aligned_trans,
        inside_mesh=inside_mesh,
        outside_mesh=outside_mesh,
        mri_scalp=mri_scalp,
        bem_solution=bem_solution,
    )


def prepare_sample_dataset(
    work_dir: str | Path,
    *,
    sample_files: dict[str, str] | None = None,
    fs_sample_url: str = DEFAULT_SAMPLE_FS_URL,
    fs_zip_name: str = "FS_Sample.zip",
    raw_name: str = "VEP_DS-raw.fif",
) -> SampleDataset:
    """Generic alias for the packaged sample dataset helper."""

    return prepare_york_sample_dataset(
        work_dir=work_dir,
        sample_files=sample_files,
        fs_sample_url=fs_sample_url,
        fs_zip_name=fs_zip_name,
        raw_name=raw_name,
    )