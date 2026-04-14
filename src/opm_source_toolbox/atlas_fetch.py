"""Atlas import helpers for importing annotation-based atlases into a subjects directory."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import shutil
from typing import Callable, Optional

from .core import (
    DEFAULT_ATLAS_PARC,
    _find_packaged_annotation_paths,
    _infer_atlas_subject_from_annotation,
    default_atlas_subjects_dir,
)


@dataclass
class AtlasFetchResult:
    atlas_name: str
    atlas_parc: str
    atlas_subject: str
    atlas_subjects_dir: str
    label_dir: str
    atlas_paths: list[str]
    source: str


def _annotation_pair_from_source_dir(source_dir: str, atlas_parc: str) -> tuple[str, str]:
    paths = [
        os.path.join(source_dir, f"lh.{atlas_parc}.annot"),
        os.path.join(source_dir, f"rh.{atlas_parc}.annot"),
    ]
    missing = [path for path in paths if not os.path.exists(path)]
    if missing:
        raise FileNotFoundError(
            f"Missing annotation files for atlas_parc={atlas_parc} in {source_dir}: {missing}"
        )
    return str(paths[0]), str(paths[1])


def import_annotation_pair(
    *,
    atlas_name: str,
    atlas_parc: str,
    lh_annot_path: str,
    rh_annot_path: str,
    atlas_subject: Optional[str] = None,
    atlas_subjects_dir: Optional[str] = None,
    force: bool = False,
    source: Optional[str] = None,
) -> AtlasFetchResult:
    inferred_subject = atlas_subject or _infer_atlas_subject_from_annotation(lh_annot_path)
    if not inferred_subject:
        raise ValueError(
            "Could not infer atlas_subject from the left-hemisphere annotation; pass --atlas-subject explicitly"
        )

    subjects_dir = default_atlas_subjects_dir(atlas_subjects_dir, create=True)
    if not subjects_dir:
        raise RuntimeError("Could not resolve a writable atlas_subjects_dir")

    label_dir = os.path.join(subjects_dir, inferred_subject, "label")
    os.makedirs(label_dir, exist_ok=True)
    out_paths = [
        os.path.join(label_dir, f"lh.{atlas_parc}.annot"),
        os.path.join(label_dir, f"rh.{atlas_parc}.annot"),
    ]
    for src, dst in zip((lh_annot_path, rh_annot_path), out_paths):
        if os.path.exists(dst) and not force:
            continue
        shutil.copy2(src, dst)

    return AtlasFetchResult(
        atlas_name=str(atlas_name),
        atlas_parc=str(atlas_parc),
        atlas_subject=str(inferred_subject),
        atlas_subjects_dir=str(subjects_dir),
        label_dir=str(label_dir),
        atlas_paths=[str(path) for path in out_paths],
        source=str(source or "annotation-pair"),
    )


def fetch_schaefer_annotations(
    *,
    atlas_parc: str = DEFAULT_ATLAS_PARC,
    atlas_subject: Optional[str] = None,
    atlas_subjects_dir: Optional[str] = None,
    source_dir: Optional[str] = None,
    force: bool = False,
) -> AtlasFetchResult:
    if source_dir:
        lh_annot_path, rh_annot_path = _annotation_pair_from_source_dir(source_dir, atlas_parc)
        source = os.path.abspath(source_dir)
    else:
        packaged_paths = _find_packaged_annotation_paths(atlas_parc)
        if not packaged_paths:
            raise FileNotFoundError(
                f"No packaged Schaefer annotations found for atlas_parc={atlas_parc}"
            )
        lh_annot_path, rh_annot_path = packaged_paths
        source = "packaged-schaefer"

    return import_annotation_pair(
        atlas_name="schaefer",
        atlas_parc=atlas_parc,
        lh_annot_path=lh_annot_path,
        rh_annot_path=rh_annot_path,
        atlas_subject=atlas_subject,
        atlas_subjects_dir=atlas_subjects_dir,
        force=force,
        source=source,
    )


def fetch_atlas(
    atlas_name: str,
    *,
    atlas_parc: str = DEFAULT_ATLAS_PARC,
    atlas_subject: Optional[str] = None,
    atlas_subjects_dir: Optional[str] = None,
    source_dir: Optional[str] = None,
    lh_annot_path: Optional[str] = None,
    rh_annot_path: Optional[str] = None,
    force: bool = False,
) -> AtlasFetchResult:
    normalized_name = str(atlas_name).strip().lower()
    fetchers: dict[str, Callable[[], AtlasFetchResult]] = {
        "schaefer": lambda: fetch_schaefer_annotations(
            atlas_parc=atlas_parc,
            atlas_subject=atlas_subject,
            atlas_subjects_dir=atlas_subjects_dir,
            source_dir=source_dir,
            force=force,
        ),
        "annotation-pair": lambda: import_annotation_pair(
            atlas_name="annotation-pair",
            atlas_parc=atlas_parc,
            lh_annot_path=str(lh_annot_path),
            rh_annot_path=str(rh_annot_path),
            atlas_subject=atlas_subject,
            atlas_subjects_dir=atlas_subjects_dir,
            force=force,
            source="annotation-pair",
        ),
        "annot-pair": lambda: import_annotation_pair(
            atlas_name="annot-pair",
            atlas_parc=atlas_parc,
            lh_annot_path=str(lh_annot_path),
            rh_annot_path=str(rh_annot_path),
            atlas_subject=atlas_subject,
            atlas_subjects_dir=atlas_subjects_dir,
            force=force,
            source="annotation-pair",
        ),
    }
    if normalized_name in {"annotation-pair", "annot-pair"}:
        if not lh_annot_path or not rh_annot_path:
            raise ValueError("annotation-pair fetch requires both lh_annot_path and rh_annot_path")
    fetcher = fetchers.get(normalized_name)
    if fetcher is None:
        raise ValueError(f"Unsupported atlas fetcher: {atlas_name}")
    return fetcher()


def fetch_atlas_to_path(
    atlas_name: str,
    *,
    target_dir: str | Path,
    atlas_parc: str = DEFAULT_ATLAS_PARC,
    atlas_subject: Optional[str] = None,
    source_dir: Optional[str] = None,
    lh_annot_path: Optional[str] = None,
    rh_annot_path: Optional[str] = None,
    force: bool = False,
) -> AtlasFetchResult:
    return fetch_atlas(
        atlas_name,
        atlas_parc=atlas_parc,
        atlas_subject=atlas_subject,
        atlas_subjects_dir=str(target_dir),
        source_dir=source_dir,
        lh_annot_path=lh_annot_path,
        rh_annot_path=rh_annot_path,
        force=force,
    )
