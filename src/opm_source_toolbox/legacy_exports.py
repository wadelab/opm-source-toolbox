"""Helpers for the legacy runXX_condYY CSV export layout.

These functions support the vibroMEG project wrappers and downstream analyses that
still rely on per-subject directories containing:

  <subject>/runXX_condYY.csv
  <subject>/metadata.json

They are intentionally kept outside the generic source-imaging core so the reusable
package boundary stays focused on manifest-driven sensor/FIF to ROI conversion.
"""

from __future__ import annotations

import glob
import json
import os
from typing import List, Optional, Tuple

import numpy as np

from .core import load_matrix_csv


def collect_subjects(in_root: str) -> List[str]:
    return sorted(
        os.path.basename(path)
        for path in glob.glob(os.path.join(in_root, "*"))
        if os.path.isdir(path)
    )


def collect_paths_for_subject(in_root: str, subject: str) -> List[str]:
    return sorted(glob.glob(os.path.join(in_root, subject, "run??_cond??.csv")))


def parse_run_cond(path: str) -> Tuple[int, int]:
    base = os.path.basename(path)
    run = int(base.split("_")[0].replace("run", ""))
    cond = int(base.split("_")[1].replace("cond", "").replace(".csv", ""))
    return run, cond


def load_metadata(in_root: str, subject: str) -> dict:
    meta_path = os.path.join(in_root, subject, "metadata.json")
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_subject_sensor_exports(in_root: str, subject: str) -> Tuple[List[str], np.ndarray, dict]:
    meta = load_metadata(in_root, subject)
    paths = collect_paths_for_subject(in_root, subject)
    if not paths:
        raise FileNotFoundError(f"No exported condition CSVs found for subject {subject}")

    condition_arrays = []
    ch_names_ref: Optional[List[str]] = None
    for path in paths:
        ch_names, data = load_matrix_csv(path, name_col="ch_name")
        if ch_names_ref is None:
            ch_names_ref = ch_names
        elif ch_names != ch_names_ref:
            raise ValueError(
                f"Channel order changed within subject {subject}; source export expects stable ordering"
            )
        condition_arrays.append(data)

    if ch_names_ref is None:
        raise RuntimeError(f"No channel names loaded for subject {subject}")

    return ch_names_ref, np.stack(condition_arrays, axis=0), meta


def load_subject_roi_exports(in_root: str, subject: str) -> Tuple[List[str], np.ndarray, dict]:
    meta = load_metadata(in_root, subject)
    paths = collect_paths_for_subject(in_root, subject)
    if not paths:
        raise FileNotFoundError(f"No ROI CSVs found for subject {subject}")

    condition_arrays = []
    roi_names_ref: Optional[List[str]] = None
    for path in paths:
        roi_names, data = load_matrix_csv(path, name_col="roi_name")
        if roi_names_ref is None:
            roi_names_ref = roi_names
        elif roi_names != roi_names_ref:
            raise ValueError(
                f"ROI order changed within subject {subject}; downstream analysis expects stability"
            )
        condition_arrays.append(data)

    if roi_names_ref is None:
        raise RuntimeError(f"No ROI names loaded for subject {subject}")

    return roi_names_ref, np.stack(condition_arrays, axis=0), meta


def align_by_common_names(
    subject_to_names: dict[str, List[str]],
    subject_to_data: dict[str, np.ndarray],
) -> Tuple[List[str], dict[str, np.ndarray]]:
    subjects = list(subject_to_names.keys())
    common = set(subject_to_names[subjects[0]])
    for subject in subjects[1:]:
        common &= set(subject_to_names[subject])
    common = [name for name in subject_to_names[subjects[0]] if name in common]
    if not common:
        raise RuntimeError("No common names found across subjects")

    aligned: dict[str, np.ndarray] = {}
    for subject in subjects:
        name_to_idx = {name: idx for idx, name in enumerate(subject_to_names[subject])}
        aligned[subject] = subject_to_data[subject][:, [name_to_idx[name] for name in common], :]
    return common, aligned