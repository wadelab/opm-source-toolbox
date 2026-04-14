# OPM Source ROI

<p align="center">
  <img src="docs/OPM-MEG%20toolbox%20logo%20and%20visualization.png" alt="OPM-MEG toolbox logo and visualization" width="760">
</p>

This toolbox is for turning subject-specific OPM-MEG recordings in signal/sensor space
into ROI timecourses in a common anatomical reference space.

The motivating use case is straightforward: you collect OPM-MEG datasets for multiple
participants, but the raw sensor-space data are not yet directly comparable across
individuals. This toolbox ingests each participant's MEG data together with the MRI
coregistration and anatomy, computes source-space timecourses on the cortical surface,
and parcellates those timecourses into atlas-defined ROIs. By default, the toolbox uses
the Schaefer atlas.

The exported ROI timecourses can be written either as `.csv` tables with one column per
ROI and one row per timepoint, or as compressed binary `.npz` matrices for a more
compact output format.

This repository contains the standalone `opm-source-roi` package.

The toolbox currently exposes four reusable entrypoints:

- `opm-source-fetch-atlas`
- `opm-source-manifest-export`
- `opm-source-alignment-qc`
- `opm-source-surface-vector`

Those cover:

- atlas import into a reusable subjects directory
- sensor-space data to ROI time-series export
- sensor/head alignment QC rendering
- ROI value rendering onto cortical surfaces

Surface rendering and alignment QC remain part of this same toolbox package and CLI
surface. Optional extras only control whether the visualization dependencies are
installed; they do not indicate separate packages.

## Layout

- `src/opm_source_toolbox/`
	The package source
- `docs/`
	User-facing workflow documentation
- `tests/`
	Package-level tests for the generic workflows

## Install

From this repository:

```bash
uv pip install -e .
```

For local development and tests:

```bash
uv pip install -e ".[surface,alignment-qc,dev]"
```

## Quickstart

For the simplest case, assume:

- one subject directory such as `/data/co-reg/R9999/` containing `/data/co-reg/R9999/FS/`
- one MEG FIF input such as `/data/opm/R9999/run01_raw.fif`
- one matching transform in the same directory such as `/data/opm/R9999/run01_trans.fif`

First, install the default Schaefer atlas into the toolbox cache once:

```bash
uv run opm-source-fetch-atlas --atlas-name schaefer
```

Then export ROI timecourses:

```bash
uv run opm-source-manifest-export \
  --subject-dir /data/co-reg/R9999 \
  --fif-dir /data/opm/R9999 \
  --out-dir /data/source_roi_exports
```

In this mode the exporter scans `--fif-dir` for input `.fif` files, ignores
`*_trans.fif` sidecars, and because there is exactly one transform in that directory it
reuses that `trans` file automatically for the input FIF.

Outputs are written under `/data/source_roi_exports/R9999/`. By default the ROI matrix
is written as `.csv`; add `--output-format npz` for a compressed binary output instead.

If the FIF directory contains more than one data `.fif`, all of them will be exported.
Use `--fif-glob` to narrow the input set when needed.

## Testing

For core tests:

```bash
uv sync --extra dev
uv run python -m pytest
```

For the full optional test surface:

```bash
uv sync --extra dev --extra surface --extra alignment-qc
uv run python -m pytest
```

Using `python -m pytest` avoids falling back to a system `pytest` binary when the
project venv does not yet have the `dev` extra installed.

From another project by git URL:

```bash
uv add "git+ssh://git@github.com/<org>/opm-source-roi.git"
```

Or after publishing versioned releases:

```bash
uv add opm-source-roi
```

Python imports remain:

```python
import opm_source_toolbox
from opm_source_toolbox.sensor_to_roi import export_manifest_to_rois
from opm_source_toolbox.atlas_fetch import fetch_atlas
```

The intended consumption model is:

- install the package with `uv`
- pin a version or git revision in downstream code
- import the public API from `opm_source_toolbox`
- keep experiment-specific wrappers outside this package

The public Python API is documented in `docs/public_api.md`.

## Workflow

1. Import or stage atlas annotations with `opm-source-fetch-atlas`.
2. Export ROI timecourses with `opm-source-manifest-export` (`.csv` by default, or compressed `.npz` with `--output-format npz`). This can run from a JSON manifest or directly from a FIF directory with `--subject-dir` and `--fif-dir`.
3. Optionally inspect geometry with `opm-source-alignment-qc`.
4. Optionally render ROI values with `opm-source-surface-vector`.

Additional command and API details live in the `docs/` directory.
The release checklist for this package lives in `docs/release_checklist.md`.
