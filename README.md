# OPM Source ROI

This directory is a self-contained staging copy of the extracted generic OPM/MEG ROI
toolbox. It is intended to be the near-final handoff state before moving the package
into a dedicated repository.

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

## Layout

- `src/opm_source_toolbox/`
	The extracted package source
- `docs/`
	User-facing workflow documentation
- `tests/`
	Package-level tests for the generic workflows

## Install

From this directory:

```bash
uv pip install -e .
```

With optional visualization dependencies:

```bash
uv pip install -e ".[surface,alignment-qc,dev]"
```

To export this staging tree into a clean standalone repository directory:

```bash
./scripts/export_repo_candidate.sh /absolute/path/to/opm-source-roi
```

From another project after this is split into its own repository:

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

The intended first-release Python API is documented in `docs/public_api.md`.

## Workflow

1. Import or stage atlas annotations with `opm-source-fetch-atlas`.
2. Export ROI timecourses with `opm-source-manifest-export`.
3. Optionally inspect geometry with `opm-source-alignment-qc`.
4. Optionally render ROI values with `opm-source-surface-vector`.

The detailed extraction and packaging notes remain in `../docs/standalone_roi_package.md`.

The concrete first-repo-cut checklist lives in `docs/repo_cut_checklist.md`.