# OPM Source ROI

Standalone generic OPM/MEG source-imaging tools for ROI export, alignment QC, and
surface rendering.

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

## Install

From this directory:

```bash
uv pip install -e .
```

For local development and tests:

```bash
uv pip install -e ".[dev]"
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

Additional command and API details live in the `docs/` directory.