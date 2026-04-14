# Public API

This file defines the public Python API for the standalone toolbox.

Critical downstream code should target these top-level imports:

```python
from opm_source_toolbox import DEFAULT_ATLAS_PARC
from opm_source_toolbox import SourceProjectionConfig, SensorMatrixSpec, SubjectProjectionSpec
from opm_source_toolbox import RoiProjectionResult
from opm_source_toolbox import export_manifest_to_rois, export_subject_sensor_matrices
from opm_source_toolbox import load_subject_specs_from_manifest
from opm_source_toolbox import fetch_atlas, fetch_atlas_to_path, fetch_schaefer_annotations
from opm_source_toolbox import import_annotation_pair
from opm_source_toolbox import default_atlas_subjects_dir
```

Optional visualization helpers are also public, but they may require optional extras:

```python
from opm_source_toolbox import AlignmentQcResult
from opm_source_toolbox import render_alignment_qc_bundle, render_alignment_screenshot
from opm_source_toolbox import SurfaceRenderConfig, load_roi_value_map_csv
from opm_source_toolbox import render_roi_value_map_to_surface, render_roi_vector_to_surface
```

## Not Public

The following are considered internal implementation details and should not be treated
as stable downstream dependencies:

- underscore-prefixed helpers such as `_load_sensor_item`
- implementation details inside `core.py`
- documentation layout or generated documentation artifacts
- compatibility shims in downstream repositories such as vibroMEG

## Stability Rule

Prefer changing implementation behind these names without changing the names
themselves.

If a change requires renaming or removing one of the symbols above, treat it as a
release-note event for downstream code.
