# Repo Cut Checklist

This checklist assumes `staging/opm-source-roi/` is the source of truth for the
first standalone extraction.

## Goal

Create a separate repository that can be installed with `uv` and imported from
critical downstream code as:

```python
import opm_source_toolbox
```

while keeping experiment-specific vibroMEG logic in the current repository.

## Repository Cut

1. Create a new repository, for example `opm-source-roi`.
2. Copy the contents of this staging directory into that repository root.
3. Preserve the current `src/` layout and the `opm_source_toolbox` import package
   for the first extraction pass.
4. Preserve the existing console scripts:
   - `opm-source-fetch-atlas`
   - `opm-source-manifest-export`
   - `opm-source-alignment-qc`
   - `opm-source-surface-vector`
5. Keep the packaged Schaefer `.annot` assets in the wheel.

## What Stays In vibroMEG

These remain experiment-specific and should stay in the vibroMEG repository:

- trigger decoding and run/condition assumptions
- 23 Hz / 26 Hz steady-state analysis
- suppression analysis
- ANOVA wrappers
- vibro-specific CSV export naming conventions
- project-specific wrapper scripts around the generic toolbox

## Install Model

For development:

```bash
uv pip install -e .
```

For another repo before publishing:

```bash
uv add "git+ssh://git@github.com/<org>/opm-source-roi.git"
```

For stable downstream use after release tagging:

```bash
uv add opm-source-roi
```

## Consumption Model

Critical downstream code should:

- depend on the package through `uv`
- pin a version or git revision
- import only public API entrypoints
- avoid importing private helpers such as underscore-prefixed functions

Recommended public entrypoints:

- `SourceProjectionConfig`
- `export_manifest_to_rois`
- `fetch_atlas`
- `render_alignment_qc_bundle`
- `render_roi_value_map_to_surface`

## vibroMEG Migration

After the new repository exists:

1. Add the new package as a dependency to vibroMEG.
2. Replace in-repo imports with package imports from `opm_source_toolbox`.
3. Keep only thin wrappers in vibroMEG where the workflow is experiment-specific.
4. Run the existing vibroMEG validation and smoke tests against the external package.
5. Remove duplicated generic source code from vibroMEG only after the package-backed
   path is passing.

## Versioning

Recommended first pass:

- distribution name: `opm-source-roi`
- import package: `opm_source_toolbox`

That keeps the current Python import path stable while allowing the repository and
distribution name to describe the narrower deliverable.

## Release Gate

Before calling the first standalone extraction complete:

1. `pytest` passes in the standalone repo.
2. The wheel includes the `.annot` files.
3. The four documented CLI entrypoints resolve and run.
4. vibroMEG can install and import the package cleanly.
5. At least one real downstream workflow runs against the installed package rather
   than the in-repo copy.