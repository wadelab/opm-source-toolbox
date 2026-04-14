# Release Checklist

This checklist assumes this repository is the source of truth for the standalone
`opm-source-roi` package.

## Goal

Keep this repository releasable as an installable package that can be imported from
critical downstream code as:

```python
import opm_source_toolbox
```

while keeping experiment-specific analysis logic in downstream repositories.

## Package Boundary

1. Preserve the current `src/` layout and the `opm_source_toolbox` import package.
2. Preserve the existing console scripts:
   - `opm-source-fetch-atlas`
   - `opm-source-manifest-export`
   - `opm-source-alignment-qc`
   - `opm-source-surface-vector`
3. Keep the packaged Schaefer `.annot` assets in the wheel.
4. Keep generated HTML documentation artifacts out of source control.

## What Stays Downstream

These remain experiment-specific and should stay in consumer repositories such as
`vibroMEG`:

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

## Downstream Migration

For an existing downstream consumer such as `vibroMEG`:

1. Add the new package as a dependency to vibroMEG.
2. Replace in-repo imports with package imports from `opm_source_toolbox`.
3. Keep only thin wrappers in vibroMEG where the workflow is experiment-specific.
4. Run the existing vibroMEG validation and smoke tests against the external package.
5. Remove duplicated generic source code from vibroMEG only after the package-backed
   path is passing.

## Versioning

Recommended package identity:

- distribution name: `opm-source-roi`
- import package: `opm_source_toolbox`

That keeps the current Python import path stable while allowing the distribution name
to describe the narrower deliverable.

## Release Gate

Before tagging or publishing a release:

1. `uv sync --extra dev` followed by `uv run python -m pytest` passes in the standalone repo.
2. The wheel includes the `.annot` files.
3. The four documented CLI entrypoints resolve and run.
4. `LICENSE` is present and matches the packaged metadata.
5. Generated HTML documentation artifacts are not committed.
6. vibroMEG can install and import the package cleanly.
7. At least one real downstream workflow runs against the installed package rather
   than the in-repo copy.
