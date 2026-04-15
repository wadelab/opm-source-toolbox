"""Generic OPM/MEG source-imaging tools for ROI extraction and surface rendering."""

from importlib import import_module

from .atlas_fetch import (
	AtlasFetchResult,
	fetch_atlas,
	fetch_atlas_to_path,
	fetch_schaefer_annotations,
	import_annotation_pair,
)
from .core import DEFAULT_ATLAS_PARC, default_atlas_subjects_dir
from .sensor_to_roi import (
	RoiProjectionResult,
	SensorMatrixSpec,
	SourceProjectionConfig,
	SubjectProjectionSpec,
	export_manifest_to_rois,
	export_subject_sensor_matrices,
	load_subject_specs_from_manifest,
)
from .workflow import (
	build_bem,
	DEFAULT_SAMPLE_FILES,
	DEFAULT_SAMPLE_FS_URL,
	EventDetectionResult,
	DEFAULT_YORK_FS_SAMPLE_URL,
	DEFAULT_YORK_SAMPLE_FILES,
	detect_primary_events,
	download_file,
	extract_zip_once,
	find_freesurfer_subject_dir,
	prepare_sample_dataset,
	prepare_york_sample_dataset,
	SampleDataset,
	YorkSampleDataset,
)

__all__ = [
	"AlignmentQcResult",
	"AtlasFetchResult",
	"DEFAULT_ATLAS_PARC",
	"build_bem",
	"DEFAULT_SAMPLE_FILES",
	"DEFAULT_SAMPLE_FS_URL",
	"DEFAULT_YORK_FS_SAMPLE_URL",
	"DEFAULT_YORK_SAMPLE_FILES",
	"detect_primary_events",
	"download_file",
	"EventDetectionResult",
	"default_atlas_subjects_dir",
	"fetch_atlas",
	"fetch_atlas_to_path",
	"fetch_schaefer_annotations",
	"extract_zip_once",
	"find_freesurfer_subject_dir",
	"import_annotation_pair",
	"load_roi_value_map_csv",
	"prepare_sample_dataset",
	"prepare_york_sample_dataset",
	"RoiProjectionResult",
	"SampleDataset",
	"SensorMatrixSpec",
	"SourceProjectionConfig",
	"SurfaceRenderConfig",
	"SubjectProjectionSpec",
	"export_manifest_to_rois",
	"export_subject_sensor_matrices",
	"load_subject_specs_from_manifest",
	"render_alignment_qc_bundle",
	"render_alignment_screenshot",
	"render_roi_value_map_to_surface",
	"render_roi_vector_to_surface",
	"YorkSampleDataset",
]


def __getattr__(name: str):
	if name in {"AlignmentQcResult", "render_alignment_qc_bundle", "render_alignment_screenshot"}:
		module = import_module(".alignment_qc", __name__)
		return getattr(module, name)
	if name in {
		"SurfaceRenderConfig",
		"load_roi_value_map_csv",
		"render_roi_value_map_to_surface",
		"render_roi_vector_to_surface",
	}:
		module = import_module(".roi_surface_render", __name__)
		return getattr(module, name)
	raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
