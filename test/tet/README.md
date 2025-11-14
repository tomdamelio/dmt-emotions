# TET Testing and Development Scripts

This directory contains testing, inspection, verification, and development scripts for the TET (Temporal Experience Tracking) analysis pipeline.

## Purpose

These scripts are **NOT** part of the core production pipeline. They are utility scripts for:

- **Quick Testing**: Rapid testing of individual components during development
- **Result Inspection**: Examining intermediate analysis results
- **Data Verification**: Validating data integrity and format
- **Method Comparison**: Comparing different analytical approaches
- **Feature Demonstration**: Demonstrating specific features or analyses

## Organization

### Testing Scripts
- `test_preprocessor_quick.py` - Quick test of preprocessing module
- `test_lme_quick.py` - Quick test of LME modeling
- `test_pca_quick.py` - Quick test of PCA analysis
- `test_peak_auc_quick.py` - Quick test of peak/AUC analysis

### Inspection Scripts
- `inspect_preprocessed_data.py` - Inspect preprocessed TET data
- `inspect_descriptive_results.py` - Inspect descriptive statistics
- `inspect_lme_results.py` - Inspect LME model results
- `inspect_peak_auc_results.py` - Inspect peak/AUC results
- `inspect_pca_results.py` - Inspect PCA results
- `inspect_clustering_results.py` - Inspect clustering results

### Verification Scripts
- `validate_tet_data.py` - Validate TET data integrity
- `verify_tet_data_compatibility.py` - Verify data compatibility
- `verify_tet_dimensions.py` - Verify dimension definitions
- `verify_tet_timing.py` - Verify temporal resolution
- `investigate_tet_timing.py` - Investigate timing issues

### Comparison and Demo Scripts
- `compare_tet_dimensions.py` - Compare dimension definitions
- `compare_tet_loading_methods.py` - Compare data loading methods
- `demo_tet_30s_aggregation.py` - Demonstrate 30-second aggregation

## Production Pipeline

For running the complete TET analysis pipeline, use:

```bash
python scripts/run_tet_analysis.py
```

For core TET analysis modules, see: `scripts/tet/`

For comprehensive documentation, see: `docs/TET_ANALYSIS_GUIDE.md`
