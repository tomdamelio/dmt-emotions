# Analysis Pipelines

This directory contains the main pipeline orchestration scripts for running complete analyses. These are the primary entry points for executing analysis workflows.

## Purpose

Pipeline scripts (`run_*.py`) orchestrate complete analysis workflows by:
- Calling individual analysis scripts in the correct sequence
- Managing dependencies between analysis stages
- Validating inputs and outputs
- Logging execution progress
- Handling errors gracefully

## Organization

### TET (Temporal Experience Tracking) Analysis
- **`run_tet_analysis.py`** - Complete TET analysis pipeline
  - Preprocessing → Descriptive Stats → LME → PCA → ICA → Physio Correlation → Clustering → Figures → Report
  - Single entry point for all TET analyses
  - Includes physiological-affective integration analysis
  - See: `docs/TET_ANALYSIS_GUIDE.md` for details

### Physiological Signal Analysis

#### EDA (Electrodermal Activity)
- **`run_eda_smna_analysis.py`** - Sympathetic Nervous Activity (phasic continuous)
- **`run_eda_scr_analysis.py`** - Skin Conductance Responses (phasic discrete events)

#### ECG (Electrocardiography)
- **`run_ecg_hr_analysis.py`** - Heart Rate analysis

#### Respiration
- **`run_resp_rvt_analysis.py`** - Respiratory Volume per Time analysis

#### Composite Indices
- **`run_composite_arousal_index.py`** - Multi-modal arousal index

## Usage

### Running a Pipeline

```bash
# TET analysis (complete pipeline)
python pipelines/run_tet_analysis.py

# EDA analysis
python pipelines/run_eda_smna_analysis.py
python pipelines/run_eda_scr_analysis.py

# ECG analysis
python pipelines/run_ecg_hr_analysis.py

# Respiration analysis
python pipelines/run_resp_rate_analysis.py
```

### Pipeline Options

Most pipelines support common options:

```bash
# Verbose output
python pipelines/run_tet_analysis.py --verbose

# Custom output directory
python pipelines/run_eda_scl_analysis.py --output results/eda/scl_custom

# Help
python pipelines/run_tet_analysis.py --help
```

### TET Pipeline Specific Options

The TET pipeline has additional orchestration options:

```bash
# Run specific stages
python pipelines/run_tet_analysis.py --stages preprocessing descriptive lme

# Skip stages
python pipelines/run_tet_analysis.py --skip-stages clustering

# Run from specific stage onward
python pipelines/run_tet_analysis.py --from-stage pca

# Dry run (validation only)
python pipelines/run_tet_analysis.py --dry-run
```

### TET Pipeline Stages

The TET pipeline consists of the following stages (in execution order):

1. **preprocessing** - Load and preprocess raw TET data
   - Dependencies: Raw TET data files
   - Outputs: `results/tet/preprocessed/tet_preprocessed.csv`

2. **descriptive** - Compute descriptive statistics
   - Dependencies: Preprocessed TET data
   - Outputs: Time course data, summary metrics

3. **lme** - Fit Linear Mixed Effects models
   - Dependencies: Preprocessed TET data
   - Outputs: `results/tet/lme/lme_results.csv`

4. **pca** - Principal Component Analysis
   - Dependencies: Preprocessed TET data
   - Outputs: PCA loadings, scores, variance explained

5. **ica** - Independent Component Analysis
   - Dependencies: Preprocessed TET data, PCA results
   - Outputs: ICA mixing matrix, scores, PCA correlations

6. **physio_correlation** - Physiological-TET correlation analysis
   - Dependencies: 
     - Preprocessed TET data (`results/tet/preprocessed/tet_preprocessed.csv`)
     - Composite physiological data (`results/composite/arousal_index_long.csv`)
     - PCA loadings (`results/composite/pca_loadings_pc1.csv`)
   - Outputs: Correlation matrices, regression results, CCA results, visualizations
   - Note: Requires running `run_composite_arousal_index.py` first to generate physiological data

7. **clustering** - Cluster analysis and state modeling
   - Dependencies: Preprocessed TET data
   - Outputs: Cluster assignments, state metrics, dose tests

8. **figures** - Generate publication-ready figures
   - Dependencies: LME results, other analysis outputs
   - Outputs: PNG figures in `results/tet/figures/`

9. **report** - Generate comprehensive results report
   - Dependencies: All analysis outputs
   - Outputs: `docs/tet_comprehensive_results.md`

### Running Physiological-TET Integration

To run the physiological-TET correlation analysis, you must first generate the composite physiological data:

```bash
# Step 1: Generate composite arousal index (includes HR, EDA, Resp PCA)
python pipelines/run_composite_arousal_index.py

# Step 2: Run TET pipeline including physio_correlation stage
python pipelines/run_tet_analysis.py --stages physio_correlation

# Or run complete pipeline (will include physio_correlation)
python pipelines/run_tet_analysis.py
```

## Directory Structure

```
project/
├── pipelines/              # Pipeline orchestrators (THIS DIRECTORY)
│   ├── run_tet_analysis.py
│   ├── run_eda_*.py
│   ├── run_ecg_*.py
│   └── run_resp_*.py
├── scripts/                # Individual analysis scripts
│   ├── preprocess_*.py
│   ├── compute_*.py
│   ├── generate_*.py
│   └── tet/               # TET analysis modules
├── test/                   # Testing and development
│   └── tet/               # TET testing scripts
└── results/                # Analysis outputs
    ├── tet/
    ├── eda/
    ├── ecg/
    └── resp/
```

## Development

### Creating a New Pipeline

When creating a new pipeline script:

1. Name it `run_<analysis_type>.py`
2. Place it in this directory (`pipelines/`)
3. Import analysis scripts from `scripts/`
4. Implement command-line interface with argparse
5. Add logging and error handling
6. Document usage in this README

### Best Practices

- **Single Responsibility**: Each pipeline should orchestrate one complete analysis workflow
- **Clear Naming**: Use descriptive names that indicate the analysis type
- **Comprehensive Logging**: Log all major steps and errors
- **Input Validation**: Validate inputs before starting analysis
- **Error Handling**: Handle errors gracefully, provide helpful messages
- **Documentation**: Document all command-line options and usage examples

## Related Documentation

- **TET Analysis**: `docs/TET_ANALYSIS_GUIDE.md`
- **General Pipeline**: `docs/PIPELINE.md`
- **Individual Scripts**: See docstrings in `scripts/` directory
- **Testing**: See `test/tet/README.md` for testing scripts

## Support

For issues or questions:
1. Check the relevant documentation in `docs/`
2. Review script docstrings for detailed usage
3. Check execution logs in `results/*/` directories
4. Consult the project README.md
