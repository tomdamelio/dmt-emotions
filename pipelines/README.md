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
  - Preprocessing → Descriptive Stats → LME → Peak/AUC → PCA → Clustering → Figures → Report
  - Single entry point for all TET analyses
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
