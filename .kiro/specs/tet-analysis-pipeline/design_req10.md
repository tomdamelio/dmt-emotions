# Design Document: Requirement 10 - Code Organization and Pipeline Orchestration

## Overview

This document describes the design for reorganizing the TET analysis codebase into a clean, maintainable structure with a single entry point for pipeline execution. The system SHALL consolidate core analysis modules in `scripts/tet/`, separate testing code in `test/tet/`, provide a unified orchestration script at `scripts/run_tet_analysis.py`, organize all outputs in `results/tet/`, and maintain a single comprehensive documentation file at `docs/TET_ANALYSIS_GUIDE.md`.

## Requirement Summary

**Requirement 10**: Code Organization and Pipeline Orchestration

**Key Acceptance Criteria**:
- 10.1: Organize core TET modules in `scripts/tet/`
- 10.2: Organize testing scripts in `test/tet/`
- 10.3: Provide single orchestration script at `scripts/run_tet_analysis.py`
- 10.4: Execute pipeline in correct sequence
- 10.5: Save all results in `results/tet/` with clear subdirectories
- 10.6: Save CSV files with descriptive names
- 10.7: Save figures in `results/tet/figures/`
- 10.8: Save figure captions in `results/tet/figures/captions/`
- 10.9: Generate final report at `results/tet/tet_analysis_report.md` (APA/Nature style)
- 10.10: Include Methods section in final report
- 10.11: Include Results section in final report with APA notation
- 10.12: Ensure final report is clearly identifiable
- 10.13: Provide command-line options for stage-specific execution
- 10.14: Log pipeline execution to `results/tet/pipeline_execution.log`
- 10.15: Validate input files before each stage
- 10.16: Provide single documentation at `docs/TET_ANALYSIS_GUIDE.md`
- 10.17: Organize documentation with clear sections
- 10.18: Archive redundant documentation files
- 10.19: Maintain documentation as authoritative reference

## Architecture

### Component Overview

```
TET Analysis System (Reorganized)
│
├── scripts/
│   ├── run_tet_analysis.py          # Single entry point orchestrator
│   └── tet/                          # Core analysis modules
│       ├── __init__.py
│       ├── preprocessor.py
│       ├── data_loader.py
│       ├── validator.py
│       ├── time_course.py
│       ├── session_metrics.py
│       ├── lme_analyzer.py
│       ├── contrast_analyzer.py
│       ├── peak_auc_analyzer.py
│       ├── pca_analyzer.py
│       ├── pca_lme_analyzer.py
│       ├── state_model_analyzer.py
│       ├── state_dose_analyzer.py
│       ├── state_visualization.py
│       ├── time_series_visualizer.py
│       ├── results_synthesizer.py
│       ├── results_analyzer.py
│       ├── results_formatter.py
│       ├── report_utils.py
│       ├── reporter.py
│       └── metadata.py
│
├── test/
│   └── tet/                          # Testing and development scripts
│       ├── test_preprocessor_quick.py
│       ├── test_lme_quick.py
│       ├── test_pca_quick.py
│       ├── test_peak_auc_quick.py
│       ├── compare_tet_dimensions.py
│       ├── compare_tet_loading_methods.py
│       ├── demo_tet_30s_aggregation.py
│       ├── inspect_preprocessed_data.py
│       ├── inspect_descriptive_results.py
│       ├── inspect_lme_results.py
│       ├── inspect_peak_auc_results.py
│       ├── inspect_pca_results.py
│       ├── inspect_clustering_results.py
│       ├── validate_tet_data.py
│       ├── verify_tet_data_compatibility.py
│       ├── verify_tet_dimensions.py
│       ├── verify_tet_timing.py
│       └── investigate_tet_timing.py
│
├── results/
│   └── tet/                          # All TET analysis outputs
│       ├── tet_preprocessed.csv
│       ├── tet_analysis_report.md    # Final comprehensive report (APA/Nature style)
│       ├── pipeline_execution.log
│       ├── descriptive/
│       │   ├── time_course_all_dimensions.csv
│       │   └── session_summaries.csv
│       ├── lme/
│       │   ├── lme_results.csv
│       │   └── lme_contrasts.csv
│       ├── peak_auc/
│       │   ├── peak_auc_metrics.csv
│       │   └── peak_auc_tests.csv
│       ├── pca/
│       │   ├── pca_loadings.csv
│       │   ├── pca_variance_explained.csv
│       │   └── pca_lme_results.csv
│       ├── clustering/
│       │   ├── clustering_kmeans_assignments.csv
│       │   ├── clustering_state_metrics.csv
│       │   ├── clustering_evaluation.csv
│       │   └── clustering_bootstrap_stability.csv
│       └── figures/
│           ├── timeseries_all_dimensions.png
│           ├── lme_coefficients_forest.png
│           ├── peak_dose_comparison.png
│           ├── time_to_peak_dose_comparison.png
│           ├── auc_dose_comparison.png
│           └── captions/
│               ├── timeseries_all_dimensions.txt
│               ├── lme_coefficients_forest.txt
│               └── peak_dose_comparison.txt
│
└── docs/
    ├── TET_ANALYSIS_GUIDE.md         # Single comprehensive documentation
    └── PIPELINE.md                   # General project pipeline (includes physiology)
```

### Design Principles

1. **Single Entry Point**: One script to run entire pipeline
2. **Clear Separation**: Core modules vs testing code
3. **Organized Outputs**: All results in structured subdirectories
4. **Comprehensive Documentation**: Single authoritative guide
5. **Maintainability**: Clean imports, clear dependencies
6. **Flexibility**: Run full pipeline or individual stages
7. **Traceability**: Execution logging and validation

## Detailed Design

### 10.1-10.2: Code Organization

**Purpose**: Separate production code from testing/development code.

**Core Modules** (`scripts/tet/`):
- Already well-organized with 20 modules
- All production-ready analysis components
- Clean imports and dependencies
- Comprehensive docstrings

**Testing Scripts** (`test/tet/`):
- Quick test scripts for rapid development
- Inspection scripts for result validation
- Comparison and verification utilities
- Demo and investigation scripts

**Migration Strategy**:
1. Create `test/tet/` directory
2. Move testing scripts from `scripts/` to `test/tet/`
3. Update imports in moved scripts
4. Verify all scripts still function correctly

### 10.3-10.4: Pipeline Orchestration Script

**Purpose**: Provide single entry point for complete TET analysis.

**Script**: `scripts/run_tet_analysis.py`

**Architecture**:
```python
class TETAnalysisPipeline:
    """Orchestrates complete TET analysis pipeline."""
    
    def __init__(self, config):
        self.config = config
        self.logger = self._setup_logging()
        self.stages = self._define_stages()
    
    def _define_stages(self):
        """Define pipeline stages in execution order."""
        return [
            ('preprocessing', self._run_preprocessing),
            ('descriptive', self._run_descriptive_stats),
            ('lme', self._run_lme_models),
            ('peak_auc', self._run_peak_auc_analysis),
            ('pca', self._run_pca_analysis),
            ('clustering', self._run_clustering_analysis),
            ('figures', self._run_figure_generation),
            ('report', self._run_report_generation)
        ]
    
    def run(self, stages=None, skip_stages=None):
        """Execute pipeline stages."""
        # Validate inputs
        # Execute selected stages
        # Log progress
        # Handle errors gracefully
    
    def _run_preprocessing(self):
        """Execute preprocessing stage."""
        from tet.preprocessor import TETPreprocessor
        # Implementation
    
    # ... other stage methods
```

**Execution Sequence**:
1. **Preprocessing**: Load and standardize raw TET data
2. **Descriptive Statistics**: Compute time courses and session metrics
3. **LME Modeling**: Fit mixed effects models for each dimension
4. **Peak/AUC Analysis**: Compare dose effects on intensity metrics
5. **PCA Analysis**: Dimensionality reduction and component interpretation
6. **Clustering Analysis**: Identify discrete experiential states
7. **Figure Generation**: Create all publication-ready visualizations
8. **Report Generation**: Synthesize findings into comprehensive report

### 10.5-10.8: Results Organization

**Purpose**: Maintain clean, organized output structure.

**Directory Structure**:
```
results/tet/
├── tet_preprocessed.csv              # Preprocessed data
├── tet_analysis_report.md            # Final comprehensive report
├── pipeline_execution.log            # Execution log
├── descriptive/                      # Descriptive statistics
├── lme/                              # LME model results
├── peak_auc/                         # Peak and AUC analysis
├── pca/                              # PCA results
├── clustering/                       # Clustering results
└── figures/                          # All visualizations
    └── captions/                     # Figure captions
```

**File Naming Conventions**:
- CSV files: `{analysis_type}_{content_description}.csv`
- Figures: `{content_description}.png`
- Captions: `{figure_name}.txt`

**Implementation**:
- Create directories automatically if they don't exist
- Validate output paths before writing
- Use consistent naming across all scripts
- Document file formats in guide

### 10.9-10.12: Final Report Generation

**Purpose**: Provide publication-ready report with Methods and Results sections.

**Report**: `results/tet/tet_analysis_report.md`

**Format**: APA style + Nature Human Behaviour conventions

**Structure**:
```markdown
# Temporal Experience Tracking Analysis: DMT Dose-Response Study

## Abstract
[Brief summary of study, methods, and key findings]

## Methods

### Participants
[Sample description: N subjects, demographics, exclusions]

### Experimental Design
[Within-subjects design: RS vs DMT, Low vs High dose]

### Data Collection
[TET procedure: 15 dimensions, 30-second intervals, rating scales]

### Data Preprocessing
[Within-subject standardization, composite indices, time windows]

### Statistical Analysis

#### Linear Mixed Effects Models
[Model specification, fixed/random effects, FDR correction]

#### Peak and Area Under Curve Analysis
[Wilcoxon signed-rank tests, effect sizes, bootstrap CIs]

#### Dimensionality Reduction
[PCA procedure, component retention, interpretation]

#### Clustering Analysis
[KMeans/GMM algorithms, model selection, stability assessment]

### Data Availability
[Statement on data access and code availability]

## Results

### Descriptive Patterns
[Temporal dynamics, peak timing, dose-dependent trajectories]

### State and Dose Effects
[LME results: β = X.XX, 95% CI [X.XX, X.XX], p < .001]

### Dose-Response Relationships
[Peak/AUC comparisons: r = X.XX, 95% CI [X.XX, X.XX], p < .001]

### Principal Components of Experience
[PC interpretation, variance explained, temporal dynamics]

### Discrete Experiential States
[Cluster characterization, prevalence, dose sensitivity]

### Convergent Findings
[Patterns consistent across multiple analyses]

## Discussion
[Brief interpretation of main findings]

## References
[Key methodological references]

## Figures
[List of figures with captions]

## Supplementary Materials
[Links to detailed results, code, additional analyses]
```

**Statistical Notation** (APA format):
- Coefficients: β = X.XX, 95% CI [X.XX, X.XX], p < .001
- Effect sizes: r = X.XX, 95% CI [X.XX, X.XX]
- Model fit: F(df1, df2) = X.XX, p < .001
- Descriptive: M = X.XX, SD = X.XX
- Italicize statistical symbols: *β*, *r*, *p*, *F*, *M*, *SD*

**Nature Human Behaviour Style**:
- Clear, concise writing
- Active voice where appropriate
- Present tense for established facts
- Past tense for study procedures
- Precise numerical reporting
- Transparent methods description

### 10.13: Command-Line Interface

**Purpose**: Flexible pipeline execution for development and debugging.

**Options**:
```bash
# Run complete pipeline
python scripts/run_tet_analysis.py

# Run specific stages
python scripts/run_tet_analysis.py --stages preprocessing descriptive lme

# Skip specific stages
python scripts/run_tet_analysis.py --skip-stages clustering

# Run only preprocessing
python scripts/run_tet_analysis.py --preprocessing-only

# Run from specific stage onward
python scripts/run_tet_analysis.py --from-stage lme

# Dry run (validate without executing)
python scripts/run_tet_analysis.py --dry-run

# Verbose output
python scripts/run_tet_analysis.py --verbose

# Custom config file
python scripts/run_tet_analysis.py --config custom_config.yaml
```

**Implementation**:
```python
parser = argparse.ArgumentParser(
    description='Run TET analysis pipeline',
    formatter_class=argparse.RawDescriptionHelpFormatter
)

parser.add_argument(
    '--stages',
    nargs='+',
    choices=['preprocessing', 'descriptive', 'lme', 'peak_auc', 
             'pca', 'clustering', 'figures', 'report'],
    help='Specific stages to run'
)

parser.add_argument(
    '--skip-stages',
    nargs='+',
    choices=['preprocessing', 'descriptive', 'lme', 'peak_auc', 
             'pca', 'clustering', 'figures', 'report'],
    help='Stages to skip'
)

parser.add_argument(
    '--from-stage',
    choices=['preprocessing', 'descriptive', 'lme', 'peak_auc', 
             'pca', 'clustering', 'figures', 'report'],
    help='Start from this stage onward'
)

parser.add_argument('--preprocessing-only', action='store_true')
parser.add_argument('--dry-run', action='store_true')
parser.add_argument('--verbose', action='store_true')
parser.add_argument('--config', type=str, default='config.py')
```

### 10.14: Execution Logging

**Purpose**: Track pipeline execution for debugging and reproducibility.

**Log File**: `results/tet/pipeline_execution.log`

**Log Format**:
```
2025-11-14 10:30:00 - INFO - ========================================
2025-11-14 10:30:00 - INFO - TET Analysis Pipeline Started
2025-11-14 10:30:00 - INFO - ========================================
2025-11-14 10:30:00 - INFO - Configuration:
2025-11-14 10:30:00 - INFO -   Input: data/original/reports/resampled/
2025-11-14 10:30:00 - INFO -   Output: results/tet/
2025-11-14 10:30:00 - INFO -   Stages: all
2025-11-14 10:30:00 - INFO - ========================================
2025-11-14 10:30:00 - INFO - Stage 1/8: Preprocessing
2025-11-14 10:30:00 - INFO -   Loading raw TET data...
2025-11-14 10:30:05 - INFO -   Loaded 19 subjects, 76 sessions
2025-11-14 10:30:05 - INFO -   Validating data integrity...
2025-11-14 10:30:06 - INFO -   Computing within-subject standardization...
2025-11-14 10:30:10 - INFO -   Exporting preprocessed data...
2025-11-14 10:30:12 - INFO -   ✓ Preprocessing complete (12.3s)
2025-11-14 10:30:12 - INFO - ========================================
2025-11-14 10:30:12 - INFO - Stage 2/8: Descriptive Statistics
...
```

**Implementation**:
```python
import logging
from datetime import datetime

def setup_logging(log_path, verbose=False):
    """Configure logging for pipeline execution."""
    level = logging.DEBUG if verbose else logging.INFO
    
    # File handler
    file_handler = logging.FileHandler(log_path, mode='w')
    file_handler.setLevel(logging.DEBUG)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Root logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger
```

### 10.15: Input Validation

**Purpose**: Fail fast with clear error messages if dependencies missing.

**Validation Strategy**:
```python
class PipelineValidator:
    """Validates pipeline dependencies and inputs."""
    
    def validate_stage_inputs(self, stage_name):
        """Validate inputs required for a specific stage."""
        validators = {
            'preprocessing': self._validate_raw_data,
            'descriptive': self._validate_preprocessed_data,
            'lme': self._validate_preprocessed_data,
            'peak_auc': self._validate_preprocessed_data,
            'pca': self._validate_preprocessed_data,
            'clustering': self._validate_preprocessed_data,
            'figures': self._validate_analysis_results,
            'report': self._validate_all_results
        }
        
        validator = validators.get(stage_name)
        if validator:
            return validator()
        return True, ""
    
    def _validate_raw_data(self):
        """Validate raw TET data files exist."""
        required_files = [
            'data/original/reports/resampled/s01_DMT_Session1_DMT.mat',
            # ... other files
        ]
        
        missing = [f for f in required_files if not os.path.exists(f)]
        
        if missing:
            return False, f"Missing raw data files: {missing}"
        return True, ""
    
    def _validate_preprocessed_data(self):
        """Validate preprocessed data exists."""
        required = 'results/tet/tet_preprocessed.csv'
        
        if not os.path.exists(required):
            return False, (
                f"Preprocessed data not found: {required}\n"
                "Run preprocessing stage first: "
                "python scripts/run_tet_analysis.py --stages preprocessing"
            )
        return True, ""
    
    def _validate_analysis_results(self):
        """Validate analysis results exist for figure generation."""
        required = [
            'results/tet/lme/lme_results.csv',
            'results/tet/peak_auc/peak_auc_metrics.csv',
            # ... other files
        ]
        
        missing = [f for f in required if not os.path.exists(f)]
        
        if missing:
            return False, (
                f"Missing analysis results: {missing}\n"
                "Run analysis stages first."
            )
        return True, ""
```

### 10.16-10.19: Documentation Consolidation

**Purpose**: Single authoritative documentation source.

**Document**: `docs/TET_ANALYSIS_GUIDE.md`

**Structure**:
```markdown
# TET Analysis Guide

## Table of Contents
1. Introduction
2. Data Structure and Loading
3. Preprocessing and Standardization
4. Statistical Analysis Methods
5. Dimensionality Reduction and Clustering
6. Visualization and Reporting
7. Usage Instructions
8. Troubleshooting

## 1. Introduction

### Overview
[Purpose of TET analysis system]

### Key Features
- Within-subject standardization
- Mixed effects modeling
- Dimensionality reduction
- State identification
- Automated reporting

### System Requirements
- Python 3.11+
- Required packages (see environment.yml)

## 2. Data Structure and Loading

### Raw Data Format
[Description of .mat files, dimensions, temporal resolution]

### Data Organization
[Directory structure, file naming conventions]

### Loading Procedures
[How data is loaded and validated]

## 3. Preprocessing and Standardization

### Within-Subject Standardization
[Z-score computation across all dimensions and sessions]

### Composite Indices
[Affect, imagery, self indices - formulas and interpretation]

### Time Window Selection
[RS: 0-10 min, DMT: 0-20 min]

### Quality Control
[Validation checks, outlier handling]

## 4. Statistical Analysis Methods

### Linear Mixed Effects Models
[Model specification, interpretation, FDR correction]

### Peak and AUC Analysis
[Wilcoxon tests, effect sizes, bootstrap CIs]

### Contrasts and Comparisons
[High vs Low within DMT, DMT vs RS]

## 5. Dimensionality Reduction and Clustering

### Principal Component Analysis
[PCA procedure, component retention, interpretation]

### Clustering Algorithms
[KMeans, GMM, model selection, stability]

### State Characterization
[Cluster profiles, temporal dynamics, dose effects]

## 6. Visualization and Reporting

### Figure Types
[Time series, coefficients, boxplots, PCA, clustering]

### Report Structure
[Methods, Results, APA notation, Nature style]

### Output Organization
[Directory structure, file naming]

## 7. Usage Instructions

### Quick Start
```bash
# Run complete pipeline
python scripts/run_tet_analysis.py
```

### Stage-Specific Execution
```bash
# Run only preprocessing
python scripts/run_tet_analysis.py --stages preprocessing

# Skip clustering
python scripts/run_tet_analysis.py --skip-stages clustering
```

### Configuration
[How to modify config.py for custom analyses]

## 8. Troubleshooting

### Common Issues
[Missing files, import errors, memory issues]

### Error Messages
[Interpretation and solutions]

### Performance Optimization
[Tips for faster execution]

### Getting Help
[Where to find additional resources]
```

**Content Sources** (to consolidate):
1. `TET_DATA_LOADING_COMPARISON.md` → Section 2
2. `TET_DIMENSIONS_TRACEABILITY.md` → Section 2
3. `TET_TEMPORAL_RESOLUTION.md` → Section 2
4. `tet_clustering_analysis.md` → Section 5
5. `PIPELINE.md` (TET sections) → Sections 3-7

**Archival Strategy**:
- Move consolidated files to `docs/archive/`
- Add README in archive explaining consolidation
- Keep files for reference but mark as deprecated

## Implementation Plan

### Phase 1: Directory Reorganization
1. Create `test/tet/` directory
2. Move testing scripts from `scripts/` to `test/tet/`
3. Update imports in moved scripts
4. Verify functionality

### Phase 2: Orchestration Script
1. Create `scripts/run_tet_analysis.py`
2. Implement `TETAnalysisPipeline` class
3. Add command-line interface
4. Implement logging
5. Add input validation

### Phase 3: Results Organization
1. Create `results/tet/` subdirectories
2. Update all analysis scripts to use new paths
3. Implement `results/tet/figures/captions/` structure
4. Verify all outputs go to correct locations

### Phase 4: Final Report Generation
1. Create report template with Methods and Results sections
2. Implement APA/Nature formatting
3. Integrate with results synthesizer
4. Generate sample report

### Phase 5: Documentation Consolidation
1. Create `docs/TET_ANALYSIS_GUIDE.md`
2. Consolidate content from existing docs
3. Add new sections for orchestration
4. Archive old documentation files
5. Update references in code

### Phase 6: Testing and Validation
1. Run complete pipeline end-to-end
2. Verify all outputs in correct locations
3. Validate report formatting
4. Test stage-specific execution
5. Verify logging and error handling

## Testing Strategy

### Unit Tests
- Test each pipeline stage independently
- Validate input/output paths
- Test error handling

### Integration Tests
- Run complete pipeline
- Verify stage dependencies
- Test skip/select stage options

### Validation Tests
- Verify output file existence
- Check file formats
- Validate report structure

### Performance Tests
- Measure execution time per stage
- Identify bottlenecks
- Optimize slow stages

## Success Criteria

1. ✅ All core modules in `scripts/tet/`
2. ✅ All testing scripts in `test/tet/`
3. ✅ Single orchestration script functional
4. ✅ All outputs in `results/tet/` with clear structure
5. ✅ Final report generated with Methods and Results
6. ✅ Single comprehensive documentation file
7. ✅ Pipeline executes successfully end-to-end
8. ✅ Stage-specific execution works correctly
9. ✅ Logging captures all execution details
10. ✅ Input validation provides clear error messages

## Dependencies

- All existing TET analysis modules
- Python standard library (argparse, logging, pathlib)
- Configuration system (config.py)
- Existing analysis scripts

## Risks and Mitigations

**Risk**: Breaking existing functionality during reorganization
**Mitigation**: Thorough testing after each move, maintain git history

**Risk**: Import path issues after moving scripts
**Mitigation**: Systematic import updates, use relative imports where possible

**Risk**: Users confused by new structure
**Mitigation**: Clear documentation, migration guide, examples

**Risk**: Report formatting inconsistencies
**Mitigation**: Template-based generation, automated formatting checks

## Future Enhancements

1. Configuration file support (YAML/TOML)
2. Parallel stage execution where possible
3. Progress bars for long-running stages
4. HTML report generation option
5. Interactive result exploration dashboard
6. Automated quality checks and warnings
7. Integration with version control for reproducibility
