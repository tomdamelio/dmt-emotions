# Design Document: Requirement 5 - Peak and AUC Analysis

## Overview

This design document describes the implementation of peak and AUC (Area Under the Curve) analysis for TET dimensions, comparing dose effects using non-parametric statistical tests.

## Requirements Reference

- **Requirement 5.1**: Wilcoxon signed-rank tests for peak values (High vs Low dose)
- **Requirement 5.2**: Wilcoxon signed-rank tests for time_to_peak (High vs Low dose)
- **Requirement 5.3**: Wilcoxon signed-rank tests for AUC_0_9 (High vs Low dose)
- **Requirement 5.4**: BH-FDR correction across dimensions for each metric type
- **Requirement 5.5**: Effect size r with 95% CI using bootstrap (2000 iterations)
- **Requirement 5.6**: Export results as CSV with specified columns

## Architecture

### Component Structure

```
scripts/
├── tet/
│   └── peak_auc_analyzer.py    # Core analysis module
└── compute_peak_auc.py          # Main execution script
```

### Data Flow

```
Preprocessed Data (tet_preprocessed.csv)
    ↓
[Metric Computation]
    ├── Peak values (max z-score per session)
    ├── Time to peak (minutes)
    └── AUC 0-9 min (trapezoidal integration)
    ↓
[Paired Data Preparation]
    └── Pivot by subject: High dose vs Low dose
    ↓
[Statistical Testing]
    ├── Wilcoxon signed-rank tests
    ├── Effect size r with bootstrap CI
    └── BH-FDR correction per metric
    ↓
[Export Results]
    ├── peak_auc_metrics.csv
    └── peak_auc_tests.csv
```

## Components and Interfaces

### 1. TETPeakAUCAnalyzer Class

**Purpose**: Compute metrics and perform statistical comparisons

**Key Methods**:

```python
class TETPeakAUCAnalyzer:
    def __init__(self, data: pd.DataFrame, dimensions: List[str])
    def compute_metrics(self) -> pd.DataFrame
    def perform_tests(self) -> pd.DataFrame
    def export_results(self, output_dir: str) -> Dict[str, str]
    
    # Private methods
    def _compute_effect_size_r(self, x, y, n_bootstrap=2000) -> Tuple[float, float, float]
```

**Attributes**:
- `data`: Preprocessed TET data
- `dimensions`: List of z-scored dimensions
- `metrics_df`: Computed metrics (peak, time_to_peak, AUC)
- `results_df`: Statistical test results

### 2. Main Script (compute_peak_auc.py)

**Purpose**: Orchestrate the analysis workflow

**Workflow**:
1. Load preprocessed data
2. Initialize analyzer with z-scored dimensions
3. Compute metrics for all sessions
4. Perform statistical tests
5. Export results

## Data Models

### Metrics DataFrame

**Schema**: `peak_auc_metrics.csv`

| Column | Type | Description |
|--------|------|-------------|
| subject | str | Subject identifier |
| session | str | Session identifier (e.g., "DMT_Baja_1") |
| dose | str | Dose level ("Baja" or "Alta") |
| dimension | str | Dimension name (e.g., "pleasantness_z") |
| peak | float | Maximum z-score value |
| time_to_peak_min | float | Time to peak in minutes |
| auc_0_9 | float | Area under curve (0-9 minutes) |

**Example**:
```
subject,session,dose,dimension,peak,time_to_peak_min,auc_0_9
sub-01,DMT_Baja_1,Baja,pleasantness_z,1.23,4.5,8.45
sub-01,DMT_Alta_1,Alta,pleasantness_z,2.15,3.2,14.32
```

### Test Results DataFrame

**Schema**: `peak_auc_tests.csv`

| Column | Type | Description |
|--------|------|-------------|
| dimension | str | Dimension name |
| metric | str | Metric type ("peak", "time_to_peak_min", "auc_0_9") |
| n_pairs | int | Number of paired observations |
| statistic | float | Wilcoxon test statistic |
| p_value | float | Raw p-value |
| p_fdr | float | FDR-corrected p-value |
| effect_r | float | Effect size r |
| ci_lower | float | 95% CI lower bound |
| ci_upper | float | 95% CI upper bound |
| significant | bool | Significance after FDR correction |

**Example**:
```
dimension,metric,n_pairs,statistic,p_value,p_fdr,effect_r,ci_lower,ci_upper,significant
pleasantness_z,peak,18,45.0,0.0023,0.0156,0.68,0.42,0.85,True
pleasantness_z,time_to_peak_min,18,78.0,0.1234,0.2456,-0.15,-0.45,0.18,False
```

## Metric Computation Details

### 1. Peak Value

**Definition**: Maximum z-score within the analysis window (0-9 minutes)

**Computation**:
```python
peak = session_data[dimension].max()
```

**Interpretation**: 
- Higher peak = more intense experience
- Z-score units relative to RS baseline

### 2. Time to Peak

**Definition**: Time (in minutes) when peak value occurs

**Computation**:
```python
peak_idx = session_data[dimension].argmax()
time_to_peak_min = session_data['t_sec'].iloc[peak_idx] / 60
```

**Interpretation**:
- Earlier peak = faster onset
- Later peak = delayed effect

### 3. Area Under Curve (AUC)

**Definition**: Total area under the z-score curve from 0-9 minutes

**Computation**:
```python
from scipy.integrate import trapezoid

time_min = session_data['t_sec'] / 60
z_values = session_data[dimension]
auc_0_9 = trapezoid(z_values, time_min)
```

**Units**: z-score × minutes

**Interpretation**:
- Positive AUC: Elevated above baseline
- Negative AUC: Reduced below baseline
- Magnitude: Combined intensity and duration

**Why AUC works with z-scores**:
- Z-score of 0 = RS baseline (reference point)
- Positive values = above baseline
- Negative values = below baseline
- AUC captures cumulative deviation from baseline

## Statistical Testing

### Wilcoxon Signed-Rank Test

**Purpose**: Compare paired observations (High vs Low dose within subjects)

**Assumptions**:
- Paired data (same subjects in both conditions)
- Non-parametric (no normality assumption)
- Tests for differences in central tendency

**Implementation**:
```python
from scipy.stats import wilcoxon

# Paired data: High dose vs Low dose for same subjects
statistic, p_value = wilcoxon(high_dose, low_dose)
```

### Effect Size Computation

**Effect size r**:
```
r = Z / sqrt(N)
```

Where:
- Z = z-score from Wilcoxon test
- N = number of paired observations

**Interpretation**:
- |r| < 0.3: Small effect
- 0.3 ≤ |r| < 0.5: Medium effect
- |r| ≥ 0.5: Large effect

**Bootstrap Confidence Intervals**:
1. Resample paired observations with replacement (2000 iterations)
2. Compute effect size for each bootstrap sample
3. Extract 2.5th and 97.5th percentiles for 95% CI

### Multiple Testing Correction

**Method**: Benjamini-Hochberg FDR correction

**Application**: Separate correction for each metric type
- Peak values: 15 tests (one per dimension)
- Time to peak: 15 tests
- AUC: 15 tests

**Implementation**:
```python
from statsmodels.stats.multitest import multipletests

reject, p_fdr, _, _ = multipletests(p_values, method='fdr_bh')
```

## Error Handling

### Missing Data

**Issue**: Subject missing one dose condition

**Solution**: 
- Use `pivot_table` with `dropna()` to keep only complete pairs
- Log warning for subjects with incomplete data
- Require minimum n=3 pairs for valid test

### Insufficient Data

**Issue**: Too few paired observations

**Solution**:
- Skip test if n < 3 pairs
- Log warning with dimension name
- Set results to NaN

### Test Failures

**Issue**: Wilcoxon test fails (e.g., all zeros)

**Solution**:
- Catch exception
- Set statistic and p_value to NaN
- Log warning
- Continue with other dimensions

### Bootstrap Failures

**Issue**: Effect size computation fails

**Solution**:
- Catch exception
- Set effect_r and CI to NaN
- Log warning
- Continue with other tests

## Output Files

### Location

```
results/tet/peak_auc/
├── peak_auc_metrics.csv    # Raw metrics for all sessions
└── peak_auc_tests.csv      # Statistical test results
```

### File Formats

Both files use CSV format with:
- UTF-8 encoding
- Comma delimiter
- Header row
- No index column

## Testing Strategy

### Unit Tests

1. **Metric Computation**:
   - Test peak detection with known data
   - Test time-to-peak calculation
   - Test AUC computation with simple trapezoid

2. **Effect Size**:
   - Test r computation with known Z-scores
   - Test bootstrap CI generation
   - Test edge cases (all equal values)

3. **Data Preparation**:
   - Test pivot operation
   - Test handling of missing data
   - Test minimum sample size check

### Integration Tests

1. **End-to-End Workflow**:
   - Load sample preprocessed data
   - Compute metrics
   - Perform tests
   - Verify output files exist

2. **Statistical Validity**:
   - Compare results with manual calculations
   - Verify FDR correction
   - Check effect size ranges

### Validation Tests

1. **Data Integrity**:
   - Verify all dimensions processed
   - Check for missing values
   - Validate paired data structure

2. **Statistical Assumptions**:
   - Check sample sizes
   - Verify paired structure
   - Validate bootstrap convergence

## Performance Considerations

### Computational Complexity

- Metric computation: O(n × d × t)
  - n = sessions
  - d = dimensions
  - t = time bins
  
- Statistical tests: O(d × m × b)
  - d = dimensions
  - m = metrics (3)
  - b = bootstrap iterations (2000)

### Expected Runtime

- Metric computation: < 5 seconds
- Statistical tests: 1-2 minutes (due to bootstrap)
- Total: ~2 minutes

### Memory Usage

- Metrics DataFrame: ~1 MB (540 sessions × 15 dimensions)
- Results DataFrame: < 1 KB (45 tests)
- Bootstrap arrays: ~10 MB temporary

## Dependencies

### Python Packages

- `pandas`: Data manipulation
- `numpy`: Numerical operations
- `scipy`: Statistical tests and integration
- `statsmodels`: Multiple testing correction

### Internal Modules

- `config`: TET dimension definitions
- `tet.peak_auc_analyzer`: Core analysis class

## Future Enhancements

1. **Additional Metrics**:
   - Peak width (duration above threshold)
   - Onset time (time to reach threshold)
   - Offset time (time to return to baseline)

2. **Visualization**:
   - Peak distribution plots
   - AUC comparison plots
   - Effect size forest plots

3. **Composite Indices**:
   - Extend analysis to composite indices
   - Compare with individual dimensions

4. **Sensitivity Analysis**:
   - Test different time windows
   - Vary bootstrap iterations
   - Compare with parametric tests
