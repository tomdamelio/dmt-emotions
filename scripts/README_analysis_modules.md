# Analysis Utility Modules - DMT Physiological Analysis Revisions

This directory contains utility modules implementing supervisor-requested revisions to the DMT physiological analysis pipeline. These modules provide enhanced statistical analyses, temporal feature extraction, and publication-ready visualizations.

## Overview

The modules were developed to address specific limitations in the original analysis:
- **Statistical Power**: FDR correction may be overly conservative for SMNA and RVT
- **Temporal Misalignment**: Time-to-time comparisons may miss dose effects due to individual variability in response timing
- **Feature Characterization**: Need to quantify peak amplitude, onset speed, and dose-response relationships
- **Baseline Quantification**: Need to measure magnitude of DMT-induced changes relative to Resting State
- **Publication Standards**: Figures need consistent aesthetics and complete statistical reporting

## Module Descriptions

### 1. alternative_statistics.py

**Purpose**: Provide less conservative statistical criteria for time-to-time comparisons

**Key Functions**:
- `compute_one_tailed_tests()`: One-tailed paired t-tests for directional hypotheses (High > Low)
- `compute_pointwise_uncorrected()`: Pointwise comparisons without FDR correction
- `format_alternative_results()`: Human-readable summary of results

**Scientific Rationale**:
- One-tailed tests have greater power when directional hypothesis is justified (High > Low for arousal)
- Effectively halves p-values compared to two-tailed tests
- Should be tried BEFORE uncorrected tests (more principled approach)
- Uncorrected tests provide exploratory insights but require careful interpretation

**Usage Priority**:
1. **First**: Try one-tailed tests with FDR correction (default in `compute_one_tailed_tests()`)
2. **Second**: Try one-tailed tests without FDR (set `apply_fdr=False`)
3. **Last Resort**: Try uncorrected two-tailed tests (use `compute_pointwise_uncorrected()`)

**Example**:
```python
from scripts.alternative_statistics import compute_one_tailed_tests

# One-tailed test with FDR correction (recommended)
results = compute_one_tailed_tests(
    data_high, data_low, 
    alternative='greater',
    apply_fdr=True  # Default
)

# Check for significant segments
if len(results['significant_segments']) > 0:
    print(f"Found {len(results['significant_segments'])} significant segments")
    print(f"Test type: {results['test_type']}")
```

**Requirements**: 1.1, 1.2, 1.3, 1.4, 1.5

---

### 2. phase_analyzer.py

**Purpose**: Analyze physiological signals averaged within distinct temporal phases

**Key Functions**:
- `define_temporal_phases()`: Define phase boundaries (default: onset 0-3 min, recovery 3-9 min)
- `compute_phase_averages()`: Average signals within each phase for each participant
- `compare_doses_within_phases()`: Paired t-tests comparing High vs Low within phases

**Scientific Rationale**:
- Temporal misalignment may obscure dose differences in pointwise comparisons
- Phase averaging reduces noise and captures overall trajectory
- Flexible boundaries allow sensitivity analyses
- Aligns with pharmacokinetic profile of inhaled DMT (rapid onset, gradual recovery)

**Example**:
```python
from scripts.phase_analyzer import (
    define_temporal_phases,
    compute_phase_averages,
    compare_doses_within_phases
)

# Define phases
phases = define_temporal_phases(
    total_duration_sec=540,
    phase_boundaries=[0, 180, 540]  # Onset: 0-3 min, Recovery: 3-9 min
)

# Compute phase averages
phase_df = compute_phase_averages(
    df, phases, 
    value_column='hr_z',
    window_size_sec=30
)

# Compare doses within phases
comparison_df = compare_doses_within_phases(phase_df)

# Check for significant phases
significant_phases = comparison_df[comparison_df['p_value'] < 0.05]
print(f"Significant phases: {len(significant_phases)}")
```

**Requirements**: 2.1, 2.2, 2.3, 2.4, 2.5

---

### 3. feature_extractor.py

**Purpose**: Extract temporal features from physiological time series

**Key Functions**:
- `extract_peak_amplitude()`: Maximum value and time of occurrence
- `extract_time_to_peak()`: Time to reach maximum (in minutes)
- `extract_threshold_crossings()`: Times when signal crosses 33% and 50% of max
- `extract_all_features()`: Batch extraction for all subjects/sessions

**Scientific Rationale**:
- Peak amplitude reflects magnitude of physiological response
- Time-to-peak reflects onset speed
- Threshold crossings characterize dose-response relationship
- These features capture aspects not visible in time-aligned comparisons

**Example**:
```python
from scripts.feature_extractor import extract_all_features

# Extract features for all subjects/sessions
features_df = extract_all_features(
    df,
    time_column='time',
    value_column='hr_z'
)

# Features extracted:
# - peak_amplitude: Maximum HR value
# - time_to_peak: Time to reach maximum (minutes)
# - t_33: Time to cross 33% of max (minutes)
# - t_50: Time to cross 50% of max (minutes)

# Perform paired t-tests on features
from scipy import stats
high_peak = features_df[features_df['Dose'] == 'High']['peak_amplitude']
low_peak = features_df[features_df['Dose'] == 'Low']['peak_amplitude']
t_stat, p_value = stats.ttest_rel(high_peak, low_peak)
```

**Requirements**: 3.1, 3.2, 3.3, 3.4, 3.5

---

### 4. baseline_comparator.py

**Purpose**: Compare extracted features between DMT conditions and Resting State baseline

**Key Functions**:
- `compare_features_to_baseline()`: Paired t-tests comparing DMT (collapsed) vs RS
- `compute_baseline_summary_stats()`: Descriptive statistics for static baselines
- `visualize_baseline_comparisons()`: Bar plots with significance markers
- `format_baseline_comparison_report()`: Human-readable summary

**Scientific Rationale**:
- Quantifies magnitude of DMT-induced changes independent of dose
- Collapsing doses increases statistical power
- Static baseline statistics provide reference values
- Complements dose-specific analyses

**⚠️ IMPORTANT**: These analyses do NOT address dose-dependent effects. They only quantify overall DMT effects relative to baseline.

**Example**:
```python
from scripts.baseline_comparator import (
    compare_features_to_baseline,
    visualize_baseline_comparisons
)

# Compare DMT features to baseline
comparison_df = compare_features_to_baseline(
    features_df,
    feature_columns=['peak_amplitude', 'time_to_peak', 't_33', 't_50']
)

# Visualize comparisons
visualize_baseline_comparisons(
    comparison_df,
    'results/baseline_comparison.pdf',
    feature_labels={
        'peak_amplitude': 'Peak Amplitude (z-score)',
        'time_to_peak': 'Time to Peak (min)'
    }
)

# Check for significant differences
significant = comparison_df[comparison_df['p_value'] < 0.05]
print(f"Significant features: {list(significant['feature'])}")
```

**Requirements**: 4.1, 4.2, 4.3, 4.4, 4.5

---

### 5. enhanced_visualizer.py

**Purpose**: Add significance markers and ensure homogeneous figure aesthetics

**Key Functions**:
- `add_significance_markers()`: Add *, **, *** based on FDR-corrected p-values
- `apply_homogeneous_aesthetics()`: Consistent fonts, aspect ratios, panel labels
- `save_figure_vector()`: Save in PDF/SVG with optional timestamps
- `get_consistent_aspect_ratio()`: Calculate consistent aspect ratio for time-series
- `format_pvalue_text()`: Format p-values for display

**Global Constants**:
```python
FONT_SIZES = {
    'axis_label': 10,      # Axis labels
    'title': 12,           # Subplot titles
    'tick_label': 8,       # Tick labels
    'panel_label': 14      # Panel identifiers (A, B, C, D)
}
```

**Scientific Rationale**:
- Significance markers provide immediate visual feedback on statistical significance
- Homogeneous aesthetics ensure publication-ready quality
- Vector formats preserve quality at any resolution
- Timestamps enable version tracking and reproducibility

**Example**:
```python
from scripts.enhanced_visualizer import (
    add_significance_markers,
    apply_homogeneous_aesthetics,
    save_figure_vector
)

# Create multi-panel figure
fig, axes = plt.subplots(2, 2, figsize=(10, 8))

# ... plot data ...

# Add significance markers to coefficient plot
add_significance_markers(
    axes[0, 0],
    x_positions=range(5),
    y_positions=coefficients,
    p_values_fdr=p_values_fdr
)

# Apply homogeneous aesthetics
apply_homogeneous_aesthetics(
    fig, axes,
    panel_labels=['A', 'B', 'C', 'D']
)

# Save with timestamp
save_figure_vector(
    fig,
    'results/figures/my_figure',
    formats=['pdf', 'svg'],
    add_timestamp=True  # Adds timestamp to filename
)
```

**Requirements**: 5.1, 5.2, 5.3, 5.4, 5.5, 6.1, 6.2, 6.3, 6.4, 6.5

---

### 6. statistical_reporter.py

**Purpose**: Generate APA-style statistical reports with complete information

**Key Functions**:
- `format_ttest_result()`: Format t-tests in APA style
- `format_lme_result()`: Format LME results in APA style
- `format_correlation_result()`: Format correlations in APA style
- `save_results_table()`: Save results to CSV or text file
- `format_phase_comparison_results()`: Format phase comparisons
- `format_feature_comparison_results()`: Format feature comparisons

**Scientific Rationale**:
- APA style is standard for psychological and neuroscience publications
- Complete reporting (test statistic, df, p-value, effect size) enables meta-analyses
- Standardized formatting reduces manual errors and ensures consistency

**Example**:
```python
from scripts.statistical_reporter import (
    format_ttest_result,
    format_lme_result,
    save_results_table
)

# Format t-test result
result_str = format_ttest_result(
    t_stat=2.45,
    df=17,
    p_value=0.025,
    cohens_d=0.58
)
print(result_str)  # "t(17) = 2.45, p = .025, d = 0.58"

# Format LME result
lme_str = format_lme_result(
    beta=0.45,
    ci_lower=0.12,
    ci_upper=0.78,
    p_fdr=0.008,
    parameter_name='State[T.DMT]'
)
print(lme_str)  # "State[T.DMT]: β = 0.45, 95% CI [0.12, 0.78], p_FDR = .008"

# Save results table
save_results_table(
    results_dict,
    'results/statistical_results.csv',
    format='csv'
)
```

**Requirements**: 8.1, 8.2, 8.3, 8.4, 8.5

---

## Integration with Main Analysis Scripts

These modules are imported and used by the main analysis scripts in `src/`:

- `src/run_ecg_hr_analysis.py`: HR analysis with all enhancements
- `src/run_eda_smna_analysis.py`: SMNA analysis with all enhancements
- `src/run_resp_rvt_analysis.py`: RVT analysis with all enhancements
- `src/run_composite_arousal_index.py`: Composite arousal index with enhancements

Each main script follows this pattern:
1. Load and preprocess data
2. Run LME models with FDR correction (original analysis)
3. Apply alternative statistics (one-tailed tests first, then uncorrected if needed)
4. Perform phase-based analysis
5. Extract temporal features
6. Compare features to baseline
7. Generate enhanced visualizations with significance markers
8. Save complete statistical reports

## Naming Conventions

All modules follow these conventions:
- **Module names**: `snake_case.py` (e.g., `alternative_statistics.py`)
- **Function names**: `snake_case()` (e.g., `compute_one_tailed_tests()`)
- **Variable names**: `snake_case` with domain-specific terminology (e.g., `p_values_fdr`, `cohens_d`)
- **Constants**: `UPPER_SNAKE_CASE` (e.g., `FONT_SIZES`)
- **Output directories**: `snake_case/` (e.g., `alternative_tests/`, `phase_analysis/`)

## Version Control

All modules include:
- **Comprehensive docstrings**: Google-style with scientific rationale
- **Type hints**: All function signatures include type annotations
- **Requirements references**: Each function documents which requirements it addresses
- **Examples**: Practical usage examples in docstrings
- **Error handling**: Input validation with informative error messages

## Testing

Unit tests for these modules are located in `test/test_utility_modules_smoke.py`. The tests verify:
- Basic functionality with synthetic data
- Input validation and error handling
- Output format correctness
- Integration with main analysis scripts

Run tests with:
```bash
micromamba run -n dmt-emotions pytest test/test_utility_modules_smoke.py -v
```

## Backward Compatibility

All enhancements are additive and do not modify existing analysis outputs. The original FDR-corrected results are preserved, and new analyses are saved to separate subdirectories:
- `alternative_tests/`: One-tailed and uncorrected test results
- `phase_analysis/`: Phase-averaged comparisons
- `features/`: Extracted temporal features
- `baseline_comparison/`: DMT vs RS comparisons

This ensures reproducibility and allows comparison between original and revised analyses.

## References

For detailed scientific rationale and design decisions, see:
- `.kiro/specs/dmt-analysis-revisions/requirements.md`: Complete requirements specification
- `.kiro/specs/dmt-analysis-revisions/design.md`: Detailed design document with correctness properties
- `.kiro/specs/dmt-analysis-revisions/tasks.md`: Implementation task list

## Citation

If you use these analysis modules in your research, please cite the DMT-Emotions project and acknowledge the specific modules used.

---

**Last Updated**: 2026-01-11  
**Version**: 1.0.0  
**Author**: DMT Analysis Pipeline Team
