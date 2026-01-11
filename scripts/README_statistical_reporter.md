# Statistical Reporter Module

## Overview

The `statistical_reporter.py` module provides APA-style formatting functions for statistical results in the DMT physiological analysis pipeline. It ensures complete and standardized reporting across all analyses.

## Functions

### Core Formatting Functions

#### `format_ttest_result(t_stat, df, p_value, cohens_d, mean_diff=None)`
Formats t-test results in APA style.

**Example:**
```python
from scripts.statistical_reporter import format_ttest_result

result = format_ttest_result(2.45, 17, 0.025, 0.58)
# Output: "t(17) = 2.45, p = .025, d = 0.58"
```

#### `format_lme_result(beta, ci_lower, ci_upper, p_fdr, parameter_name)`
Formats Linear Mixed-Effects model results in APA style.

**Example:**
```python
from scripts.statistical_reporter import format_lme_result

result = format_lme_result(0.45, 0.12, 0.78, 0.008, 'State[T.DMT]')
# Output: "State[T.DMT]: β = 0.45, 95% CI [0.12, 0.78], p_FDR = .008"
```

#### `format_correlation_result(r, n, p_fdr)`
Formats correlation results in APA style.

**Example:**
```python
from scripts.statistical_reporter import format_correlation_result

result = format_correlation_result(0.67, 18, 0.002)
# Output: "r(16) = .67, p_FDR = .002"
```

#### `save_results_table(results_dict, output_path, format='csv')`
Saves statistical results to CSV or text file.

**Example:**
```python
from scripts.statistical_reporter import save_results_table

results = {
    'phase1': {'t_stat': 2.45, 'p_value': 0.025, 'cohens_d': 0.58},
    'phase2': {'t_stat': 1.89, 'p_value': 0.075, 'cohens_d': 0.45}
}

save_results_table(results, 'results/phase_analysis.csv')
```

### Convenience Functions

#### `format_phase_comparison_results(phase_results)`
Formats multiple phase comparisons into a readable text report.

**Example:**
```python
from scripts.statistical_reporter import format_phase_comparison_results
import pandas as pd

phase_df = pd.DataFrame({
    'phase': [0, 1],
    'phase_label': ['Onset', 'Recovery'],
    'State': ['DMT', 'DMT'],
    't_stat': [2.45, 1.89],
    'p_value': [0.025, 0.075],
    'cohens_d': [0.58, 0.45],
    'mean_high': [1.23, 0.98],
    'mean_low': [0.67, 0.54],
    'sem_high': [0.12, 0.10],
    'sem_low': [0.08, 0.09]
})

report = format_phase_comparison_results(phase_df)
print(report)
```

#### `format_feature_comparison_results(feature_results)`
Formats multiple feature comparisons into a readable text report.

**Example:**
```python
from scripts.statistical_reporter import format_feature_comparison_results
import pandas as pd

feature_df = pd.DataFrame({
    'feature': ['peak_amplitude', 'time_to_peak'],
    't_stat': [3.21, -1.45],
    'df': [17, 17],
    'p_value': [0.005, 0.165],
    'cohens_d': [0.76, -0.34],
    'mean_high': [2.45, 3.2],
    'mean_low': [1.67, 3.8],
    'std_high': [0.45, 0.8],
    'std_low': [0.38, 0.9]
})

report = format_feature_comparison_results(feature_df)
print(report)
```

## Integration with Analysis Pipeline

This module is designed to be imported and used in the main analysis scripts:

```python
# In src/run_ecg_hr_analysis.py
from scripts.statistical_reporter import (
    format_ttest_result,
    format_lme_result,
    save_results_table
)

# Format and save results
ttest_str = format_ttest_result(t_stat, df, p_value, cohens_d)
print(f"Dose comparison: {ttest_str}")

save_results_table(all_results, 'results/ecg/hr/statistical_summary.csv')
```

## Requirements Validation

This module satisfies all requirements from Requirement 8:

- ✅ 8.1: T-test reporting includes t, df, p, Cohen's d
- ✅ 8.2: LME reporting includes β, 95% CI, p_FDR
- ✅ 8.3: Correlation reporting includes r, n, p_FDR
- ✅ 8.4: CSV output includes all statistics in separate columns
- ✅ 8.5: Text reports use APA style formatting

## APA Style Conventions

The module follows standard APA formatting conventions:

- P-values < 0.001 are reported as "p < .001"
- P-values ≥ 0.001 are reported with 3 decimal places (e.g., "p = .025")
- Leading zeros are omitted for values that cannot exceed 1 (e.g., ".67" not "0.67")
- Degrees of freedom are reported in parentheses
- Effect sizes (Cohen's d) are always included with t-tests
- Confidence intervals are reported as [lower, upper]
