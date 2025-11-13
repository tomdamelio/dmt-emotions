# Design Document: Requirement 3 - Descriptive Statistics and Time Course Analysis

## Overview

This design document describes the implementation of descriptive statistics and time course analysis for TET data. The system will compute group-level time courses (mean ± SEM) and session-level summary metrics to characterize temporal dynamics of subjective experiences.

## Architecture

### Components

1. **TETTimeCourseAnalyzer**: Computes group-level time courses (mean ± SEM by time bin)
2. **TETSessionMetrics**: Computes session-level summary metrics (peak, AUC, slopes)
3. **Main Script**: Orchestrates analysis and exports results

### Data Flow

```
Preprocessed Data (tet_preprocessed.csv)
    ↓
TETTimeCourseAnalyzer
    ├→ Group by (state, dose, t_bin, dimension)
    ├→ Compute mean and SEM
    └→ Export time_course_*.csv
    
TETSessionMetrics
    ├→ Group by (subject, session_id, state, dose, dimension)
    ├→ Compute peak, time_to_peak, AUC, slopes
    └→ Export session_metrics_*.csv
```

## Component Design

### 1. TETTimeCourseAnalyzer

**Purpose**: Compute group-level time courses for visualization and analysis.

**Input**: 
- Preprocessed TET data (DataFrame)
- List of dimensions to analyze (z-scored dimensions + composite indices)

**Processing**:
1. For each dimension:
   - Group by (state, dose, t_bin)
   - Compute mean and SEM across subjects
   - Create long-format DataFrame

**Output**:
- CSV file with columns: `dimension`, `state`, `dose`, `t_bin`, `t_sec`, `mean`, `sem`, `n`

**Key Methods**:
- `compute_time_course(dimension)`: Compute time course for one dimension
- `compute_all_time_courses()`: Compute for all dimensions
- `export_time_courses(output_dir)`: Save results to CSV

### 2. TETSessionMetrics

**Purpose**: Compute session-level summary metrics for statistical modeling.

**Input**:
- Preprocessed TET data (DataFrame)
- List of dimensions to analyze

**Processing**:
For each (subject, session_id, state, dose, dimension) combination:

1. **Peak Value**: Maximum value in the session
2. **Time to Peak**: Time (in seconds) when peak occurs
3. **AUC (0-9 min)**: Area under curve from 0-540 seconds using trapezoidal rule
4. **Slope (0-2 min)**: Linear regression slope from 0-120 seconds
5. **Slope (5-9 min)**: Linear regression slope from 300-540 seconds

**Output**:
- CSV file with columns: `subject`, `session_id`, `state`, `dose`, `dimension`, `peak_value`, `time_to_peak`, `auc_0_9min`, `slope_0_2min`, `slope_5_9min`

**Key Methods**:
- `compute_session_metrics(subject, session_id, state, dose, dimension)`: Compute metrics for one session
- `compute_all_session_metrics()`: Compute for all sessions and dimensions
- `export_session_metrics(output_dir)`: Save results to CSV

## Implementation Details

### Time Course Computation

```python
# Group by state, dose, t_bin
grouped = data.groupby(['state', 'dose', 't_bin'])

# Compute mean and SEM
time_course = grouped[dimension].agg([
    ('mean', 'mean'),
    ('sem', lambda x: x.std(ddof=1) / np.sqrt(len(x))),
    ('n', 'count')
]).reset_index()
```

### Session Metrics Computation

```python
# For each session
session_data = data[
    (data['subject'] == subject) &
    (data['session_id'] == session_id) &
    (data['state'] == state)
]

# Peak value and time
peak_value = session_data[dimension].max()
peak_idx = session_data[dimension].idxmax()
time_to_peak = session_data.loc[peak_idx, 't_sec']

# AUC (0-9 min) using trapezoidal rule
auc_data = session_data[session_data['t_sec'] <= 540]
auc = np.trapz(auc_data[dimension], auc_data['t_sec'])

# Slope (0-2 min) using linear regression
slope_0_2_data = session_data[session_data['t_sec'] <= 120]
slope_0_2 = np.polyfit(slope_0_2_data['t_sec'], slope_0_2_data[dimension], 1)[0]

# Slope (5-9 min) using linear regression
slope_5_9_data = session_data[
    (session_data['t_sec'] >= 300) & (session_data['t_sec'] <= 540)
]
slope_5_9 = np.polyfit(slope_5_9_data['t_sec'], slope_5_9_data[dimension], 1)[0]
```

## Data Schema

### Time Course Output

| Column | Type | Description |
|--------|------|-------------|
| dimension | str | Dimension name (e.g., 'pleasantness_z') |
| state | str | 'RS' or 'DMT' |
| dose | str | Dose level |
| t_bin | int | Time bin index (0-based) |
| t_sec | float | Time in seconds |
| mean | float | Mean value across subjects |
| sem | float | Standard error of the mean |
| n | int | Number of observations |

### Session Metrics Output

| Column | Type | Description |
|--------|------|-------------|
| subject | str | Subject ID |
| session_id | int | Session number |
| state | str | 'RS' or 'DMT' |
| dose | str | Dose level |
| dimension | str | Dimension name |
| peak_value | float | Maximum value in session |
| time_to_peak | float | Time (seconds) when peak occurs |
| auc_0_9min | float | Area under curve from 0-540s |
| slope_0_2min | float | Linear slope from 0-120s |
| slope_5_9min | float | Linear slope from 300-540s |

## Dimensions to Analyze

### Z-scored Dimensions (15)
All dimensions with `_z` suffix:
- pleasantness_z, unpleasantness_z, emotional_intensity_z
- elementary_imagery_z, complex_imagery_z
- auditory_z, interoception_z
- bliss_z, anxiety_z, entity_z
- selfhood_z, disembodiment_z
- salience_z, temporality_z, general_intensity_z

### Composite Indices (3)
- affect_index_z
- imagery_index_z
- self_index_z

**Total: 18 dimensions**

## Error Handling

1. **Missing Data**: Skip sessions with insufficient data points
2. **Insufficient Points for Slopes**: Return NaN if < 2 points in time window
3. **Invalid Time Ranges**: Validate that time windows exist in data

## Output Files

```
results/tet/descriptive/
├── time_course_all_dimensions.csv      # All dimensions in long format
└── session_metrics_all_dimensions.csv  # All session metrics in long format
```

## Testing Strategy

1. **Unit Tests**:
   - Test time course computation with synthetic data
   - Test session metrics computation with known values
   - Test edge cases (single point, missing data)

2. **Integration Tests**:
   - Test complete pipeline with real preprocessed data
   - Verify output file structure
   - Verify all dimensions processed

3. **Validation**:
   - Verify SEM calculation: SEM = std / sqrt(n)
   - Verify AUC calculation with simple geometric shapes
   - Verify slope calculation with known linear data

## Performance Considerations

- Use vectorized operations where possible
- Process dimensions in parallel if needed
- Use efficient groupby operations
- Minimize memory usage by processing one dimension at a time if needed

## Dependencies

- pandas: Data manipulation
- numpy: Numerical computations (trapz, polyfit)
- scipy: Statistical functions (if needed)

## Usage Example

```python
from tet.time_course import TETTimeCourseAnalyzer
from tet.session_metrics import TETSessionMetrics
import pandas as pd

# Load preprocessed data
data = pd.read_csv('results/tet/preprocessed/tet_preprocessed.csv')

# Compute time courses
tc_analyzer = TETTimeCourseAnalyzer(data)
time_courses = tc_analyzer.compute_all_time_courses()
tc_analyzer.export_time_courses('results/tet/descriptive')

# Compute session metrics
sm_analyzer = TETSessionMetrics(data)
session_metrics = sm_analyzer.compute_all_session_metrics()
sm_analyzer.export_session_metrics('results/tet/descriptive')
```

## Notes

- Time courses are computed using z-scored dimensions only (not raw dimensions)
- Session metrics are computed for the analysis windows (RS: 0-10 min, DMT: 0-20 min)
- AUC is computed only for 0-9 minutes to match the LME analysis window
- Slopes provide information about early (0-2 min) and late (5-9 min) dynamics
