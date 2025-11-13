# Design Document: Requirement 4b - Time Series Visualization with Statistical Annotations

## Overview

This design document describes the implementation of time series visualization for TET data with statistical annotations. The system will generate publication-ready figures showing dose effects over time with visual indicators of statistical significance.

## Visual Design

### Figure Layout

```
┌─────────────────────────────────────────────────────────────┐
│  Legend: 20mg (blue) | 40mg (red) | DMT+RS (p<.05) | 40mg>20mg (p<.05) │
│                      DMT Onset (dashed line)                │
├─────────────────────────────────────────────────────────────┤
│  Dimension 1 (Strongest Effect)                             │
│  [Time series plot with annotations]                        │
├─────────────────────────────────────────────────────────────┤
│  Dimension 2                                                │
│  [Time series plot with annotations]                        │
├─────────────────────────────────────────────────────────────┤
│  ...                                                        │
└─────────────────────────────────────────────────────────────┘
```

### Individual Plot Components

1. **RS Baseline Point**: First point (t=0) shows mean across entire RS condition
2. **DMT Time Series**: Points from t=0 to t=20 minutes
3. **Shaded Error**: SEM shading around mean trajectories
4. **Grey Background**: Time points where DMT ≠ RS (main effect, p<0.05)
5. **Black Bars**: Time points with State:Dose interaction (p<0.05)
6. **Dashed Line**: Vertical line at DMT onset

## Architecture

### Components

1. **TETTimeSeriesVisualizer**: Main visualization class
2. **StatisticalAnnotator**: Handles significance testing and annotations
3. **Main Script**: Orchestrates visualization generation

### Data Flow

```
Preprocessed Data + LME Results
    ↓
Compute RS Baseline (mean across entire RS)
    ↓
Compute DMT Time Courses (mean ± SEM by dose)
    ↓
Identify Significant Time Points
    ├→ Main Effect: DMT vs RS (t-tests per time bin)
    └→ Interaction: State:Dose from LME results
    ↓
Order Dimensions by State Effect Strength
    ↓
Generate Multi-Panel Figure
    ├→ Plot time series with SEM shading
    ├→ Add grey background for main effects
    ├→ Add black bars for interactions
    └→ Add RS baseline and DMT onset line
    ↓
Export High-Resolution PNG
```

## Component Design

### 1. TETTimeSeriesVisualizer

**Purpose**: Generate time series plots with statistical annotations.

**Input**:
- Preprocessed TET data
- LME results (for ordering and interaction effects)
- Time course data (mean ± SEM)

**Processing**:

1. **Compute RS Baseline**:
   - For each dimension and dose: mean across all RS time points
   - This becomes the first point (t=0) in the plot

2. **Compute DMT Time Courses**:
   - For each dimension, dose, and time bin: mean ± SEM
   - Use data from t_bin 0-300 (0-20 minutes)

3. **Identify Significant Time Points**:
   - **Main Effect**: For each time bin, test if DMT (both doses) ≠ RS baseline
     - Use independent t-test comparing DMT values to RS baseline
     - Apply FDR correction across time bins
   - **Interaction Effect**: Extract from LME results
     - Use State:Dose interaction p-values per dimension
     - Mark time bins where interaction is significant

4. **Order Dimensions**:
   - Extract State effect coefficients from LME results
   - Sort dimensions by absolute value of coefficient (descending)

5. **Generate Figure**:
   - Create multi-panel figure (3 columns × 5 rows = 15 dimensions)
   - For each dimension:
     - Plot RS baseline point at t=0
     - Plot DMT time series (blue for 20mg, red for 40mg)
     - Add SEM shading
     - Add grey background for significant main effects
     - Add black bars for significant interactions
     - Add dashed vertical line at DMT onset

**Output**:
- High-resolution PNG figure (300 DPI, 12×16 inches)

**Key Methods**:
- `compute_rs_baseline()`: Compute RS baseline for each dimension
- `compute_dmt_timecourses()`: Compute DMT time courses with SEM
- `identify_significant_timepoints()`: Statistical testing for annotations
- `order_dimensions_by_effect()`: Sort dimensions by LME coefficients
- `plot_dimension()`: Plot single dimension panel
- `generate_figure()`: Create complete multi-panel figure
- `export_figure()`: Save as high-resolution PNG

### 2. StatisticalAnnotator

**Purpose**: Perform statistical tests for time point annotations.

**Input**:
- Preprocessed data
- LME results

**Processing**:

1. **Main Effect Testing** (per dimension, per time bin):
   ```python
   # For each time bin in DMT
   dmt_values = data[(data['state'] == 'DMT') & (data['t_bin'] == bin)]
   rs_baseline = data[data['state'] == 'RS'][dimension].mean()
   
   # One-sample t-test: DMT vs RS baseline
   t_stat, p_value = scipy.stats.ttest_1samp(dmt_values[dimension], rs_baseline)
   ```

2. **Interaction Effect Extraction**:
   - Extract State:Dose interaction p-values from LME results
   - If p < 0.05, mark all time bins for that dimension

**Output**:
- DataFrame with columns: dimension, t_bin, main_effect_sig, interaction_sig

**Key Methods**:
- `test_main_effects()`: Test DMT vs RS at each time bin
- `extract_interaction_effects()`: Get interaction significance from LME
- `apply_fdr_correction()`: Apply FDR correction to main effect tests

## Implementation Details

### RS Baseline Computation

```python
# Compute RS baseline (mean across entire RS condition)
rs_baseline = data[data['state'] == 'RS'].groupby(['dose', 'dimension'])[dimension].mean()
```

### DMT Time Course Computation

```python
# Compute DMT time courses
dmt_tc = data[data['state'] == 'DMT'].groupby(['dose', 't_bin', 'dimension']).agg({
    dimension: ['mean', lambda x: x.std(ddof=1) / np.sqrt(len(x))]
})
```

### Plotting with Matplotlib

```python
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

fig, axes = plt.subplots(5, 3, figsize=(12, 16), dpi=300)

for idx, dimension in enumerate(ordered_dimensions):
    ax = axes.flatten()[idx]
    
    # Plot RS baseline
    ax.scatter(0, rs_baseline_low, color='blue', s=50, zorder=3)
    ax.scatter(0, rs_baseline_high, color='red', s=50, zorder=3)
    
    # Plot DMT time series with SEM shading
    ax.plot(time_min, mean_low, color='blue', linewidth=2, label='20mg')
    ax.fill_between(time_min, mean_low - sem_low, mean_low + sem_low, 
                     color='blue', alpha=0.3)
    
    ax.plot(time_min, mean_high, color='red', linewidth=2, label='40mg')
    ax.fill_between(time_min, mean_high - sem_high, mean_high + sem_high,
                     color='red', alpha=0.3)
    
    # Add grey background for main effects
    for t_bin in sig_main_effect_bins:
        ax.axvspan(t_bin * 4 / 60, (t_bin + 1) * 4 / 60, 
                   color='grey', alpha=0.2, zorder=0)
    
    # Add black bars for interactions
    if interaction_significant:
        ax.plot([0, 20], [y_max, y_max], color='black', linewidth=3)
    
    # Add dashed line at DMT onset
    ax.axvline(x=0, color='grey', linestyle='--', linewidth=1)
    
    # Labels and formatting
    ax.set_title(dimension.replace('_z', '').replace('_', ' ').title())
    ax.set_xlabel('Time (minutes)')
    ax.set_ylabel('Z-scored Intensity')
    ax.set_xlim(-1, 20)
```

## Data Requirements

### Input Data

1. **Preprocessed TET Data**: `results/tet/preprocessed/tet_preprocessed.csv`
   - Columns: subject, session_id, state, dose, t_bin, t_sec, dimension_z values

2. **LME Results**: `results/tet/lme/lme_results.csv`
   - Columns: dimension, effect, beta, p_value, p_fdr, significant
   - Used for: ordering dimensions and interaction effects

3. **Time Course Data**: `results/tet/descriptive/time_course_all_dimensions.csv`
   - Columns: dimension, state, dose, t_bin, t_sec, mean, sem, n
   - Used for: plotting mean trajectories

### Output

- **Figure**: `results/tet/figures/time_series_with_annotations.png`
  - Format: PNG
  - Resolution: 300 DPI
  - Size: 12×16 inches (3600×4800 pixels)

## Visual Specifications

### Colors

- **20mg (Low Dose)**: Blue (#4472C4)
- **40mg (High Dose)**: Red (#C44444)
- **RS Baseline**: Grey markers
- **Main Effect Background**: Grey (#CCCCCC, alpha=0.2)
- **Interaction Bar**: Black (#000000)
- **DMT Onset Line**: Grey dashed (#888888)

### Layout

- **Figure Size**: 12×16 inches
- **Subplot Grid**: 5 rows × 3 columns
- **Spacing**: hspace=0.3, wspace=0.3
- **Font Sizes**:
  - Title: 10pt
  - Axis labels: 8pt
  - Tick labels: 7pt
  - Legend: 8pt

### Axes

- **X-axis**: Time in minutes (-1 to 20)
  - Ticks: 0, 5, 10, 15, 20
- **Y-axis**: Z-scored intensity
  - Auto-scaled per dimension
  - Ticks: 5 evenly spaced values

## Statistical Testing

### Main Effect Test (per time bin)

- **Null Hypothesis**: DMT mean = RS baseline
- **Test**: One-sample t-test
- **Correction**: BH-FDR across time bins within each dimension
- **Threshold**: p_fdr < 0.05

### Interaction Effect

- **Source**: LME results (State:Dose interaction)
- **Threshold**: p_fdr < 0.05
- **Annotation**: If significant, mark entire dimension with black bar

## Error Handling

1. **Missing Data**: Skip dimensions with insufficient data
2. **Failed Tests**: Log warning and continue
3. **Plot Errors**: Log error and skip dimension

## Dependencies

- **matplotlib**: Plotting
- **seaborn**: Color palettes (optional)
- **scipy**: Statistical tests
- **pandas**: Data manipulation
- **numpy**: Numerical computations

## Usage Example

```python
from tet.time_series_visualizer import TETTimeSeriesVisualizer
import pandas as pd

# Load data
data = pd.read_csv('results/tet/preprocessed/tet_preprocessed.csv')
lme_results = pd.read_csv('results/tet/lme/lme_results.csv')
time_courses = pd.read_csv('results/tet/descriptive/time_course_all_dimensions.csv')

# Create visualizer
visualizer = TETTimeSeriesVisualizer(data, lme_results, time_courses)

# Generate figure
fig = visualizer.generate_figure()

# Export
visualizer.export_figure('results/tet/figures/time_series_with_annotations.png')
```

## Notes

- RS baseline is computed as the mean across the entire RS condition (not just first time bin)
- DMT onset is marked at t=0 (first DMT time bin)
- Dimensions are ordered by absolute value of State effect coefficient
- Main effect tests are performed at each time bin independently
- Interaction effects apply to the entire dimension (not time-specific)
- Figure is designed to match the style of the reference figure provided
