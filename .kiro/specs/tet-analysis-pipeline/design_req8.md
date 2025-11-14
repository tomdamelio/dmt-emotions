# Design Document: Requirement 8 - Figure Generation

## Overview

This document describes the design for implementing publication-ready figure generation for the TET Analysis Pipeline.


## Requirement Summary

**Requirement 8**: Generate publication-ready figures for all analysis components including time series plots, coefficient plots, boxplots, PCA visualizations, and clustering visualizations.

**Key Acceptance Criteria**:
- 8.1: Annotated time series plots with dose comparisons and statistical annotations
- 8.2: LME coefficient plots with confidence intervals
- 8.3: Boxplots for peak/AUC comparisons with statistical annotations
- 8.4: PCA scree plots showing variance explained
- 8.5: PCA loading heatmaps/bar charts
- 8.6: KMeans centroid profile plots (replicating Fig. 3.5)
- 8.7: KMeans cluster probability time courses (replicating Fig. 3.6)
- 8.8: GLHMM figures (optional - only if GLHMM is implemented)
- 8.9: Save all figures as PNG with 300+ DPI

## Architecture

### Component Overview

```
Figure Generation System
├── Time Series Visualizer (8.1)
│   ├── Dose comparison plots
│   ├── Statistical annotation layer
│   └── RS baseline reference
├── Statistical Plots Generator (8.2, 8.3)
│   ├── Coefficient plots
│   └── Boxplots with tests
├── PCA Visualizer (8.4, 8.5)
│   ├── Scree plots
│   └── Loading heatmaps
└── Clustering Visualizer (8.6, 8.7, 8.8)
    ├── KMeans centroid profiles
    ├── Cluster probability time courses
    └── GLHMM figures (optional)
```

### Design Principles

1. **Consistency**: All figures use consistent styling, colors, and formatting
2. **Publication Quality**: 300 DPI minimum, appropriate dimensions
3. **Interpretability**: Clear labels, legends, and annotations
4. **Modularity**: Each figure type has dedicated generation function
5. **Reproducibility**: Fixed random seeds, documented parameters


## Detailed Design

### 8.1: Annotated Time Series Plots

**Purpose**: Visualize temporal dynamics of each dimension with dose comparisons and statistical annotations.

**Implementation**:
- **Module**: `scripts/plot_time_series.py`
- **Function**: `plot_annotated_time_series()`
- **Input**: 
  - Time course data (mean ± SEM per time bin)
  - LME results for statistical annotations
  - RS baseline values
- **Output**: PNG files per dimension (e.g., `timeseries_pleasantness.png`)

**Visual Elements**:
1. **Main Plot**:
   - X-axis: Time in minutes (0-20 for DMT, 0-10 for RS)
   - Y-axis: Z-scored intensity
   - Blue line: Low dose (20mg) with shaded SEM
   - Red line: High dose (40mg) with shaded SEM
   - Grey dashed vertical line: DMT onset (RS baseline ends)

2. **Statistical Annotations**:
   - Grey background shading: Time bins where DMT differs from RS baseline (p < 0.05)
   - Black horizontal bars: Time bins with significant State:Dose interaction (p < 0.05)

3. **Ordering**: Dimensions ordered by strength of State main effect (strongest first)

**Color Scheme**:
- Blue (#4472C4): Low dose (20mg)
- Red (#C55A11): High dose (40mg)
- Grey (#808080): RS baseline and statistical shading

**Figure Specifications**:
- Size: 12 × 16 inches (multi-panel figure)
- DPI: 300
- Format: PNG
- Font: Arial or Helvetica, 10-12pt


### 8.2: LME Coefficient Plots

**Purpose**: Display fixed effect estimates with confidence intervals for all dimensions.

**Implementation**:
- **Module**: `scripts/plot_lme_coefficients.py`
- **Function**: `plot_coefficient_forest()`
- **Input**: LME results CSV with beta, CI, p_fdr
- **Output**: `lme_coefficients_forest.png`

**Visual Elements**:
1. **Forest Plot Layout**:
   - Y-axis: Dimensions (ordered by State effect strength)
   - X-axis: Beta coefficient value
   - Points: Beta estimates
   - Error bars: 95% confidence intervals
   - Vertical line at x=0: Null effect reference

2. **Significance Markers**:
   - Filled circles: p_fdr < 0.05
   - Open circles: p_fdr ≥ 0.05
   - Color coding by effect type (State, Dose, Interaction)

3. **Panels**: Separate panels for each fixed effect type

**Figure Specifications**:
- Size: 10 × 8 inches
- DPI: 300
- Format: PNG

### 8.3: Peak and AUC Boxplots

**Purpose**: Compare High vs Low dose for peak values, time_to_peak, and AUC.

**Implementation**:
- **Module**: `scripts/plot_peak_auc.py`
- **Function**: `plot_dose_comparison_boxplots()`
- **Input**: Peak/AUC analysis results with Wilcoxon test results
- **Output**: `peak_auc_dose_comparison.png`

**Visual Elements**:
1. **Boxplot Layout**:
   - X-axis: Dimensions
   - Y-axis: Metric value (peak, time_to_peak, or AUC)
   - Blue boxes: Low dose
   - Red boxes: High dose
   - Paired lines connecting individual subjects

2. **Statistical Annotations**:
   - Significance stars: * p < 0.05, ** p < 0.01, *** p < 0.001
   - Effect size (r) with 95% CI displayed
   - Only show annotations for p_fdr < 0.05

3. **Panels**: Three panels (peak, time_to_peak, AUC_0_9)

**Figure Specifications**:
- Size: 14 × 6 inches
- DPI: 300
- Format: PNG


### 8.4: PCA Scree Plots

**Purpose**: Show variance explained by each principal component.

**Implementation**:
- **Module**: `scripts/plot_pca_results.py` (already implemented)
- **Function**: `plot_scree()`
- **Input**: PCA variance explained data
- **Output**: `pca_scree_plot.png`

**Visual Elements**:
1. **Bar Chart**:
   - X-axis: Component number (PC1, PC2, ...)
   - Y-axis: Variance explained (%)
   - Bars: Individual variance per component
   - Line: Cumulative variance explained

2. **Threshold Line**:
   - Horizontal dashed line at 70-80% cumulative variance
   - Indicates component retention threshold

**Figure Specifications**:
- Size: 8 × 6 inches
- DPI: 300
- Format: PNG

### 8.5: PCA Loading Heatmaps

**Purpose**: Display dimension contributions to retained components.

**Implementation**:
- **Module**: `scripts/plot_pca_results.py` (already implemented)
- **Function**: `plot_loadings_heatmap()`
- **Input**: PCA loadings matrix
- **Output**: `pca_loadings_heatmap.png`

**Visual Elements**:
1. **Heatmap Layout**:
   - Rows: Dimensions (ordered by loading magnitude on PC1)
   - Columns: Retained components (PC1, PC2, ...)
   - Color scale: Diverging (blue-white-red) for negative-zero-positive loadings
   - Annotations: Loading values displayed in cells

2. **Ordering**: Dimensions sorted by absolute loading on PC1 (descending)

**Figure Specifications**:
- Size: 8 × 10 inches
- DPI: 300
- Format: PNG


### 8.6: KMeans Centroid Profile Plots

**Purpose**: Replicate Fig. 3.5 from preliminary analysis showing characteristic dimension patterns for each cluster.

**Implementation**:
- **Module**: `scripts/tet/state_visualization.py` (already implemented)
- **Class**: `TETStateVisualization`
- **Method**: `plot_kmeans_centroid_profiles()`
- **Input**: KMeans cluster assignments and centroids
- **Output**: `clustering_kmeans_centroids_k2.png`

**Visual Elements**:
1. **Horizontal Bar Charts**:
   - One panel per cluster (side-by-side for k=2)
   - Y-axis: 15 dimensions (labeled with readable names)
   - X-axis: Normalized contribution (-1 to +1)
   - Blue bars: Positive contributions (elevated dimensions)
   - Red bars: Negative contributions (suppressed dimensions)
   - Vertical line at x=0: Neutral reference

2. **Normalization**: Each centroid normalized by its maximum absolute value

3. **Dimension Labels**: 
   - Remove '_z' suffix
   - Convert underscores to spaces
   - Title case formatting

**Figure Specifications**:
- Size: 12 × 6 inches
- DPI: 300
- Format: PNG
- **Status**: ✅ Already implemented and tested

### 8.7: KMeans Cluster Probability Time Courses

**Purpose**: Replicate Fig. 3.6 showing temporal dynamics of cluster probabilities.

**Implementation**:
- **Module**: `scripts/tet/state_visualization.py` (already implemented)
- **Class**: `TETStateVisualization`
- **Method**: `plot_kmeans_cluster_timecourses()`
- **Input**: KMeans soft probabilities over time
- **Output**: `clustering_kmeans_prob_timecourses_dmt_only.png`

**Visual Elements**:
1. **Time Course Plots**:
   - Separate panels for each condition (DMT High, DMT Low, optionally RS)
   - X-axis: Time in minutes
   - Y-axis: Cluster probability (0-1)
   - One line per cluster (different colors)
   - Shaded SEM bands around mean trajectories

2. **Color Scheme**:
   - Distinct colors per cluster (e.g., Set2 colormap)
   - Consistent across panels

3. **Layout**: 2×2 grid (DMT High, DMT Low, RS High, RS Low) or 1×2 (DMT only)

**Figure Specifications**:
- Size: 14 × 8 inches
- DPI: 300
- Format: PNG
- **Status**: ✅ Already implemented and tested


### 8.8: GLHMM Figures (Optional - Future Implementation)

**Purpose**: Visualize temporal state modeling results if GLHMM is implemented.

**Implementation Status**: ⚠️ **NOT IMPLEMENTED** - Deferred to optional tasks 27+

**Planned Figures** (when GLHMM is implemented):

1. **GLHMM State Probability Time Courses**:
   - Similar to 8.7 but using GLHMM gamma probabilities
   - Shows temporal evolution of GLHMM states
   - Separate panels for State × Dose conditions
   - **Module**: `scripts/tet/state_visualization.py`
   - **Method**: `plot_glhmm_state_timecourses()`
   - **Output**: `glhmm_state_prob_timecourses.png`

2. **KMeans-GLHMM Correspondence Heatmap**:
   - Shows how KMeans clusters map to GLHMM states
   - Rows: KMeans clusters
   - Columns: GLHMM states
   - Cell values: P(GLHMM state | KMeans cluster)
   - **Module**: `scripts/tet/state_visualization.py`
   - **Method**: `plot_kmeans_glhmm_crosswalk()`
   - **Output**: `kmeans_glhmm_crosswalk.png`

**Note**: These figures will only be generated if:
- GLHMM library is installed (`pip install git+https://github.com/vidaurre/glhmm`)
- GLHMM models are successfully trained
- Tasks 27.1-27.5 are completed

**Current Workaround**: 
- Analysis reports note that GLHMM figures are not available
- KMeans-only analysis is complete and valid
- GLHMM can be added later without affecting current results


### 8.9: Figure Export Specifications

**Purpose**: Ensure all figures meet publication quality standards.

**Implementation**:
- **Common Parameters**: Applied to all figure generation functions
- **Output Directory**: `results/tet/figures/`
- **Naming Convention**: Descriptive names indicating content

**Technical Specifications**:

1. **Resolution**:
   - Minimum: 300 DPI
   - Recommended: 300-600 DPI for print
   - Format: PNG (lossless compression)

2. **Dimensions**:
   - Time series: 12 × 16 inches (multi-panel)
   - Coefficient plots: 10 × 8 inches
   - Boxplots: 14 × 6 inches
   - PCA plots: 8 × 6 inches (scree), 8 × 10 inches (heatmap)
   - Clustering plots: 12 × 6 inches (centroids), 14 × 8 inches (time courses)

3. **File Naming**:
   - `timeseries_{dimension}.png` - Individual time series
   - `timeseries_all_dimensions.png` - Combined multi-panel
   - `lme_coefficients_forest.png` - Coefficient forest plot
   - `peak_auc_dose_comparison.png` - Peak/AUC boxplots
   - `pca_scree_plot.png` - PCA scree plot
   - `pca_loadings_heatmap.png` - PCA loadings
   - `clustering_kmeans_centroids_k{k}.png` - Centroid profiles
   - `clustering_kmeans_prob_timecourses_{condition}.png` - Cluster time courses
   - `glhmm_state_prob_timecourses.png` - GLHMM time courses (optional)
   - `kmeans_glhmm_crosswalk.png` - Correspondence heatmap (optional)

4. **Matplotlib Configuration**:
   ```python
   import matplotlib.pyplot as plt
   
   # Set publication-quality defaults
   plt.rcParams['figure.dpi'] = 300
   plt.rcParams['savefig.dpi'] = 300
   plt.rcParams['font.family'] = 'sans-serif'
   plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica']
   plt.rcParams['font.size'] = 10
   plt.rcParams['axes.labelsize'] = 11
   plt.rcParams['axes.titlesize'] = 12
   plt.rcParams['xtick.labelsize'] = 9
   plt.rcParams['ytick.labelsize'] = 9
   plt.rcParams['legend.fontsize'] = 9
   plt.rcParams['figure.titlesize'] = 13
   
   # Save with tight bounding box
   plt.savefig(output_path, dpi=300, bbox_inches='tight')
   ```


## Implementation Status

### Completed (✅)

1. **PCA Visualizations** (8.4, 8.5):
   - ✅ Scree plots implemented in `scripts/plot_pca_results.py`
   - ✅ Loading heatmaps implemented
   - ✅ Tested and generating correct outputs

2. **Clustering Visualizations** (8.6, 8.7):
   - ✅ KMeans centroid profiles implemented in `scripts/tet/state_visualization.py`
   - ✅ Cluster probability time courses implemented
   - ✅ Tested with real data, replicating Fig. 3.5 and 3.6
   - ✅ Dimension labels visible and interpretable

3. **Figure Export** (8.9):
   - ✅ All figures save as PNG with 300 DPI
   - ✅ Consistent naming convention
   - ✅ Organized in `results/tet/figures/` directory

### Pending (⏳)

1. **Time Series Plots** (8.1):
   - ⏳ Needs implementation of annotated time series with statistical overlays
   - ⏳ Requires LME results for annotation
   - ⏳ Should integrate with existing time course data

2. **LME Coefficient Plots** (8.2):
   - ⏳ Needs forest plot implementation
   - ⏳ Requires LME results CSV
   - ⏳ Should show all fixed effects with CIs

3. **Peak/AUC Boxplots** (8.3):
   - ⏳ Needs boxplot implementation with paired lines
   - ⏳ Requires Wilcoxon test results
   - ⏳ Should show effect sizes with CIs

### Optional/Future (⚠️)

1. **GLHMM Figures** (8.8):
   - ⚠️ Deferred to tasks 27.3-27.5
   - ⚠️ Requires GLHMM implementation first
   - ⚠️ Not blocking current analysis completion


## Data Flow

### Input Data Sources

```
Figure Generation Pipeline
│
├── Time Series Plots (8.1)
│   ├── Input: results/tet/descriptive/timecourse_*.csv
│   ├── Input: results/tet/lme/lme_coefficients_*.csv
│   └── Output: results/tet/figures/timeseries_*.png
│
├── Coefficient Plots (8.2)
│   ├── Input: results/tet/lme/lme_coefficients_all_dimensions.csv
│   └── Output: results/tet/figures/lme_coefficients_forest.png
│
├── Peak/AUC Plots (8.3)
│   ├── Input: results/tet/peak_auc/peak_comparison_results.csv
│   ├── Input: results/tet/peak_auc/auc_comparison_results.csv
│   └── Output: results/tet/figures/peak_auc_dose_comparison.png
│
├── PCA Plots (8.4, 8.5)
│   ├── Input: results/tet/pca/pca_variance_explained.csv
│   ├── Input: results/tet/pca/pca_loadings.csv
│   ├── Output: results/tet/figures/pca_scree_plot.png
│   └── Output: results/tet/figures/pca_loadings_heatmap.png
│
└── Clustering Plots (8.6, 8.7, 8.8)
    ├── Input: results/tet/clustering/clustering_kmeans_assignments.csv
    ├── Input: results/tet/clustering/clustering_glhmm_probabilities.csv (optional)
    ├── Output: results/tet/figures/clustering_kmeans_centroids_k2.png
    ├── Output: results/tet/figures/clustering_kmeans_prob_timecourses_*.png
    ├── Output: results/tet/figures/glhmm_state_prob_timecourses.png (optional)
    └── Output: results/tet/figures/kmeans_glhmm_crosswalk.png (optional)
```

### Master Figure Generation Script

**Purpose**: Centralized script to generate all figures in correct order.

**Implementation**:
- **Script**: `scripts/generate_all_figures.py`
- **Usage**: `python scripts/generate_all_figures.py --input results/tet --output results/tet/figures`

**Workflow**:
1. Check for required input files
2. Generate PCA figures (8.4, 8.5)
3. Generate clustering figures (8.6, 8.7)
4. Generate time series figures (8.1) if LME results available
5. Generate coefficient plots (8.2) if LME results available
6. Generate peak/AUC plots (8.3) if peak analysis complete
7. Generate GLHMM figures (8.8) if GLHMM data available
8. Create summary HTML index of all figures
9. Log generation status and any missing inputs


## Testing Strategy

### Visual Inspection Checklist

For each figure type, verify:

1. **Resolution and Quality**:
   - [ ] DPI ≥ 300
   - [ ] No pixelation or artifacts
   - [ ] Text is crisp and readable
   - [ ] Lines are smooth

2. **Layout and Formatting**:
   - [ ] Axes labeled correctly
   - [ ] Legend present and positioned well
   - [ ] Title descriptive and clear
   - [ ] No overlapping text
   - [ ] Appropriate margins

3. **Data Accuracy**:
   - [ ] Values match source data
   - [ ] Statistical annotations correct
   - [ ] Color coding consistent
   - [ ] Scale appropriate for data range

4. **Consistency**:
   - [ ] Font sizes consistent across figures
   - [ ] Color scheme consistent
   - [ ] Naming convention followed
   - [ ] Style matches other figures

### Automated Tests

**Unit Tests** (`tests/test_figure_generation.py`):
- Test that each figure generation function runs without errors
- Test that output files are created
- Test that output files have correct dimensions
- Test that figures contain expected elements (axes, labels, etc.)

**Integration Tests**:
- Test master figure generation script
- Test with missing input files (should handle gracefully)
- Test with different data subsets
- Test figure regeneration (overwrite existing)

### Example Test

```python
def test_kmeans_centroid_plot_generation():
    """Test KMeans centroid profile plot generation."""
    # Load test data
    data = pd.read_csv('tests/fixtures/test_tet_data.csv')
    assignments = pd.read_csv('tests/fixtures/test_kmeans_assignments.csv')
    
    # Initialize visualizer
    viz = TETStateVisualization(data=data, kmeans_assignments=assignments)
    
    # Generate plot
    output_path = viz.plot_kmeans_centroid_profiles(
        k=2,
        output_dir='tests/output'
    )
    
    # Assertions
    assert os.path.exists(output_path)
    assert output_path.endswith('.png')
    
    # Check image properties
    from PIL import Image
    img = Image.open(output_path)
    assert img.size[0] >= 3600  # 12 inches * 300 DPI
    assert img.size[1] >= 1800  # 6 inches * 300 DPI
```


## Dependencies

### Python Libraries

```python
# Core plotting
matplotlib >= 3.5.0
seaborn >= 0.11.0

# Data manipulation
pandas >= 1.3.0
numpy >= 1.21.0

# Image handling
Pillow >= 8.0.0  # For image property checks

# Optional (for GLHMM figures)
glhmm  # pip install git+https://github.com/vidaurre/glhmm
```

### Input Data Requirements

**Required for All Figures**:
- Preprocessed TET data: `results/tet/tet_preprocessed.csv`

**Required for Specific Figures**:
- Time series (8.1): Time course CSVs, LME results
- Coefficients (8.2): LME coefficients CSV
- Peak/AUC (8.3): Peak and AUC analysis results
- PCA (8.4, 8.5): PCA variance and loadings CSVs
- Clustering (8.6, 8.7): KMeans assignments and probabilities
- GLHMM (8.8): GLHMM Viterbi paths and gamma probabilities (optional)

## Error Handling

### Missing Input Files

**Strategy**: Graceful degradation
- Log warning for missing files
- Skip figure generation for that component
- Continue with available figures
- Generate summary report of what was/wasn't created

**Example**:
```python
def generate_all_figures(input_dir, output_dir):
    """Generate all available figures."""
    generated = []
    skipped = []
    
    # Try PCA figures
    if os.path.exists(f"{input_dir}/pca/pca_variance_explained.csv"):
        try:
            generate_pca_figures(input_dir, output_dir)
            generated.append("PCA figures")
        except Exception as e:
            logger.error(f"PCA figure generation failed: {e}")
            skipped.append(("PCA figures", str(e)))
    else:
        skipped.append(("PCA figures", "Input data not found"))
    
    # ... repeat for other figure types
    
    # Generate summary
    print(f"\nGenerated: {len(generated)} figure types")
    print(f"Skipped: {len(skipped)} figure types")
    
    return generated, skipped
```

### Invalid Data

**Strategy**: Validation before plotting
- Check for NaN/Inf values
- Verify expected columns present
- Validate data ranges
- Log specific issues found

### Plotting Errors

**Strategy**: Try-except with informative messages
- Catch matplotlib errors
- Log full traceback
- Provide suggestions for fixes
- Don't crash entire pipeline


## Future Enhancements

### Interactive Figures

**Plotly Integration**:
- Generate interactive HTML versions of key figures
- Allow zooming, panning, hover tooltips
- Useful for exploratory analysis
- Complement static PNG figures

### Figure Customization

**Configuration File**:
- YAML/JSON config for figure styling
- User-customizable colors, fonts, sizes
- Preset themes (publication, presentation, poster)

**Example** (`figure_config.yaml`):
```yaml
style:
  dpi: 300
  font_family: Arial
  font_size: 10
  
colors:
  low_dose: "#4472C4"
  high_dose: "#C55A11"
  baseline: "#808080"
  
dimensions:
  time_series: [12, 16]
  coefficient_plot: [10, 8]
  boxplot: [14, 6]
```

### Automated Figure Comparison

**Version Control for Figures**:
- Compare figures across analysis runs
- Detect visual changes
- Useful for validating analysis updates
- Tools: `pytest-mpl` for matplotlib figure comparison

### Publication-Ready Composites

**Multi-Panel Figures**:
- Combine related figures into publication panels
- Add panel labels (A, B, C, ...)
- Consistent sizing and alignment
- Export as single high-res file

**Example**: Combine centroid profiles (8.6) and time courses (8.7) into single figure replicating thesis figures 3.5-3.6.


## Summary

### Implementation Priorities

**Phase 1: Core Clustering Figures** (✅ Complete)
- KMeans centroid profiles (8.6)
- Cluster probability time courses (8.7)
- PCA scree plots (8.4)
- PCA loading heatmaps (8.5)

**Phase 2: Statistical Figures** (⏳ Pending)
- Annotated time series plots (8.1)
- LME coefficient forest plots (8.2)
- Peak/AUC boxplots (8.3)

**Phase 3: Optional Enhancements** (⚠️ Future)
- GLHMM figures (8.8) - requires tasks 27+
- Interactive versions
- Automated comparison tools
- Multi-panel composites

### Key Design Decisions

1. **Modularity**: Each figure type has dedicated function/class
2. **Consistency**: Shared styling parameters across all figures
3. **Quality**: 300 DPI minimum for publication
4. **Flexibility**: Optional figures handled gracefully
5. **Reproducibility**: Fixed random seeds, documented parameters

### Success Criteria

Figure generation is successful when:
- ✅ All required figures (8.1-8.7, 8.9) generate without errors
- ✅ Figures meet 300 DPI quality standard
- ✅ Visual elements match specifications
- ✅ Files saved with correct naming convention
- ✅ Missing optional data (GLHMM) handled gracefully
- ✅ Generated figures replicate preliminary analysis (Fig. 3.5, 3.6)

### Current Status

**Completed**: 4/8 acceptance criteria (8.4, 8.5, 8.6, 8.7, 8.9)
**Pending**: 3/8 acceptance criteria (8.1, 8.2, 8.3)
**Optional**: 1/8 acceptance criteria (8.8 - GLHMM figures)

**Next Steps**:
1. Implement time series plots with statistical annotations (8.1)
2. Implement LME coefficient forest plots (8.2)
3. Implement peak/AUC boxplots (8.3)
4. Create master figure generation script
5. Add automated tests for figure generation
6. Document figure interpretation in user guide

---

**Document Version**: 1.0  
**Last Updated**: 2025-11-13  
**Status**: Draft - Ready for Implementation
