# Design Document: Requirement 6b - Independent Component Analysis (ICA)

## Overview

This design document specifies the implementation of Independent Component Analysis (ICA) for TET data dimensionality reduction. The system will perform group-level ICA on z-scored dimensions to identify statistically independent sources of experiential variation that may not be captured by variance-based PCA, particularly beyond the first two principal components. ICA will be compared directly with PCA to assess whether independence-based decomposition reveals latent experiential patterns masked by the dominant variance structure.

## Requirements Reference

**Requirement 6b: Independent Component Analysis (ICA)**

User Story: As a researcher, I want the system to perform ICA on standardized TET data in addition to PCA, so that I can identify statistically independent sources of experiential variation that may not be captured by the variance-based principal components, particularly beyond the first two components that explain most variance in PCA.

Acceptance Criteria:
- 6b.1: Perform ICA on z-scored dimensions using FastICA algorithm
- 6b.2: Extract same number of components as retained PCA components
- 6b.3: Compute mixing matrix coefficients
- 6b.4: Fit LME models for IC1 and IC2 with State, Dose, Time_c interactions
- 6b.5: Compute correlations between ICA and PCA component scores
- 6b.6: Export ICA mixing matrix as CSV
- 6b.7: Export ICA component scores as CSV
- 6b.8: Export IC1 and IC2 LME results
- 6b.9: Generate comparison visualizations (ICA vs PCA)
- 6b.10: Document whether ICA reveals patterns beyond PC1 and PC2

## Architecture

### Component Overview

```
TETICAAnalyzer
├── fit_ica()              # Fit group-level ICA using FastICA
├── transform_data()       # Transform all data to IC scores
├── get_mixing_matrix()    # Extract mixing matrix (dimension contributions)
├── compute_pca_correlation() # Correlate IC scores with PC scores
└── export_results()       # Save mixing matrix, IC scores, correlations

TETICALMEAnalyzer
├── prepare_ic_data()      # Prepare IC scores for LME
├── fit_ic_models()        # Fit LME for IC1, IC2
├── extract_results()      # Extract coefficients and p-values
└── export_results()       # Save LME results for ICs

TETICAComparator
├── compare_loadings()     # Compare ICA mixing vs PCA loadings
├── compare_lme_results()  # Compare IC vs PC LME effects
├── generate_visualizations() # Create comparison plots
└── generate_report()      # Document findings
```


### Data Flow

```
Preprocessed TET Data (tet_preprocessed.csv)
    ↓
[Extract z-scored dimensions matrix]
    ↓ (n_observations × n_dimensions)
    ↓ (16200 × 15)
    ↓
[Fit group-level ICA] ← sklearn.decomposition.FastICA
    ↓
[Extract n_components (same as PCA)]
    ↓
[Transform all data to IC scores]
    ↓
[Add metadata back (subject, session, state, dose, time)]
    ↓
[Compute correlations with PC scores]
    ↓
[Prepare for LME: add Time_c, interactions]
    ↓
[Fit LME models for IC1, IC2] ← statsmodels MixedLM
    ↓
[Compare with PCA: loadings, LME effects, interpretability]
    ↓
[Export: mixing matrix, IC scores, correlations, LME results, comparison report]
```

## Detailed Design

### 1. TETICAAnalyzer Class

**Purpose**: Perform group-level ICA on z-scored TET dimensions to identify statistically independent sources.

**Rationale**: 
- ICA identifies sources based on statistical independence (non-Gaussianity)
- Complementary to PCA which identifies sources based on variance
- May reveal latent experiential patterns masked by dominant variance structure
- Particularly useful when variance-based components don't capture all meaningful structure
- Group-level ICA allows comparison of IC scores across subjects

#### 1.1 Initialization

```python
class TETICAAnalyzer:
    def __init__(self, data: pd.DataFrame, dimensions: List[str], 
                 n_components: int, pca_scores: Optional[pd.DataFrame] = None,
                 random_state: int = 42):
        """
        Initialize ICA analyzer.
        
        Args:
            data: Preprocessed TET data with z-scored dimensions
            dimensions: List of z-scored dimension column names
            n_components: Number of components (same as PCA for comparison)
            pca_scores: Optional PC scores for correlation analysis
            random_state: Random seed for reproducibility
        """
        self.data = data.copy()
        self.dimensions = dimensions
        self.n_components = n_components
        self.pca_scores = pca_scores
        self.random_state = random_state
        self.ica_model = None  # Single ICA model for all data
        self.ic_scores = None  # DataFrame with IC scores
        self.mixing_matrix_df = None
        self.pca_correlation_df = None
```


**Design Decisions**:
- Single ICA model for all subjects (group-level analysis)
- Same n_components as PCA for direct comparison
- FastICA algorithm (efficient, widely used)
- Fixed random_state for reproducibility
- Optional PCA scores for correlation analysis

#### 1.2 Fit Group-Level ICA

```python
def fit_ica(self) -> FastICA:
    """
    Fit group-level ICA on z-scored dimensions across all subjects.
    
    Process:
    1. Extract z-scored dimension matrix from all data
       Shape: (n_total_timepoints × n_dimensions) = (16200 × 15)
    2. Fit FastICA with specified n_components
    3. Store fitted model and mixing matrix
    
    Algorithm: FastICA
    - Maximizes non-Gaussianity of sources
    - Uses fixed-point iteration
    - Converges to statistically independent components
    
    Returns:
        Fitted ICA model
    """
```

**Implementation Details**:
- Use `sklearn.decomposition.FastICA(n_components=n_components, random_state=random_state)`
- Input matrix: all observations × 15 dimensions (16200 × 15)
- Algorithm parameters:
  - `algorithm='parallel'`: Faster convergence
  - `whiten='unit-variance'`: Standardize components
  - `max_iter=1000`: Sufficient for convergence
  - `tol=1e-4`: Standard tolerance
- Mixing matrix A: dimensions × components (15 × n_components)
- Unmixing matrix W: components × dimensions (n_components × 15)

**ICA vs PCA Key Differences**:
```python
# PCA: Maximizes variance, orthogonal components
# Components ordered by variance explained
# Loadings: pca.components_ (orthonormal)

# ICA: Maximizes independence (non-Gaussianity)
# Components unordered (no variance hierarchy)
# Mixing matrix: ica.mixing_ (not orthogonal)
# IC scores: not necessarily uncorrelated
```


#### 1.3 Transform Data to IC Scores

```python
def transform_data(self) -> pd.DataFrame:
    """
    Transform z-scored dimensions to IC scores for all data.
    
    Process:
    1. Extract z-scored dimension matrix from all data
    2. Transform using fitted ICA model: IC_scores = (X - mean) @ W.T
    3. Create DataFrame with IC scores
    4. Add metadata columns (subject, session_id, state, dose, t_bin, t_sec)
    5. Return complete DataFrame
    
    Returns:
        DataFrame with columns:
        - subject, session_id, state, dose, t_bin, t_sec
        - IC1, IC2, ..., ICn (one column per component)
        
    Example:
        subject | session_id  | state | dose | t_bin | t_sec | IC1   | IC2   | IC3
        --------|-------------|-------|------|-------|-------|-------|-------|-----
        S01     | DMT_Alta_1  | DMT   | Alta | 0     | 0     | -0.52 | 0.18  | -0.31
        S01     | DMT_Alta_1  | DMT   | Alta | 1     | 4     | -0.41 | 0.25  | -0.19
        ...
    """
```

**Design Decisions**:
- IC columns named as IC1, IC2, etc. (1-indexed)
- Preserve all metadata for downstream LME analysis
- All observations transformed using same ICA model
- IC scores directly comparable across subjects and conditions
- Note: IC ordering is arbitrary (no variance hierarchy like PCA)

#### 1.4 Get Mixing Matrix

```python
def get_mixing_matrix(self) -> pd.DataFrame:
    """
    Extract ICA mixing matrix from group-level ICA.
    
    The mixing matrix A shows how each independent component
    contributes to each observed dimension:
        X = A @ S
    where X = observed dimensions, S = independent components
    
    Mixing matrix coefficients are analogous to PCA loadings
    but represent linear combinations that produce independent sources.
    
    Returns:
        DataFrame with columns:
        - component: Component name (IC1, IC2, ...)
        - dimension: Original dimension name
        - mixing_coef: Mixing coefficient (weight)
    
    Format (long format):
        component | dimension           | mixing_coef
        ----------|---------------------|------------
        IC1       | pleasantness_z      | 0.38
        IC1       | anxiety_z           | -0.45
        IC1       | complex_imagery_z   | 0.22
        IC2       | pleasantness_z      | 0.61
        IC2       | anxiety_z           | 0.15
        ...
    """
```


**Implementation Details**:
- Extract `ica.mixing_` (shape: n_dimensions × n_components = 15 × n_components)
- Convert to long format for easier analysis and visualization
- Mixing coefficients are identical for all subjects (group-level ICA)
- High absolute coefficient (|coef| > 0.3) indicates strong contribution
- Unlike PCA loadings, mixing coefficients are not orthogonal

**Interpretation**:
- Positive coefficient: dimension increases with IC score
- Negative coefficient: dimension decreases with IC score
- Magnitude: contribution strength to that independent source
- Pattern of coefficients defines the "signature" of each independent component

#### 1.5 Compute PCA Correlation

```python
def compute_pca_correlation(self) -> pd.DataFrame:
    """
    Compute correlations between ICA component scores and PCA component scores.
    
    Purpose:
    - Assess overlap vs complementarity between ICA and PCA
    - High correlation (|r| > 0.7): ICA component similar to PC
    - Low correlation (|r| < 0.3): ICA reveals distinct structure
    
    Returns:
        DataFrame with columns:
        - ic_component: ICA component name (IC1, IC2, ...)
        - pc_component: PCA component name (PC1, PC2, ...)
        - correlation: Pearson correlation coefficient
        - abs_correlation: Absolute correlation (for sorting)
    
    Example:
        ic_component | pc_component | correlation | abs_correlation
        -------------|--------------|-------------|----------------
        IC1          | PC1          | 0.85        | 0.85
        IC1          | PC2          | -0.12       | 0.12
        IC2          | PC1          | 0.23        | 0.23
        IC2          | PC2          | 0.68        | 0.68
        IC3          | PC1          | -0.15       | 0.15
        IC3          | PC2          | 0.31        | 0.31
    """
```

**Design Decisions**:
- Compute all pairwise correlations (IC × PC)
- Include absolute correlation for identifying strongest relationships
- High correlation suggests redundancy; low suggests complementarity
- Helps identify which ICs correspond to which PCs


#### 1.6 Export Results

```python
def export_results(self, output_dir: str) -> Dict[str, str]:
    """
    Export ICA results to CSV files.
    
    Creates:
    - ica_mixing_matrix.csv: Mixing coefficients for all components
    - ica_scores.csv: IC scores for all time points
    - ica_pca_correlation.csv: Correlations between IC and PC scores
    
    Args:
        output_dir: Directory to save output files
    
    Returns:
        Dict mapping file types to file paths
    """
```

### 2. TETICALMEAnalyzer Class

**Purpose**: Fit Linear Mixed Effects models for independent component scores.

**Rationale**:
- IC1 and IC2 analyzed for consistency with PCA analysis
- Same LME structure as PCA ensures direct comparability
- Allows testing whether independent sources show different dose/state effects than PCs

#### 2.1 Initialization

```python
class TETICALMEAnalyzer:
    def __init__(self, ic_scores: pd.DataFrame, components: List[str] = ['IC1', 'IC2']):
        """
        Initialize LME analyzer for IC scores.
        
        Args:
            ic_scores: DataFrame with IC scores from TETICAAnalyzer
            components: List of components to analyze (default: ['IC1', 'IC2'])
        """
        self.ic_scores = ic_scores.copy()
        self.components = components
        self.models = {}  # Dict[component, MixedLMResults]
        self.results_df = None
```

#### 2.2 Prepare IC Data for LME

```python
def prepare_ic_data(self) -> pd.DataFrame:
    """
    Prepare IC scores for LME analysis.
    
    Process:
    1. Center time within each state-session
       Time_c = t_sec - mean(t_sec) for each (state, session_id)
    2. Ensure categorical variables are properly encoded
       - state: categorical with RS as reference
       - dose: categorical with Baja as reference
    3. Create interaction terms (handled by formula)
    
    Returns:
        DataFrame ready for LME fitting
    """
```


**Design Decisions**:
- Identical preparation as PCA LME analysis (Requirement 6)
- Consistent reference levels for comparability
- No additional transformations on IC scores

#### 2.3 Fit LME Models for ICs

```python
def fit_ic_models(self) -> Dict[str, MixedLMResults]:
    """
    Fit LME models for each independent component.
    
    Model specification (same as PCA and original dimensions):
        IC ~ state * dose * time_c + (1 | subject)
    
    Fixed effects:
    - state: DMT vs RS
    - dose: Alta vs Baja
    - time_c: Centered time
    - state:dose: State-dose interaction
    - state:time_c: State-time interaction
    - dose:time_c: Dose-time interaction
    - state:dose:time_c: Three-way interaction
    
    Random effects:
    - (1 | subject): Random intercept per subject
    
    Returns:
        Dict mapping component name to fitted model
    """
```

**Implementation Details**:
- Use `statsmodels.formula.api.mixedlm()`
- Formula: `"IC ~ C(state, Treatment('RS')) * C(dose, Treatment('Baja')) * time_c"`
- Groups: `groups=data['subject']`
- Method: REML (default)
- Identical to PCA LME specification

#### 2.4 Extract Results

```python
def extract_results(self) -> pd.DataFrame:
    """
    Extract coefficients, standard errors, and p-values from fitted models.
    
    Returns:
        DataFrame with columns:
        - component: Component name (IC1, IC2)
        - effect: Fixed effect name
        - beta: Coefficient estimate
        - se: Standard error
        - z_value: Z-statistic
        - p_value: P-value
        - ci_lower: 95% CI lower bound
        - ci_upper: 95% CI upper bound
    """
```


**Design Decisions**:
- Same format as PCA LME results for direct comparison
- Include confidence intervals for effect size interpretation
- No FDR correction (only 2 components tested)

#### 2.5 Export Results

```python
def export_results(self, output_dir: str) -> Dict[str, str]:
    """
    Export LME results for IC scores.
    
    Creates:
    - ica_lme_results.csv: Coefficients and p-values for IC1, IC2
    
    Args:
        output_dir: Directory to save output files
    
    Returns:
        Dict mapping file types to file paths
    """
```

### 3. TETICAComparator Class

**Purpose**: Compare ICA and PCA results to assess complementarity.

**Rationale**:
- Determine if ICA reveals patterns beyond PC1 and PC2
- Identify which ICs correspond to which PCs
- Assess whether independence-based decomposition adds value

#### 3.1 Initialization

```python
class TETICAComparator:
    def __init__(self, ica_results: Dict, pca_results: Dict):
        """
        Initialize ICA-PCA comparator.
        
        Args:
            ica_results: Dict with ICA outputs (mixing, scores, LME)
            pca_results: Dict with PCA outputs (loadings, scores, LME)
        """
        self.ica_results = ica_results
        self.pca_results = pca_results
        self.comparison_report = {}
```

#### 3.2 Compare Loadings/Mixing Patterns

```python
def compare_loadings(self) -> pd.DataFrame:
    """
    Compare ICA mixing coefficients with PCA loadings.
    
    Process:
    1. Align ICA and PCA components by correlation
    2. For each aligned pair, compute pattern similarity
    3. Identify dimensions with divergent contributions
    
    Returns:
        DataFrame with comparison metrics:
        - ic_component, pc_component: Aligned components
        - pattern_correlation: Correlation between mixing/loading patterns
        - divergent_dimensions: Dimensions with different contributions
    """
```


#### 3.3 Compare LME Results

```python
def compare_lme_results(self) -> pd.DataFrame:
    """
    Compare LME effects between ICA and PCA components.
    
    Process:
    1. Align IC and PC components by score correlation
    2. Compare effect sizes (beta coefficients) for each fixed effect
    3. Identify effects that differ between ICA and PCA
    
    Returns:
        DataFrame with:
        - effect: Fixed effect name
        - ic_component, pc_component: Aligned components
        - ic_beta, pc_beta: Effect sizes
        - ic_pvalue, pc_pvalue: P-values
        - beta_difference: Absolute difference in effect sizes
        - agreement: Whether both significant, both non-sig, or disagree
    """
```

#### 3.4 Generate Visualizations

```python
def generate_visualizations(self, output_dir: str) -> List[str]:
    """
    Generate comparison visualizations.
    
    Creates:
    1. Side-by-side heatmaps: ICA mixing vs PCA loadings
    2. Scatter plots: IC scores vs PC scores (for aligned pairs)
    3. Effect size comparison: IC vs PC LME coefficients
    4. Component interpretation: Dimension contributions for each IC/PC
    
    Returns:
        List of generated figure paths
    """
```

**Visualization Details**:

1. **Mixing/Loading Heatmap**
   - Two heatmaps side-by-side
   - Rows: dimensions, Columns: components
   - Color scale: -1 (blue) to +1 (red)
   - Highlights pattern differences

2. **Score Scatter Plots**
   - One plot per aligned IC-PC pair
   - X-axis: PC score, Y-axis: IC score
   - Color by state (RS vs DMT)
   - Shows correlation and deviations

3. **Effect Size Comparison**
   - Bar plot comparing beta coefficients
   - Grouped by effect type (state, dose, interactions)
   - Error bars for confidence intervals
   - Highlights divergent effects


#### 3.5 Generate Comparison Report

```python
def generate_report(self, output_path: str) -> str:
    """
    Generate comprehensive ICA-PCA comparison report.
    
    Report sections:
    1. Executive Summary
       - Do ICA components reveal patterns beyond PC1/PC2?
       - Which ICs align with which PCs?
       - Key differences in interpretation
    
    2. Component Alignment
       - IC-PC correlation matrix
       - Aligned pairs and their similarity
       - Unaligned components (novel ICA patterns)
    
    3. Pattern Comparison
       - Dimension contribution differences
       - Experiential interpretation differences
       - Which dimensions drive ICA vs PCA
    
    4. LME Effect Comparison
       - Dose/state effects: IC vs PC
       - Convergent findings (both methods agree)
       - Divergent findings (methods disagree)
    
    5. Interpretation Guidelines
       - When to use ICA vs PCA
       - What ICA adds beyond PCA
       - Limitations and caveats
    
    6. Recommendations
       - Should ICA be included in main analysis?
       - Which components warrant further investigation?
       - Suggested follow-up analyses
    
    Returns:
        Path to generated report (markdown format)
    """
```

### 4. Main Analysis Script

**File**: `scripts/compute_ica_analysis.py`

**Purpose**: Orchestrate ICA analysis and comparison with PCA.

#### 4.1 Workflow

```python
def main():
    """
    Main ICA analysis workflow.
    
    Steps:
    1. Load preprocessed TET data
    2. Load PCA results (scores, loadings, n_components)
    3. Initialize TETICAAnalyzer with same n_components as PCA
    4. Fit ICA on z-scored dimensions
    5. Transform data to IC scores
    6. Compute correlations with PC scores
    7. Export ICA results (mixing matrix, scores, correlations)
    8. Initialize TETICALMEAnalyzer with IC scores
    9. Prepare IC data for LME
    10. Fit LME models for IC1 and IC2
    11. Export ICA LME results
    12. Initialize TETICAComparator
    13. Compare loadings/mixing patterns
    14. Compare LME results
    15. Generate comparison visualizations
    16. Generate comprehensive comparison report
    17. Print summary to console
    """
```


#### 4.2 Command-Line Interface

```python
parser.add_argument('--input', default='results/tet/preprocessed/tet_preprocessed.csv')
parser.add_argument('--pca-dir', default='results/tet/pca')
parser.add_argument('--output', default='results/tet/ica')
parser.add_argument('--components', nargs='+', default=['IC1', 'IC2'])
parser.add_argument('--random-state', type=int, default=42)
parser.add_argument('--verbose', action='store_true')
```

### 5. Integration with Pipeline

**Modification to `scripts/run_tet_analysis.py`**:

Add ICA stage after PCA:

```python
def _define_stages(self) -> List[Tuple[str, callable]]:
    return [
        ('preprocessing', self._run_preprocessing),
        ('descriptive', self._run_descriptive_stats),
        ('lme', self._run_lme_models),
        ('pca', self._run_pca_analysis),
        ('ica', self._run_ica_analysis),  # NEW STAGE
        ('clustering', self._run_clustering_analysis),
        ('figures', self._run_figure_generation),
        ('report', self._run_report_generation)
    ]

def _run_ica_analysis(self):
    """Execute ICA analysis stage."""
    self.logger.info("Computing ICA analysis...")
    from compute_ica_analysis import main as ica_main
    
    original_argv = sys.argv.copy()
    sys.argv = ['compute_ica_analysis.py']
    
    try:
        ica_main()
    finally:
        sys.argv = original_argv
```

**Validator Update**:

```python
def _validate_ica_inputs(self) -> Tuple[bool, str]:
    """Validate ICA inputs (requires PCA results)."""
    required = [
        'results/tet/preprocessed/tet_preprocessed.csv',
        'results/tet/pca/pca_scores.csv',
        'results/tet/pca/pca_loadings.csv'
    ]
    
    missing = [f for f in required if not Path(f).exists()]
    
    if missing:
        return False, (
            f"Missing required files for ICA:\n" +
            "\n".join(f"  - {f}" for f in missing) +
            "\n\nRun PCA analysis first:\n"
            "  python scripts/run_tet_analysis.py --stages pca"
        )
    
    return True, ""
```


## Data Structures

### ICA Mixing Matrix CSV

```csv
component,dimension,mixing_coef
IC1,pleasantness_z,0.38
IC1,unpleasantness_z,-0.45
IC1,emotional_intensity_z,0.52
IC1,elementary_imagery_z,0.22
IC1,complex_imagery_z,0.28
IC2,pleasantness_z,0.61
IC2,unpleasantness_z,0.15
IC2,emotional_intensity_z,-0.18
...
```

### ICA Scores CSV

```csv
subject,session_id,state,dose,t_bin,t_sec,IC1,IC2,IC3
S01,DMT_Alta_1,DMT,Alta,0,0,-0.52,0.18,-0.31
S01,DMT_Alta_1,DMT,Alta,1,4,-0.41,0.25,-0.19
...
```

### ICA-PCA Correlation CSV

```csv
ic_component,pc_component,correlation,abs_correlation
IC1,PC1,0.85,0.85
IC1,PC2,-0.12,0.12
IC1,PC3,0.23,0.23
IC2,PC1,0.23,0.23
IC2,PC2,0.68,0.68
IC2,PC3,-0.31,0.31
IC3,PC1,-0.15,0.15
IC3,PC2,0.31,0.31
IC3,PC3,0.72,0.72
```

### ICA LME Results CSV

```csv
component,effect,beta,se,z_value,p_value,ci_lower,ci_upper
IC1,Intercept,0.08,0.09,0.89,0.374,-0.10,0.26
IC1,state[T.DMT],0.52,0.13,4.00,0.0001,0.26,0.78
IC1,dose[T.Alta],0.19,0.11,1.73,0.084,-0.03,0.41
IC1,state[T.DMT]:dose[T.Alta],-0.22,0.16,-1.38,0.168,-0.53,0.09
IC2,Intercept,-0.03,0.08,-0.38,0.704,-0.19,0.13
IC2,state[T.DMT],0.28,0.12,2.33,0.020,0.04,0.52
...
```


## Error Handling

### 1. ICA Convergence Failure
- **Issue**: FastICA fails to converge within max_iter
- **Handling**: Increase max_iter to 2000, try different random_state, log warning

### 2. Singular Matrix
- **Issue**: ICA fails due to insufficient variance
- **Handling**: Check data preprocessing, ensure z-scoring is correct, skip if unrecoverable

### 3. Component Ordering Ambiguity
- **Issue**: ICA component order is arbitrary (no variance hierarchy)
- **Handling**: Order by correlation with PC1, PC2 for consistency, document ordering criterion

### 4. Sign Ambiguity
- **Issue**: ICA components have arbitrary sign (IC and -IC are equivalent)
- **Handling**: Align signs with PCA components (maximize positive correlation), document alignment

### 5. LME Convergence Failure
- **Issue**: LME model fails to converge for IC scores
- **Handling**: Try different optimizer, increase max iterations, log warning

## Testing Strategy

### Unit Tests

1. **test_ica_analyzer.py**
   - Test ICA fitting with known independent sources
   - Verify mixing matrix extraction
   - Test score transformation
   - Test PCA correlation computation

2. **test_ica_lme_analyzer.py**
   - Test data preparation (time centering)
   - Test LME fitting with synthetic IC scores
   - Verify result extraction format

3. **test_ica_comparator.py**
   - Test loading comparison logic
   - Test LME result comparison
   - Verify alignment algorithm

### Integration Tests

4. **test_ica_pipeline.py**
   - Test complete pipeline from preprocessed data to comparison report
   - Verify output file creation
   - Check CSV schema compliance
   - Verify PCA dependency handling


## Performance Considerations

- **ICA Fitting**: O(n_observations × n_dimensions² × n_iterations)
  - Expected: 16,200 observations × 15² dimensions × ~200 iterations ≈ 730M operations
  - Moderate with sklearn (5-10 seconds)
  - Single fit for all data

- **LME Fitting**: O(n_components × n_iterations)
  - Expected: 2 components × ~100 iterations ≈ 200 iterations
  - Moderate (5-10 seconds per component)

- **Correlation Computation**: O(n_components² × n_observations)
  - Expected: ~9 correlations × 16,200 observations ≈ 146K operations
  - Fast (< 1 second)

- **Memory**: Store single ICA model + scores
  - Expected: ~10KB for model + 16,200 × n_components for scores
  - Negligible (< 1MB total)

## Dependencies

- `scikit-learn>=1.0`: FastICA implementation
- `statsmodels>=0.13`: LME models
- `pandas>=1.3`: Data manipulation
- `numpy>=1.21`: Numerical operations
- `matplotlib>=3.5`: Visualization
- `seaborn>=0.11`: Enhanced visualization

## Output Files

```
results/tet/ica/
├── ica_mixing_matrix.csv         # Mixing coefficients
├── ica_scores.csv                # IC scores for all time points
├── ica_pca_correlation.csv       # IC-PC correlations
├── ica_lme_results.csv           # LME results for IC1, IC2
├── comparison_loadings.csv       # ICA vs PCA pattern comparison
├── comparison_lme.csv            # ICA vs PCA effect comparison
├── figures/
│   ├── ica_pca_mixing_heatmap.png
│   ├── ic_pc_scatter_plots.png
│   └── ica_pca_effect_comparison.png
└── ica_pca_comparison_report.md  # Comprehensive comparison
```


## Validation

### 1. Independence Check
- Verify IC scores have low mutual information
- Check: `mutual_info(IC_i, IC_j) ≈ 0` for i ≠ j
- ICA maximizes independence (not just uncorrelatedness like PCA)

### 2. Reconstruction Check
- Verify inverse transform approximates original data
- Check: `||X - A @ S|| / ||X||` (reconstruction error)
- Should be comparable to PCA reconstruction error

### 3. Mixing Matrix Validity
- Check mixing matrix is full rank
- Verify no degenerate components (all zeros)
- Ensure mixing coefficients are interpretable

### 4. PCA Correlation Validity
- Verify at least one IC highly correlates with PC1 (|r| > 0.7)
- Check for novel ICs with low correlation to all PCs (|r| < 0.3)
- Validate correlation matrix is symmetric

### 5. LME Diagnostics
- Check residual normality (Q-Q plots)
- Check homoscedasticity (residual vs fitted plots)
- Verify random effects variance > 0
- Compare with PCA LME diagnostics

## Interpretation Guidelines

### ICA vs PCA: Key Differences

| Aspect | PCA | ICA |
|--------|-----|-----|
| Objective | Maximize variance | Maximize independence |
| Orthogonality | Components orthogonal | Components not orthogonal |
| Ordering | By variance explained | Arbitrary (no hierarchy) |
| Loadings | Orthonormal | Non-orthogonal mixing |
| Interpretation | Variance modes | Independent sources |
| Best for | Correlated Gaussian data | Non-Gaussian independent sources |

### When ICA Adds Value

1. **Non-Gaussian Structure**: When experiential dimensions have non-Gaussian distributions
2. **Hidden Sources**: When multiple independent processes generate observed patterns
3. **Beyond Variance**: When important patterns are masked by dominant variance structure
4. **Artifact Separation**: When independent noise sources need isolation

### Interpreting IC Components

1. **High IC-PC Correlation (|r| > 0.7)**
   - IC captures similar structure as PC
   - Validates PCA findings
   - ICA may refine interpretation

2. **Moderate IC-PC Correlation (0.3 < |r| < 0.7)**
   - IC partially overlaps with PC
   - May reveal nuanced differences
   - Compare mixing patterns carefully

3. **Low IC-PC Correlation (|r| < 0.3)**
   - IC reveals novel structure
   - Independent source not captured by variance
   - Warrants detailed investigation


### Interpreting Mixing Coefficients

- **Positive coefficient**: Dimension increases with IC activation
- **Negative coefficient**: Dimension decreases with IC activation
- **Magnitude**: Strength of contribution (|coef| > 0.3 = substantial)
- **Pattern**: Combination defines independent source "signature"

### Comparing ICA and PCA Effects

1. **Convergent Effects** (both ICA and PCA show same effect)
   - Strong evidence for dose/state effect
   - Robust across decomposition methods
   - High confidence in interpretation

2. **Divergent Effects** (ICA shows effect, PCA doesn't or vice versa)
   - Method-specific sensitivity
   - May indicate variance vs independence trade-off
   - Requires careful interpretation

3. **Novel IC Effects** (IC with low PC correlation shows unique effect)
   - Independent source responds to manipulation
   - Not captured by variance-based analysis
   - Potentially important finding

## Research Questions Addressed

### Primary Question
**Do ICA components beyond IC1 and IC2 reveal meaningful experiential patterns not captured by PC1 and PC2?**

**Evaluation Criteria**:
1. IC-PC correlation: Are IC3+ weakly correlated with PC1-PC2?
2. Mixing patterns: Do IC3+ show distinct dimension combinations?
3. LME effects: Do IC3+ show significant dose/state effects?
4. Interpretability: Can IC3+ be meaningfully interpreted?

### Secondary Questions

1. **Do IC1 and IC2 align with PC1 and PC2?**
   - Measure: IC-PC correlation
   - Expected: High correlation (|r| > 0.7) if similar structure

2. **Does ICA reveal independent sources masked by variance?**
   - Measure: Novel ICs with low PC correlation but significant LME effects
   - Expected: At least one IC with |r| < 0.3 to all PCs

3. **Do ICA and PCA show different dose/state sensitivities?**
   - Measure: Divergent LME effects between aligned IC-PC pairs
   - Expected: Some differences in effect sizes or significance

4. **Is ICA more interpretable than PCA for TET data?**
   - Measure: Qualitative assessment of mixing patterns
   - Expected: ICA may provide clearer experiential "signatures"


## Decision Criteria for Including ICA in Main Analysis

### Include ICA if:

1. **Novel Structure**: At least one IC shows low correlation (|r| < 0.3) with all PCs
2. **Significant Effects**: Novel IC shows significant dose/state effects in LME
3. **Interpretability**: Novel IC has clear experiential interpretation
4. **Stability**: ICA results are stable across different random_state values

### Exclude ICA if:

1. **Redundancy**: All ICs highly correlate (|r| > 0.7) with PCs
2. **No New Effects**: ICA LME results identical to PCA
3. **Uninterpretable**: IC mixing patterns lack clear meaning
4. **Instability**: Results vary substantially with random_state

### Partial Inclusion:

- Report ICA as supplementary analysis
- Highlight specific novel ICs in discussion
- Use ICA to validate PCA findings
- Include ICA in methods but not main results

## Future Enhancements

1. **Stability Analysis**: Bootstrap ICA to assess component stability
2. **Optimal n_components**: Use information criteria to select n_components
3. **Temporal ICA**: Apply ICA to time-varying covariance structure
4. **Subject-Specific ICA**: Compare group-level vs within-subject ICA
5. **ICA Variants**: Test Infomax, JADE, or other ICA algorithms
6. **Sparse ICA**: Apply sparsity constraints for more interpretable mixing
7. **Hierarchical ICA**: Multi-level ICA for nested structure
8. **ICA-PCA Hybrid**: Combine variance and independence criteria

## References

- Hyvärinen, A., & Oja, E. (2000). Independent component analysis: algorithms and applications. *Neural Networks*, 13(4-5), 411-430.
- Comon, P. (1994). Independent component analysis, a new concept? *Signal Processing*, 36(3), 287-314.
- Beckmann, C. F., & Smith, S. M. (2004). Probabilistic independent component analysis for functional magnetic resonance imaging. *IEEE Transactions on Medical Imaging*, 23(2), 137-152.

## Summary

This design implements ICA as a complementary dimensionality reduction approach to PCA, specifically targeting the research question of whether independent components beyond IC1/IC2 reveal experiential patterns not captured by the variance-dominant PC1/PC2. The implementation includes:

1. **Group-level ICA** using FastICA on z-scored dimensions
2. **Direct comparison** with PCA through correlation analysis
3. **Parallel LME analysis** for IC1 and IC2
4. **Comprehensive comparison** of mixing patterns and effects
5. **Decision framework** for determining ICA's added value

The design ensures reproducibility through fixed random_state, maintains consistency with PCA analysis through identical LME specifications, and provides clear interpretation guidelines for assessing whether ICA reveals meaningful structure beyond PCA's variance-based decomposition.
