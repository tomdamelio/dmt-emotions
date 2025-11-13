# Design Document: Requirement 6 - Dimensionality Reduction via PCA

## Overview

This design document specifies the implementation of Principal Component Analysis (PCA) for TET data dimensionality reduction. The system will perform a single group-level PCA on z-scored dimensions across all subjects and time points to identify principal modes of experiential variation, then analyze these components using Linear Mixed Effects models.

## Requirements Reference

**Requirement 6: Dimensionality Reduction via PCA**

User Story: As a researcher, I want the system to perform PCA on standardized TET data, so that I can identify the principal modes of experiential variation.

Acceptance Criteria:
- 6.1: Perform PCA on z-scored dimensions across all time bins within each subject
- 6.2: Retain components explaining 70-80% of cumulative variance
- 6.3: Fit LME models for PC1 and PC2 with State, Dose, Time_c interactions
- 6.4: Export PCA loadings as CSV
- 6.5: Export variance explained as CSV

## Architecture

### Component Overview

```
TETPCAAnalyzer
├── fit_pca()              # Fit single group-level PCA on all data
├── transform_data()       # Transform all data to PC scores
├── get_loadings()         # Extract component loadings
├── get_variance_explained() # Get variance explained ratios
└── export_results()       # Save loadings, variance, and PC scores

TETPCALMEAnalyzer
├── prepare_pc_data()      # Prepare PC scores for LME
├── fit_pc_models()        # Fit LME for PC1, PC2
├── extract_results()      # Extract coefficients and p-values
└── export_results()       # Save LME results for PCs
```

### Data Flow

```
Preprocessed TET Data (tet_preprocessed.csv)
    ↓
[Extract z-scored dimensions matrix]
    ↓ (n_observations × n_dimensions)
    ↓ (16200 × 15)
    ↓
[Fit single group-level PCA] ← sklearn.decomposition.PCA
    ↓
[Determine n_components (70-80% variance)]
    ↓
[Transform all data to PC scores]
    ↓
[Add metadata back (subject, session, state, dose, time)]
    ↓
[Prepare for LME: add Time_c, interactions]
    ↓
[Fit LME models for PC1, PC2] ← statsmodels MixedLM
    ↓
[Export: loadings, variance, PC scores, LME results]
```

## Detailed Design

### 1. TETPCAAnalyzer Class

**Purpose**: Perform group-level PCA on z-scored TET dimensions across all subjects.

**Rationale**: 
- Group-level PCA identifies common modes of experiential variation
- Single PCA space allows direct comparison of PC scores across subjects
- Z-scored dimensions ensure equal weighting across dimensions
- Retaining 70-80% variance balances dimensionality reduction with information preservation

#### 1.1 Initialization

```python
class TETPCAAnalyzer:
    def __init__(self, data: pd.DataFrame, dimensions: List[str], 
                 variance_threshold: float = 0.75):
        """
        Initialize PCA analyzer.
        
        Args:
            data: Preprocessed TET data with z-scored dimensions
            dimensions: List of z-scored dimension column names (e.g., ['pleasantness_z', ...])
            variance_threshold: Target cumulative variance (default: 0.75 = 75%)
        """
        self.data = data.copy()
        self.dimensions = dimensions
        self.variance_threshold = variance_threshold
        self.pca_model = None  # Single PCA model for all data
        self.n_components = None  # Number of components retained
        self.pc_scores = None  # DataFrame with PC scores
        self.loadings_df = None
        self.variance_df = None
```

**Design Decisions**:
- Single PCA model for all subjects (group-level analysis)
- Default variance threshold of 75% (middle of 70-80% range)
- Separate storage for scores, loadings, and variance
- PC scores directly comparable across subjects

#### 1.2 Fit Group-Level PCA

```python
def fit_pca(self) -> PCA:
    """
    Fit single group-level PCA on z-scored dimensions across all subjects.
    
    Process:
    1. Extract z-scored dimension matrix from all data
       Shape: (n_total_timepoints × n_dimensions) = (16200 × 15)
    2. Fit PCA with n_components=None (fit all components initially)
    3. Calculate cumulative variance explained
    4. Determine n_components to retain based on variance_threshold
    5. Refit PCA with selected n_components for efficiency
    6. Store fitted model
    
    Returns:
        Fitted PCA model
    """
```

**Implementation Details**:
- Use `sklearn.decomposition.PCA(n_components=None)` for initial fit
- Input matrix: all observations × 15 dimensions (16200 × 15)
- Calculate cumulative variance: `np.cumsum(pca.explained_variance_ratio_)`
- Find minimum n_components where cumulative variance ≥ threshold
- Refit with selected n_components for efficiency

**Variance Threshold Logic**:
```python
# Initial fit to determine variance structure
pca_full = PCA(n_components=None)
pca_full.fit(X)

# Find n_components for target variance
cumsum_var = np.cumsum(pca_full.explained_variance_ratio_)
n_components = np.argmax(cumsum_var >= self.variance_threshold) + 1
n_components = max(2, n_components)  # Ensure at least PC1 and PC2

# Refit with selected n_components
self.pca_model = PCA(n_components=n_components)
self.pca_model.fit(X)
self.n_components = n_components
```

#### 1.3 Transform Data to PC Scores

```python
def transform_data(self) -> pd.DataFrame:
    """
    Transform z-scored dimensions to PC scores for all data.
    
    Process:
    1. Extract z-scored dimension matrix from all data
    2. Transform using fitted PCA model: PC_scores = X @ components.T
    3. Create DataFrame with PC scores
    4. Add metadata columns (subject, session_id, state, dose, t_bin, t_sec)
    5. Return complete DataFrame
    
    Returns:
        DataFrame with columns:
        - subject, session_id, state, dose, t_bin, t_sec
        - PC1, PC2, ..., PCn (one column per retained component)
        
    Example:
        subject | session_id  | state | dose | t_bin | t_sec | PC1   | PC2   | PC3
        --------|-------------|-------|------|-------|-------|-------|-------|-----
        S01     | DMT_Alta_1  | DMT   | Alta | 0     | 0     | -0.45 | 0.23  | 0.12
        S01     | DMT_Alta_1  | DMT   | Alta | 1     | 4     | -0.38 | 0.31  | 0.08
        ...
    """
```

**Design Decisions**:
- PC columns named as PC1, PC2, etc. (1-indexed for interpretability)
- Preserve all metadata for downstream LME analysis
- All observations transformed using same PCA model (group-level space)
- PC scores directly comparable across subjects and conditions

#### 1.4 Get Loadings

```python
def get_loadings(self) -> pd.DataFrame:
    """
    Extract PCA loadings (component weights) from group-level PCA.
    
    Loadings represent the contribution of each original dimension
    to each principal component. Since this is group-level PCA,
    loadings are the same for all subjects.
    
    Returns:
        DataFrame with columns:
        - component: Component name (PC1, PC2, ...)
        - dimension: Original dimension name
        - loading: Loading value (weight)
    
    Format (long format):
        component | dimension           | loading
        ----------|---------------------|--------
        PC1       | pleasantness_z      | 0.45
        PC1       | anxiety_z           | -0.32
        PC1       | complex_imagery_z   | 0.28
        PC2       | pleasantness_z      | 0.12
        PC2       | anxiety_z           | 0.67
        ...
    """
```

**Implementation Details**:
- Extract `pca.components_` (shape: n_components × n_dimensions)
- Convert to long format for easier analysis and visualization
- Loadings are identical for all subjects (group-level PCA)
- High absolute loading (|loading| > 0.3) indicates strong contribution

#### 1.5 Get Variance Explained

```python
def get_variance_explained(self) -> pd.DataFrame:
    """
    Extract variance explained by each component from group-level PCA.
    
    Returns:
        DataFrame with columns:
        - component: Component name (PC1, PC2, ...)
        - variance_explained: Proportion of variance (0-1)
        - cumulative_variance: Cumulative proportion (0-1)
    
    Example:
        component | variance_explained | cumulative_variance
        ----------|--------------------|--------------------
        PC1       | 0.35               | 0.35
        PC2       | 0.22               | 0.57
        PC3       | 0.18               | 0.75
    """
```

**Design Decisions**:
- Include both individual and cumulative variance
- Useful for verifying variance threshold was met
- Single set of variance values (group-level PCA)
- Cumulative variance should meet or exceed threshold (70-80%)

#### 1.6 Export Results

```python
def export_results(self, output_dir: str) -> Dict[str, str]:
    """
    Export PCA results to CSV files.
    
    Creates:
    - pca_loadings.csv: Component loadings for all subjects
    - pca_variance_explained.csv: Variance explained by each component
    - pca_scores.csv: PC scores for all time points
    
    Args:
        output_dir: Directory to save output files
    
    Returns:
        Dict mapping file types to file paths
    """
```

### 2. TETPCALMEAnalyzer Class

**Purpose**: Fit Linear Mixed Effects models for principal component scores.

**Rationale**:
- PC1 and PC2 capture most variance and are most interpretable
- Same LME structure as original dimensions ensures consistency
- Allows testing dose effects on latent experiential modes

#### 2.1 Initialization

```python
class TETPCALMEAnalyzer:
    def __init__(self, pc_scores: pd.DataFrame, components: List[str] = ['PC1', 'PC2']):
        """
        Initialize LME analyzer for PC scores.
        
        Args:
            pc_scores: DataFrame with PC scores from TETPCAAnalyzer
            components: List of components to analyze (default: ['PC1', 'PC2'])
        """
        self.pc_scores = pc_scores.copy()
        self.components = components
        self.models = {}  # Dict[component, MixedLMResults]
        self.results_df = None
```

#### 2.2 Prepare PC Data for LME

```python
def prepare_pc_data(self) -> pd.DataFrame:
    """
    Prepare PC scores for LME analysis.
    
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
- Same time centering as original LME analysis (Requirement 4)
- Consistent reference levels for comparability
- No additional transformations on PC scores (already standardized)

#### 2.3 Fit LME Models for PCs

```python
def fit_pc_models(self) -> Dict[str, MixedLMResults]:
    """
    Fit LME models for each principal component.
    
    Model specification (same as Requirement 4):
        PC ~ state * dose * time_c + (1 | subject)
    
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
- Formula: `"PC ~ C(state, Treatment('RS')) * C(dose, Treatment('Baja')) * time_c"`
- Groups: `groups=data['subject']`
- Method: REML (default)

#### 2.4 Extract Results

```python
def extract_results(self) -> pd.DataFrame:
    """
    Extract coefficients, standard errors, and p-values from fitted models.
    
    Returns:
        DataFrame with columns:
        - component: Component name (PC1, PC2)
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
- Same format as original LME results for consistency
- Include confidence intervals for effect size interpretation
- No FDR correction (only 2 components tested)

#### 2.5 Export Results

```python
def export_results(self, output_dir: str) -> Dict[str, str]:
    """
    Export LME results for PC scores.
    
    Creates:
    - pca_lme_results.csv: Coefficients and p-values for PC1, PC2
    
    Args:
        output_dir: Directory to save output files
    
    Returns:
        Dict mapping file types to file paths
    """
```

### 3. Main Analysis Script

**File**: `scripts/compute_pca_analysis.py`

**Purpose**: Orchestrate PCA analysis and LME fitting for PC scores.

#### 3.1 Workflow

```python
def main():
    """
    Main PCA analysis workflow.
    
    Steps:
    1. Load preprocessed TET data
    2. Initialize TETPCAAnalyzer with z-scored dimensions
    3. Fit PCA per subject
    4. Transform data to PC scores
    5. Export PCA results (loadings, variance, scores)
    6. Initialize TETPCALMEAnalyzer with PC scores
    7. Prepare PC data for LME
    8. Fit LME models for PC1 and PC2
    9. Export LME results
    10. Print summary to console
    """
```

#### 3.2 Command-Line Interface

```python
parser.add_argument('--input', default='results/tet/preprocessed/tet_preprocessed.csv')
parser.add_argument('--output', default='results/tet/pca')
parser.add_argument('--variance-threshold', type=float, default=0.75)
parser.add_argument('--components', nargs='+', default=['PC1', 'PC2'])
parser.add_argument('--verbose', action='store_true')
```

### 4. Inspection Script

**File**: `scripts/inspect_pca_results.py`

**Purpose**: Display PCA results in human-readable format.

#### 4.1 Display Elements

1. **Variance Explained Summary**
   - Total variance explained by retained components
   - Variance explained by PC1, PC2 individually
   - Number of components retained
   - Verification that threshold (70-80%) was met

2. **Top Loadings**
   - For each PC, show dimensions with highest absolute loadings
   - Interpretation guide (positive/negative loadings)
   - Identify which experiential dimensions drive each PC

3. **LME Results for PCs**
   - Significant effects for PC1 and PC2
   - Effect sizes and confidence intervals
   - Comparison with original dimension results
   - Interpretation of what PC effects mean experientially

## Data Structures

### PCA Loadings CSV

```csv
component,dimension,loading
PC1,pleasantness_z,0.45
PC1,unpleasantness_z,-0.32
PC1,emotional_intensity_z,0.52
PC1,elementary_imagery_z,0.28
PC1,complex_imagery_z,0.31
PC2,pleasantness_z,0.12
PC2,unpleasantness_z,0.67
PC2,emotional_intensity_z,-0.15
...
```

### PCA Variance Explained CSV

```csv
component,variance_explained,cumulative_variance
PC1,0.35,0.35
PC2,0.22,0.57
PC3,0.18,0.75
```

### PCA Scores CSV

```csv
subject,session_id,state,dose,t_bin,t_sec,PC1,PC2,PC3
S01,DMT_Alta_1,DMT,Alta,0,0,-0.45,0.23,0.12
S01,DMT_Alta_1,DMT,Alta,1,4,-0.38,0.31,0.08
...
```

### PCA LME Results CSV

```csv
component,effect,beta,se,z_value,p_value,ci_lower,ci_upper
PC1,Intercept,0.12,0.08,1.50,0.134,-0.04,0.28
PC1,state[T.DMT],0.45,0.12,3.75,0.0002,0.21,0.69
PC1,dose[T.Alta],0.23,0.10,2.30,0.021,0.03,0.43
PC1,state[T.DMT]:dose[T.Alta],-0.18,0.15,-1.20,0.230,-0.47,0.11
PC2,Intercept,-0.05,0.07,-0.71,0.478,-0.19,0.09
PC2,state[T.DMT],0.32,0.11,2.91,0.004,0.10,0.54
...
```

## Error Handling

### 1. Insufficient Variance
- **Issue**: Subject has very low variance (e.g., all dimensions near zero)
- **Handling**: Skip subject, log warning, continue with remaining subjects

### 2. Singular Matrix
- **Issue**: PCA fails due to perfect collinearity
- **Handling**: Remove constant dimensions, retry PCA

### 3. LME Convergence Failure
- **Issue**: LME model fails to converge for PC scores
- **Handling**: Try different optimizer, increase max iterations, log warning

### 4. Missing PC Scores
- **Issue**: Some subjects have fewer components than others
- **Handling**: Pad with NaN, exclude from LME for that component

## Testing Strategy

### Unit Tests

1. **test_pca_analyzer.py**
   - Test PCA fitting with known covariance structure
   - Verify variance threshold logic
   - Test loading extraction
   - Test score transformation

2. **test_pca_lme_analyzer.py**
   - Test data preparation (time centering)
   - Test LME fitting with synthetic PC scores
   - Verify result extraction format

### Integration Tests

3. **test_pca_pipeline.py**
   - Test complete pipeline from preprocessed data to results
   - Verify output file creation
   - Check CSV schema compliance

## Performance Considerations

- **PCA Fitting**: O(n_observations × n_dimensions²)
  - Expected: 16,200 observations × 15² dimensions ≈ 3.6M operations
  - Fast with sklearn (< 1 second total)
  - Single fit for all data

- **LME Fitting**: O(n_components × n_iterations)
  - Expected: 2 components × ~100 iterations ≈ 200 iterations
  - Moderate (5-10 seconds per component)

- **Memory**: Store single PCA model
  - Expected: ~10KB for model + 16,200 × n_components for scores
  - Negligible (< 1MB total)

## Dependencies

- `scikit-learn>=1.0`: PCA implementation
- `statsmodels>=0.13`: LME models
- `pandas>=1.3`: Data manipulation
- `numpy>=1.21`: Numerical operations

## Output Files

```
results/tet/pca/
├── pca_loadings.csv              # Component loadings per subject
├── pca_variance_explained.csv    # Variance explained per component
├── pca_scores.csv                # PC scores for all time points
└── pca_lme_results.csv           # LME results for PC1, PC2
```

## Validation

### 1. Variance Threshold Check
- Verify cumulative variance ≥ 70% (target threshold)
- Log actual cumulative variance achieved

### 2. Orthogonality Check
- Verify PC scores are uncorrelated (by construction)
- Check: `corr(PC1, PC2) ≈ 0` (should be exactly 0 for PCA)

### 3. Reconstruction Check
- Verify inverse transform approximates original data
- Check: `||X - X_reconstructed|| / ||X|| < 0.3` (given 70-80% variance)
- Reconstruction error should match (1 - cumulative_variance)

### 4. LME Diagnostics
- Check residual normality (Q-Q plots)
- Check homoscedasticity (residual vs fitted plots)
- Verify random effects variance > 0

## Interpretation Guidelines

### PC1 Interpretation
- Typically captures overall intensity/arousal
- High loadings on general_intensity, emotional_intensity
- Positive scores = high intensity, negative = low intensity

### PC2 Interpretation
- Typically captures valence or specific experiential mode
- May contrast positive (pleasantness, bliss) vs negative (anxiety, unpleasantness)
- Or contrast sensory (imagery) vs cognitive (selfhood, salience)

### Loading Interpretation
- Positive loading: dimension increases with PC score
- Negative loading: dimension decreases with PC score
- Magnitude: contribution strength (|loading| > 0.3 = substantial)

### LME Results Interpretation
- Significant state effect: PC differs between DMT and RS
- Significant dose effect: PC differs between high and low dose
- Significant interaction: dose effect differs by state

## Future Enhancements

1. **Varimax Rotation**: Apply rotation for more interpretable components
2. **Scree Plots**: Visualize variance explained by each component
3. **Biplot**: Visualize loadings and scores together
4. **PC Time Courses**: Plot PC scores over time by condition
5. **Subject-Specific PCA**: Compare group-level vs within-subject PCA structures
6. **Cross-Validation**: Assess stability of PCA structure with bootstrap resampling
