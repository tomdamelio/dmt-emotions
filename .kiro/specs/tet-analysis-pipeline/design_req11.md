# Design Document: Requirement 11 - Physiological-Affective Integration Analysis

## Overview

This design document specifies the implementation of physiological-affective integration analysis for the TET pipeline. The system will analyze relationships between autonomic physiological signals (Heart Rate, Skin Conductance, Respiratory Volume) and affective TET dimensions to test hypotheses about the autonomic correlates of subjective emotional experiences during psychedelic states. The analysis includes correlation analysis, regression with physiological PC1, hypothesis testing (arousal vs valence coupling), and Canonical Correlation Analysis (CCA) to identify shared latent dimensions.

## Requirements Reference

**Requirement 11: Physiological-Affective Integration Analysis**

User Story: As a researcher, I want the system to analyze relationships between physiological signals (ECG, EDA, Resp) and affective TET dimensions, so that I can test hypotheses about autonomic correlates of subjective emotional experiences during psychedelic states.

Key Acceptance Criteria:
- 11.1-11.2: Load and temporally align physiological data with TET
- 11.3-11.6: Compute correlations between TET affective dimensions and physiological measures
- 11.7-11.9: Regression analysis with physiological PC1, hypothesis testing
- 11.10-11.13: Canonical Correlation Analysis (CCA)
- 11.14-11.17: Generate visualizations
- 11.18-11.22: Export results and integrate into reports

## Architecture

### Component Overview

```
TETPhysioDataLoader
├── load_physiological_data()    # Load HR, SMNA AUC, RVT
├── load_tet_data()               # Load preprocessed TET data
├── temporal_alignment()          # Align physio to TET sampling rate
└── merge_datasets()              # Create unified dataset

TETPhysioCorrelationAnalyzer
├── compute_correlations()        # Pearson correlations
├── apply_fdr_correction()        # BH-FDR correction
├── compute_physio_pca()          # PCA on physiological signals
├── regression_analysis()         # Predict TET from physio PC1
├── test_arousal_valence_hypothesis() # Steiger's Z-test
└── export_results()              # Save correlation results

TETPhysioCCAAnalyzer
├── prepare_matrices()            # Prepare physio and TET matrices
├── fit_cca()                     # Canonical Correlation Analysis
├── extract_canonical_variates()  # Get canonical correlations
├── compute_canonical_loadings()  # Dimension contributions
├── test_significance()           # Wilks' Lambda test
└── export_results()              # Save CCA results

TETPhysioVisualizer
├── plot_correlation_heatmaps()   # TET-physio correlation matrices
├── plot_regression_scatter()     # TET vs physio PC1 scatter
├── plot_cca_loadings()           # Canonical loading biplots
└── export_figures()              # Save all figures
```



### Data Flow

```
Physiological Data                    TET Data
(results/ecg/, eda/, resp/)          (results/tet/preprocessed/)
    ↓                                     ↓
[Load HR, SMNA AUC, RVT]             [Load z-scored TET dimensions]
    ↓                                     ↓
[Temporal Alignment to 0.25 Hz]           ↓
    ↓                                     ↓
[Merge on (subject, session, t_bin)]
    ↓
[Unified Dataset: physio + TET]
    ↓
    ├─→ [Correlation Analysis]
    │   ├─ Arousal vs Physio (HR, SMNA, RVT)
    │   ├─ Valence vs Physio
    │   ├─ All TET_AFFECTIVE vs Physio
    │   └─ FDR correction
    │
    ├─→ [PCA on Physiological Signals]
    │   └─ Extract PC1 (dominant autonomic mode)
    │
    ├─→ [Regression Analysis]
    │   ├─ Arousal ~ Physio_PC1
    │   ├─ Valence ~ Physio_PC1
    │   └─ Steiger's Z-test (arousal vs valence)
    │
    └─→ [Canonical Correlation Analysis]
        ├─ Physio Matrix (3 measures) × TET Matrix (6 affective dims)
        ├─ Extract 2 canonical variates
        ├─ Compute canonical loadings
        └─ Test significance (Wilks' Lambda)
    ↓
[Export: correlations, regressions, CCA results, figures]
```

## Detailed Design

### 1. TETPhysioDataLoader Class

**Purpose**: Load and temporally align physiological signals with TET data.

**Rationale**:
- Physiological data sampled at higher rates (250 Hz raw, downsampled to various rates)
- TET data originally at 0.25 Hz (4-second bins), aggregate to 30-second bins
- 30-second bins match paper original specification and reduce noise
- Temporal alignment critical for valid correlation analysis
- Must handle missing data and session matching



#### 1.1 Initialization

```python
class TETPhysioDataLoader:
    def __init__(self, 
                 ecg_dir: str = 'results/ecg',
                 eda_dir: str = 'results/eda', 
                 resp_dir: str = 'results/resp',
                 tet_path: str = 'results/tet/preprocessed/tet_preprocessed.csv',
                 target_sampling_rate: float = 0.25):
        """
        Initialize physiological-TET data loader.
        
        Args:
            ecg_dir: Directory containing ECG results (HR)
            eda_dir: Directory containing EDA results (SMNA AUC)
            resp_dir: Directory containing Resp results (RVT)
            tet_path: Path to preprocessed TET data
            target_sampling_rate: Target sampling rate in Hz (0.25 = 4-second bins)
        """
        self.ecg_dir = Path(ecg_dir)
        self.eda_dir = Path(eda_dir)
        self.resp_dir = Path(resp_dir)
        self.tet_path = Path(tet_path)
        self.target_bin_duration_sec = 30  # 30-second bins
        self.target_sampling_rate = 1.0 / 30  # ~0.033 Hz
        
        self.physio_data = None
        self.tet_data = None
        self.merged_data = None
```

**Design Decisions**:
- Target bin duration: 30 seconds (matches paper original specification)
- Both TET and physiological data aggregated to 30-second bins
- Separate directories for each physiological modality
- Flexible paths for different project structures



#### 1.2 Load Physiological Data

```python
def load_physiological_data(self) -> pd.DataFrame:
    """
    Load preprocessed physiological measures from analysis results.
    
    Expected file structure:
    - results/ecg/{subject}_{session}_hr.csv
    - results/eda/{subject}_{session}_smna_auc.csv
    - results/resp/{subject}_{session}_rvt.csv
    
    Each CSV contains:
    - time_sec: Time in seconds
    - measure_value: HR (bpm), SMNA AUC (μS·s), or RVT (L/s)
    
    Process:
    1. Scan directories for all subject-session files
    2. Load each file and add metadata (subject, session, measure_type)
    3. Concatenate into single DataFrame
    4. Validate data completeness
    
    Returns:
        DataFrame with columns:
        - subject: Subject ID (S01-S20)
        - session: Session ID (DMT_1, DMT_2, Reposo_1, Reposo_2)
        - time_sec: Time in seconds
        - HR: Heart rate (bpm)
        - SMNA_AUC: Skin conductance AUC (μS·s)
        - RVT: Respiratory volume per time (L/s)
    """
```

**Implementation Details**:
- Use glob patterns to find files: `{subject}_{session}_*.csv`
- Parse subject and session from filename
- Handle missing files gracefully (log warning, continue)
- Pivot to wide format (one row per timepoint, columns for each measure)
- Validate expected ranges:
  - HR: 40-180 bpm (typical range)
  - SMNA_AUC: 0-10 μS·s (typical range)
  - RVT: -2 to 2 L/s (typical range)



#### 1.3 Temporal Alignment

```python
def temporal_alignment(self, physio_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate physiological data to 30-second bins.
    
    Process:
    1. For each subject-session:
       a. Extract physiological time series
       b. Create target time grid: 0, 30, 60, 90, ... seconds
       c. Aggregate to 30-second bins using mean
       d. Assign t_bin index (0, 1, 2, ...)
    
    2. Aggregation strategy:
       - Bin edges: [0-30), [30-60), [60-90), ... seconds
       - Aggregation function: mean (robust to outliers)
       - Minimum data points per bin: 50% of expected samples
    
    3. Handle edge cases:
       - Missing data: forward-fill then backward-fill (max 1 bin = 30s)
       - Session length mismatch: trim to shortest common duration
       - Bins with <50% data: mark as NaN
    
    4. Expected bin counts:
       - RS sessions: 600s / 30s = 20 bins
       - DMT sessions: 1200s / 30s = 40 bins
    
    Returns:
        DataFrame with columns:
        - subject, session, t_bin, t_sec
        - HR, SMNA_AUC, RVT (aggregated to 30-second bins)
    
    Example:
        subject | session    | t_bin | t_sec | HR   | SMNA_AUC | RVT
        --------|------------|-------|-------|------|----------|-----
        S01     | DMT_1      | 0     | 0     | 72.3 | 2.15     | 0.42
        S01     | DMT_1      | 1     | 30    | 73.1 | 2.18     | 0.45
        S01     | DMT_1      | 2     | 60    | 74.5 | 2.22     | 0.48
    """
```

**Implementation Details**:
- Use `pandas.resample('30S')` for time-based aggregation
- Aggregation function: mean (robust to outliers)
- No interpolation needed (only aggregation from higher to lower rate)
- Missing data tolerance: max 1 consecutive bin (30 seconds)
- Minimum data requirement: 50% of expected samples per bin
- Log alignment statistics (% data retained, bins with missing data)



#### 1.4 Load TET Data

```python
def load_tet_data(self) -> pd.DataFrame:
    """
    Load preprocessed TET data and aggregate to 30-second bins.
    
    Process:
    1. Load preprocessed TET data (originally at 4-second resolution)
    2. Aggregate to 30-second bins using mean
       - Bin edges: [0-30), [30-60), [60-90), ... seconds
       - Each bin contains ~7.5 original timepoints (30s / 4s)
    3. Recompute t_bin index for 30-second bins
    4. Extract z-scored affective dimensions
    
    Returns:
        DataFrame with columns:
        - subject, session_id, state, dose, t_bin, t_sec
        - All z-scored dimensions (*_z) aggregated to 30s bins
        - Focus on TET_AFFECTIVE_COLUMNS:
          * pleasantness_z
          * unpleasantness_z
          * emotional_intensity_z
          * interoception_z
          * bliss_z
          * anxiety_z
        - Computed indices:
          * valence_index_z = pleasantness_z - unpleasantness_z
    
    Expected bin counts after aggregation:
    - RS: 150 timepoints @ 4s → 20 bins @ 30s
    - DMT: 300 timepoints @ 4s → 40 bins @ 30s
    """
```

**Design Decisions**:
- Use preprocessed data (already z-scored within subject)
- Aggregate from 4s to 30s bins using mean (reduces noise)
- Extract only affective dimensions for analysis
- Compute valence index on-the-fly
- Preserve all metadata for grouping (state, dose)
- Use config.aggregate_tet_to_30s_bins() utility function

#### 1.5 Merge Datasets

```python
def merge_datasets(self) -> pd.DataFrame:
    """
    Merge physiological and TET data on (subject, session, t_bin).
    
    Process:
    1. Standardize session naming:
       - Physio: DMT_1, DMT_2, Reposo_1, Reposo_2
       - TET: session_id format (may differ)
       - Create mapping if needed
    
    2. Merge on (subject, session, t_bin)
       - Inner join: keep only matching timepoints
       - Log merge statistics (% data retained)
    
    3. Validate merged data:
       - Check for missing values
       - Verify expected session lengths
       - Ensure balanced state/dose distribution
    
    Returns:
        DataFrame with columns:
        - subject, session_id, state, dose, t_bin, t_sec
        - HR, SMNA_AUC, RVT
        - pleasantness_z, unpleasantness_z, emotional_intensity_z,
          interoception_z, bliss_z, anxiety_z
        - valence_index_z
    """
```

**Implementation Details**:
- Use `pd.merge()` with `how='inner'` (only complete cases)
- Session name mapping: handle different naming conventions
- Drop rows with any missing values in key variables
- Log merge report: N subjects, N sessions, N timepoints (30s bins), % retained
- Expected final N per subject:
  - RS: ~20 bins × 2 sessions = 40 timepoints
  - DMT: ~40 bins × 2 sessions = 80 timepoints
  - Total per subject: ~120 timepoints @ 30s bins



### 2. TETPhysioCorrelationAnalyzer Class

**Purpose**: Compute correlations between TET affective dimensions and physiological measures.

**Rationale**:
- Test specific hypotheses about autonomic-affective coupling
- Arousal (emotional_intensity) expected to correlate with autonomic activation
- Valence (pleasantness - unpleasantness) may show weaker or different patterns
- State-specific analysis (RS vs DMT) reveals context-dependent coupling

#### 2.1 Initialization

```python
class TETPhysioCorrelationAnalyzer:
    def __init__(self, merged_data: pd.DataFrame):
        """
        Initialize correlation analyzer.
        
        Args:
            merged_data: Merged physio-TET dataset from TETPhysioDataLoader
        """
        self.data = merged_data.copy()
        self.physio_measures = ['HR', 'SMNA_AUC', 'RVT']
        self.tet_affective = [
            'pleasantness_z', 'unpleasantness_z', 'emotional_intensity_z',
            'interoception_z', 'bliss_z', 'anxiety_z'
        ]
        self.correlation_results = None
        self.regression_results = None
        self.physio_pca_model = None
        self.physio_pc1_scores = None
```



#### 2.2 Compute Correlations

```python
def compute_correlations(self, by_state: bool = True) -> pd.DataFrame:
    """
    Compute Pearson correlations between TET and physiological measures.
    
    Analysis families (for FDR correction):
    1. Arousal-Physiology: emotional_intensity vs (HR, SMNA_AUC, RVT)
    2. Valence-Physiology: valence_index_z vs (HR, SMNA_AUC, RVT)
    3. All Affective-Physiology: 6 TET dims × 3 physio measures = 18 tests
    
    Process:
    1. For each state (RS, DMT) if by_state=True:
       a. Filter data by state
       b. For each TET-physio pair:
          - Compute Pearson r
          - Compute p-value
          - Compute 95% CI (Fisher z-transform)
       c. Store results
    
    2. Apply FDR correction within each family
    
    Returns:
        DataFrame with columns:
        - tet_dimension: TET dimension name
        - physio_measure: Physiological measure name
        - state: RS or DMT (if by_state=True)
        - r: Pearson correlation coefficient
        - p_value: Two-tailed p-value
        - p_fdr: FDR-corrected p-value
        - ci_lower, ci_upper: 95% confidence interval
        - n_obs: Number of observations
    """
```

**Implementation Details**:
- Use `scipy.stats.pearsonr()` for correlation and p-value
- Fisher z-transform for confidence intervals:
  ```python
  z = np.arctanh(r)
  se_z = 1 / np.sqrt(n - 3)
  ci_z = z ± 1.96 * se_z
  ci_r = np.tanh(ci_z)
  ```
- FDR correction: `statsmodels.stats.multitest.multipletests(method='fdr_bh')`
- Separate FDR correction for each analysis family
- Minimum N observations: 30 (log warning if less)



#### 2.3 Compute Physiological PCA

```python
def compute_physio_pca(self) -> Tuple[PCA, pd.DataFrame]:
    """
    Perform PCA on physiological signals to extract PC1.
    
    Purpose:
    - PC1 captures dominant mode of autonomic variation
    - Represents shared variance across HR, SMNA, RVT
    - Hypothesis: arousal correlates more strongly with PC1 than valence
    
    Process:
    1. Standardize physiological measures (z-score within each measure)
    2. Fit PCA on (HR_z, SMNA_AUC_z, RVT_z)
    3. Extract PC1 scores for all timepoints
    4. Compute variance explained by PC1
    
    Returns:
        - pca_model: Fitted PCA model
        - pc1_scores: DataFrame with PC1 scores and metadata
    
    PC1 Interpretation:
    - High PC1: Increased autonomic activation (↑HR, ↑SC, ↑RVT)
    - Low PC1: Decreased autonomic activation (↓HR, ↓SC, ↓RVT)
    - Loadings show contribution of each measure to PC1
    """
```

**Implementation Details**:
- Use `sklearn.decomposition.PCA(n_components=3)`
- Standardize inputs: `StandardScaler().fit_transform()`
- Extract PC1 scores: `pca.transform()[:, 0]`
- Store loadings for interpretation
- Typical PC1 variance explained: 40-60%



#### 2.4 Regression Analysis

```python
def regression_analysis(self, by_state: bool = True) -> pd.DataFrame:
    """
    Fit linear regression models predicting TET from physiological PC1.
    
    Models:
    1. emotional_intensity_z ~ physio_PC1
    2. valence_index_z ~ physio_PC1
    3. Each TET_AFFECTIVE dimension ~ physio_PC1
    
    For each state (RS, DMT) if by_state=True:
    - Fit OLS regression
    - Extract standardized beta, R², p-value
    - Compute 95% CI for beta
    
    Returns:
        DataFrame with columns:
        - outcome_variable: TET dimension name
        - predictor: 'physio_PC1'
        - state: RS or DMT
        - beta: Standardized regression coefficient
        - r_squared: Proportion of variance explained
        - p_value: Significance of beta
        - ci_lower, ci_upper: 95% CI for beta
        - n_obs: Number of observations
    """
```

**Implementation Details**:
- Use `statsmodels.api.OLS()` for regression
- Standardize both predictor and outcome for standardized beta
- Extract statistics from model summary
- Confidence intervals from `model.conf_int()`
- Check assumptions:
  - Linearity: residual plot
  - Homoscedasticity: Breusch-Pagan test
  - Normality: Shapiro-Wilk test (if N < 5000)



#### 2.5 Test Arousal vs Valence Hypothesis

```python
def test_arousal_valence_hypothesis(self) -> pd.DataFrame:
    """
    Test hypothesis: arousal-PC1 correlation > valence-PC1 correlation.
    
    Uses Steiger's Z-test for comparing dependent correlations:
    - H0: |r_arousal_PC1| = |r_valence_PC1|
    - H1: |r_arousal_PC1| > |r_valence_PC1|
    
    Process:
    1. Compute r_arousal_PC1 and r_valence_PC1
    2. Compute r_arousal_valence (correlation between arousal and valence)
    3. Apply Steiger's Z-test formula:
       Z = (z1 - z2) / sqrt(SE)
       where z1, z2 are Fisher z-transforms
       SE accounts for correlation between arousal and valence
    
    4. Compute one-tailed p-value (directional hypothesis)
    5. Repeat for each state (RS, DMT)
    
    Returns:
        DataFrame with columns:
        - state: RS or DMT
        - r_arousal_pc1: Arousal-PC1 correlation
        - r_valence_pc1: Valence-PC1 correlation
        - r_arousal_valence: Arousal-valence correlation
        - z_statistic: Steiger's Z
        - p_value: One-tailed p-value
        - conclusion: 'arousal > valence' or 'no difference'
    """
```

**Implementation Details**:
- Use `pingouin.corr()` or implement Steiger's test manually
- Formula for Steiger's Z:
  ```python
  z1 = np.arctanh(r_arousal_pc1)
  z2 = np.arctanh(r_valence_pc1)
  r_jk = r_arousal_valence
  
  denominator = np.sqrt(
      (2 * (n - 1) * (1 - r_jk)) / 
      ((n - 3) * (1 + r_jk))
  )
  z_stat = (z1 - z2) / denominator
  p_value = 1 - stats.norm.cdf(z_stat)  # one-tailed
  ```
- Significance threshold: α = 0.05
- Effect size: difference in r values



### 3. TETPhysioCCAAnalyzer Class

**Purpose**: Perform Canonical Correlation Analysis to identify shared latent dimensions.

**Rationale**:
- CCA finds linear combinations of physio and TET that maximize correlation
- Reveals multivariate relationships beyond pairwise correlations
- Identifies shared latent dimensions (e.g., "autonomic arousal" dimension)
- Tests whether physio-affective coupling differs between RS and DMT

#### 3.1 Initialization

```python
class TETPhysioCCAAnalyzer:
    def __init__(self, merged_data: pd.DataFrame):
        """
        Initialize CCA analyzer.
        
        Args:
            merged_data: Merged physio-TET dataset
        """
        self.data = merged_data.copy()
        self.physio_measures = ['HR', 'SMNA_AUC', 'RVT']
        self.tet_affective = [
            'pleasantness_z', 'unpleasantness_z', 'emotional_intensity_z',
            'interoception_z', 'bliss_z', 'anxiety_z'
        ]
        self.cca_models = {}  # Dict[state, CCA]
        self.canonical_correlations = {}
        self.canonical_loadings = {}
```



#### 3.2 Prepare Matrices

```python
def prepare_matrices(self, state: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare physiological and TET matrices for CCA.
    
    Process:
    1. Filter data by state (RS or DMT)
    2. Extract physiological matrix X (n_obs × 3)
       Columns: HR, SMNA_AUC, RVT
    3. Extract TET affective matrix Y (n_obs × 6)
       Columns: pleasantness_z, unpleasantness_z, emotional_intensity_z,
                interoception_z, bliss_z, anxiety_z
    4. Standardize both matrices (z-score each column)
    5. Remove rows with missing values
    
    Args:
        state: 'RS' or 'DMT'
    
    Returns:
        X: Physiological matrix (n_obs × 3)
        Y: TET affective matrix (n_obs × 6)
    """
```

**Design Decisions**:
- Standardize inputs for CCA (required for interpretability)
- Complete cases only (drop rows with any missing values)
- Separate CCA for each state (test context-dependent coupling)
- Minimum N: 100 observations (rule of thumb: 10× variables)



#### 3.3 Fit CCA

```python
def fit_cca(self, n_components: int = 2) -> Dict[str, CCA]:
    """
    Fit Canonical Correlation Analysis for each state.
    
    CCA finds linear combinations:
    - U = X @ W_x (canonical variates for physio)
    - V = Y @ W_y (canonical variates for TET)
    
    Such that corr(U_i, V_i) is maximized for each pair i.
    
    Process:
    1. For each state (RS, DMT):
       a. Prepare matrices X (physio) and Y (TET)
       b. Fit CCA with n_components canonical variates
       c. Extract canonical correlations (r_i)
       d. Store fitted model
    
    Args:
        n_components: Number of canonical variate pairs (default: 2)
    
    Returns:
        Dict mapping state to fitted CCA model
    
    Interpretation:
    - Canonical variate 1: Strongest shared dimension
    - Canonical variate 2: Second strongest (orthogonal to 1)
    - Canonical correlation r_i: Strength of relationship for pair i
    """
```

**Implementation Details**:
- Use `sklearn.cross_decomposition.CCA(n_components=2)`
- Fit separately for RS and DMT
- Extract canonical correlations: `np.corrcoef(U.T, V.T)`
- Typical r_1: 0.4-0.7 (moderate to strong)
- Typical r_2: 0.2-0.4 (weak to moderate)



#### 3.4 Extract Canonical Variates and Test Significance

```python
def extract_canonical_variates(self) -> pd.DataFrame:
    """
    Extract canonical correlations and test significance.
    
    For each state and canonical variate pair:
    1. Extract canonical correlation r_i
    2. Compute Wilks' Lambda for significance test
    3. Compute degrees of freedom
    4. Compute p-value (chi-square approximation)
    
    Wilks' Lambda test:
    - H0: No relationship between X and Y
    - Lambda = product of (1 - r_i²) for all i
    - Chi-square statistic: -n * ln(Lambda)
    - df = p * q (p = dim(X), q = dim(Y))
    
    Returns:
        DataFrame with columns:
        - state: RS or DMT
        - canonical_variate: 1 or 2
        - canonical_correlation: r_i
        - wilks_lambda: Wilks' Lambda statistic
        - chi_square: Chi-square test statistic
        - df: Degrees of freedom
        - p_value: Significance of canonical correlation
    """
```

**Implementation Details**:
- Wilks' Lambda formula:
  ```python
  lambda_wilks = np.prod([1 - r**2 for r in canonical_corrs])
  chi_square = -n * np.log(lambda_wilks)
  df = p * q  # p=3 physio, q=6 TET → df=18
  p_value = 1 - stats.chi2.cdf(chi_square, df)
  ```
- Significance threshold: α = 0.05
- Typically only first 1-2 variates are significant



#### 3.5 Compute Canonical Loadings

```python
def compute_canonical_loadings(self) -> pd.DataFrame:
    """
    Compute canonical loadings (structure coefficients).
    
    Canonical loadings = correlations between original variables
    and canonical variates.
    
    For physiological variables:
    - Loading_X_i = corr(X_j, U_i) for each variable j and variate i
    
    For TET variables:
    - Loading_Y_i = corr(Y_k, V_i) for each variable k and variate i
    
    Interpretation:
    - High loading (|r| > 0.3): Variable contributes strongly to variate
    - Sign indicates direction of relationship
    - Loadings reveal "meaning" of each canonical dimension
    
    Returns:
        DataFrame with columns:
        - state: RS or DMT
        - canonical_variate: 1 or 2
        - variable_set: 'physio' or 'tet'
        - variable_name: Original variable name
        - loading: Canonical loading (correlation)
    
    Example interpretation:
        Canonical Variate 1 (r = 0.65):
        Physio loadings: HR (0.82), SMNA (0.71), RVT (0.45)
        TET loadings: emotional_intensity (0.78), anxiety (0.62), 
                      interoception (0.54)
        → Interpretation: "Autonomic arousal" dimension
    """
```

**Implementation Details**:
- Compute loadings as correlations:
  ```python
  U = cca.transform(X)  # Canonical variates for physio
  V = cca.transform(Y)  # Canonical variates for TET
  
  loadings_X = np.corrcoef(X.T, U.T)[:p, p:]
  loadings_Y = np.corrcoef(Y.T, V.T)[:q, q:]
  ```
- Convert to long format for easy visualization
- Highlight loadings with |r| > 0.3 (meaningful contribution)



### 4. TETPhysioVisualizer Class

**Purpose**: Generate publication-ready visualizations of physio-TET relationships.

#### 4.1 Correlation Heatmaps

```python
def plot_correlation_heatmaps(self, correlation_df: pd.DataFrame, 
                               output_dir: str) -> List[str]:
    """
    Generate correlation heatmaps for TET-physio relationships.
    
    Creates:
    1. Separate heatmap for each state (RS, DMT)
    2. Rows: TET affective dimensions (6)
    3. Columns: Physiological measures (3)
    4. Cell values: Pearson r
    5. Cell annotations: r value + significance markers
       * p_fdr < 0.05
       ** p_fdr < 0.01
       *** p_fdr < 0.001
    
    Color scale:
    - Blue (negative): -1 to 0
    - White (zero): 0
    - Red (positive): 0 to +1
    
    Layout:
    - 2 panels side-by-side (RS | DMT)
    - Figure size: 10×6 inches
    - 300 DPI for publication
    
    Returns:
        List of generated figure paths
    """
```

**Design Decisions**:
- Use `seaborn.heatmap()` with diverging colormap
- Annotate cells with r values (2 decimals)
- Add significance stars based on p_fdr
- Order TET dimensions by average |r| (strongest first)
- Include colorbar with clear labels



#### 4.2 Regression Scatter Plots

```python
def plot_regression_scatter(self, merged_data: pd.DataFrame,
                             pc1_scores: pd.DataFrame,
                             regression_df: pd.DataFrame,
                             output_dir: str) -> List[str]:
    """
    Generate scatter plots for TET vs physiological PC1.
    
    Creates two figures:
    
    Figure 1: Arousal vs Physio PC1
    - 2 panels (RS | DMT)
    - X-axis: Physiological PC1
    - Y-axis: Emotional intensity (z-scored)
    - Points: Individual timepoints (semi-transparent)
    - Regression line: OLS fit with 95% CI band
    - Annotations: β, R², p-value
    
    Figure 2: Valence vs Physio PC1
    - Same layout as Figure 1
    - Y-axis: Valence index (z-scored)
    
    Styling:
    - Point color: Light gray with alpha=0.3
    - Regression line: Solid, color by state
    - CI band: Shaded, alpha=0.2
    - Grid: Light gray, dashed
    
    Returns:
        List of generated figure paths
    """
```

**Implementation Details**:
- Use `seaborn.regplot()` for scatter + regression
- Add text annotations with `plt.text()`:
  ```python
  text = f"β = {beta:.2f}\nR² = {r2:.3f}\np = {p_value:.3f}"
  ```
- Limit points displayed if N > 5000 (random sample for clarity)
- Figure size: 12×5 inches (2 panels)



#### 4.3 CCA Loading Plots

```python
def plot_cca_loadings(self, canonical_loadings_df: pd.DataFrame,
                      canonical_correlations_df: pd.DataFrame,
                      output_dir: str) -> List[str]:
    """
    Generate CCA canonical loading visualizations.
    
    Creates:
    1. Biplot for each state (RS, DMT)
       - X-axis: Canonical Variate 1
       - Y-axis: Canonical Variate 2
       - Arrows: Canonical loadings for each variable
       - Arrow color: Blue (physio), Red (TET)
       - Arrow length: Loading magnitude
       - Labels: Variable names at arrow tips
    
    2. Bar chart for each canonical variate
       - Separate bars for physio (blue) and TET (red)
       - X-axis: Variables
       - Y-axis: Canonical loading
       - Horizontal line at ±0.3 (meaningful threshold)
    
    Annotations:
    - Include canonical correlation r_i in title
    - Mark significant variates (p < 0.05) with asterisk
    
    Returns:
        List of generated figure paths
    """
```

**Design Decisions**:
- Biplot shows multivariate structure at a glance
- Bar charts show individual contributions clearly
- Color coding distinguishes variable sets
- Threshold line (±0.3) highlights meaningful loadings
- Figure size: 10×8 inches for biplot, 12×5 for bar charts



## Data Structures

### Correlation Results CSV

```csv
tet_dimension,physio_measure,state,r,p_value,p_fdr,ci_lower,ci_upper,n_obs
emotional_intensity_z,HR,RS,0.42,0.001,0.003,-0.38,0.46,1250
emotional_intensity_z,SMNA_AUC,RS,0.38,0.002,0.004,0.34,0.42,1250
emotional_intensity_z,RVT,RS,0.25,0.015,0.025,0.20,0.30,1250
valence_index_z,HR,RS,0.15,0.082,0.123,0.09,0.21,1250
valence_index_z,SMNA_AUC,RS,0.08,0.245,0.294,-0.04,0.20,1250
...
```

### Regression Results CSV

```csv
outcome_variable,predictor,state,beta,r_squared,p_value,ci_lower,ci_upper,n_obs
emotional_intensity_z,physio_PC1,RS,0.48,0.23,<0.001,0.42,0.54,1250
valence_index_z,physio_PC1,RS,0.18,0.03,0.045,0.02,0.34,1250
emotional_intensity_z,physio_PC1,DMT,0.62,0.38,<0.001,0.57,0.67,2850
valence_index_z,physio_PC1,DMT,0.22,0.05,0.012,0.05,0.39,2850
...
```

### CCA Results CSV

```csv
state,canonical_variate,canonical_correlation,wilks_lambda,chi_square,df,p_value
RS,1,0.65,0.42,1125.3,18,<0.001
RS,2,0.38,0.86,187.5,10,<0.001
DMT,1,0.71,0.35,2847.2,18,<0.001
DMT,2,0.42,0.82,542.8,10,<0.001
```

### CCA Loadings CSV

```csv
state,canonical_variate,variable_set,variable_name,loading
RS,1,physio,HR,0.82
RS,1,physio,SMNA_AUC,0.71
RS,1,physio,RVT,0.45
RS,1,tet,emotional_intensity_z,0.78
RS,1,tet,anxiety_z,0.62
RS,1,tet,interoception_z,0.54
RS,1,tet,pleasantness_z,-0.32
...
```



## Error Handling

### 1. Missing Physiological Data
- **Issue**: Some subjects/sessions lack physiological data
- **Handling**: 
  - Log missing files with subject/session details
  - Continue with available data
  - Report N subjects/sessions in final results
  - Minimum threshold: 5 subjects per state for analysis

### 2. Temporal Alignment Failures
- **Issue**: Physiological and TET data have mismatched timestamps
- **Handling**:
  - Attempt interpolation/aggregation
  - If >20% data loss, log warning
  - If >50% data loss, skip that session
  - Document alignment quality in metadata

### 3. Insufficient Sample Size
- **Issue**: Too few observations for reliable statistics
- **Handling**:
  - Minimum N for correlations: 30
  - Minimum N for regression: 50
  - Minimum N for CCA: 100
  - Skip analysis and log warning if below threshold

### 4. CCA Convergence Issues
- **Issue**: CCA fails to converge or produces degenerate solutions
- **Handling**:
  - Check for multicollinearity (VIF > 10)
  - Try regularized CCA if standard CCA fails
  - Reduce n_components if needed
  - Log warning and skip if unrecoverable

### 5. Assumption Violations
- **Issue**: Regression assumptions violated (non-normality, heteroscedasticity)
- **Handling**:
  - Log diagnostic test results
  - Consider robust regression if violations severe
  - Report limitations in results document
  - Proceed with caution, note in interpretation



## Testing Strategy

### Unit Tests

1. **test_physio_data_loader.py**
   - Test loading physiological data from CSV files
   - Test temporal alignment with different sampling rates
   - Test handling of missing data
   - Verify merged dataset structure

2. **test_correlation_analyzer.py**
   - Test Pearson correlation computation
   - Test FDR correction application
   - Test confidence interval calculation
   - Verify correlation results format

3. **test_regression_analyzer.py**
   - Test OLS regression fitting
   - Test standardized beta extraction
   - Test Steiger's Z-test implementation
   - Verify regression results format

4. **test_cca_analyzer.py**
   - Test CCA fitting with synthetic data
   - Test canonical correlation extraction
   - Test Wilks' Lambda computation
   - Test canonical loading calculation
   - Verify CCA results format

### Integration Tests

5. **test_physio_tet_pipeline.py**
   - Test complete pipeline from raw data to results
   - Verify output file creation
   - Check CSV schema compliance
   - Test with real data subset

### Validation Tests

6. **test_statistical_validity.py**
   - Verify correlation p-values match scipy
   - Verify regression results match statsmodels
   - Verify CCA results match sklearn
   - Test FDR correction against known examples



## Performance Considerations

- **Data Loading**: O(n_files × file_size)
  - Expected: ~40 files × 1MB ≈ 40MB
  - Fast (< 5 seconds)

- **Temporal Alignment**: O(n_sessions × n_timepoints)
  - Expected: ~80 sessions × 300 timepoints ≈ 24K operations
  - Fast (< 2 seconds)

- **Correlation Computation**: O(n_tests × n_observations)
  - Expected: ~36 tests × 4000 observations ≈ 144K operations
  - Fast (< 1 second)

- **Regression Fitting**: O(n_models × n_observations)
  - Expected: ~12 models × 4000 observations ≈ 48K operations
  - Fast (< 2 seconds)

- **CCA Fitting**: O(n_states × n_iterations × n_observations × n_variables²)
  - Expected: 2 states × 100 iterations × 4000 obs × 9² ≈ 65M operations
  - Moderate (5-10 seconds)

- **Total Pipeline**: ~20-30 seconds

- **Memory**: Store merged dataset + results
  - Expected: ~50MB for data + 5MB for results
  - Negligible (< 100MB total)

## Dependencies

- `pandas>=1.3`: Data manipulation
- `numpy>=1.21`: Numerical operations
- `scipy>=1.7`: Statistical tests (pearsonr, chi2)
- `scikit-learn>=1.0`: PCA, CCA
- `statsmodels>=0.13`: OLS regression, FDR correction
- `matplotlib>=3.5`: Visualization
- `seaborn>=0.11`: Enhanced visualization
- `pingouin>=0.5`: Steiger's Z-test (optional, can implement manually)



## Output Files

```
results/tet/physio_correlation/
├── correlation_results.csv           # All TET-physio correlations
├── regression_results.csv            # Regression: TET ~ physio_PC1
├── arousal_valence_hypothesis.csv    # Steiger's Z-test results
├── physio_pca_results.csv            # PCA on physiological signals
├── cca_results.csv                   # Canonical correlations
├── cca_loadings.csv                  # Canonical loadings
├── figures/
│   ├── correlation_heatmap_rs.png
│   ├── correlation_heatmap_dmt.png
│   ├── arousal_vs_pc1_scatter.png
│   ├── valence_vs_pc1_scatter.png
│   ├── cca_biplot_rs.png
│   ├── cca_biplot_dmt.png
│   ├── cca_loadings_cv1.png
│   └── cca_loadings_cv2.png
└── physio_tet_integration_report.md  # Summary report
```

## Integration with Pipeline

### Modification to `scripts/run_tet_analysis.py`

Add physio-correlation stage after PCA/ICA:

```python
def _define_stages(self) -> List[Tuple[str, callable]]:
    return [
        ('preprocessing', self._run_preprocessing),
        ('descriptive', self._run_descriptive_stats),
        ('lme', self._run_lme_models),
        ('pca', self._run_pca_analysis),
        ('ica', self._run_ica_analysis),
        ('physio_correlation', self._run_physio_correlation),  # NEW STAGE
        ('clustering', self._run_clustering_analysis),
        ('figures', self._run_figure_generation),
        ('report', self._run_report_generation)
    ]

def _run_physio_correlation(self):
    """Execute physiological-TET correlation analysis stage."""
    self.logger.info("Computing physiological-TET correlation analysis...")
    from compute_physio_correlation import main as physio_corr_main
    
    original_argv = sys.argv.copy()
    sys.argv = ['compute_physio_correlation.py']
    
    try:
        physio_corr_main()
    finally:
        sys.argv = original_argv
```

### Validator Update

```python
def _validate_physio_correlation_inputs(self) -> Tuple[bool, str]:
    """Validate physio-correlation inputs."""
    required = [
        'results/tet/preprocessed/tet_preprocessed.csv',
        'results/ecg',
        'results/eda',
        'results/resp'
    ]
    
    missing = []
    for path in required:
        if not Path(path).exists():
            missing.append(path)
    
    if missing:
        return False, (
            f"Missing required files/directories for physio-correlation:\n" +
            "\n".join(f"  - {f}" for f in missing) +
            "\n\nEnsure physiological data has been processed."
        )
    
    return True, ""
```



## Main Analysis Script

**File**: `scripts/compute_physio_correlation.py`

### Workflow

```python
def main():
    """
    Main physiological-TET correlation analysis workflow.
    
    Steps:
    1. Initialize TETPhysioDataLoader
    2. Load physiological data (HR, SMNA AUC, RVT)
    3. Load TET data (preprocessed with z-scored dimensions)
    4. Perform temporal alignment to 0.25 Hz
    5. Merge datasets on (subject, session, t_bin)
    6. Log merge statistics
    
    7. Initialize TETPhysioCorrelationAnalyzer
    8. Compute correlations (arousal, valence, all affective dims)
    9. Apply FDR correction
    10. Compute physiological PCA (extract PC1)
    11. Fit regression models (TET ~ physio_PC1)
    12. Test arousal vs valence hypothesis (Steiger's Z)
    13. Export correlation and regression results
    
    14. Initialize TETPhysioCCAAnalyzer
    15. Prepare matrices for each state (RS, DMT)
    16. Fit CCA (2 canonical variates)
    17. Extract canonical correlations and test significance
    18. Compute canonical loadings
    19. Export CCA results
    
    20. Initialize TETPhysioVisualizer
    21. Generate correlation heatmaps
    22. Generate regression scatter plots
    23. Generate CCA loading plots
    24. Export all figures
    
    25. Generate integration report
    26. Print summary to console
    """
```

### Command-Line Interface

```python
parser = argparse.ArgumentParser(
    description='Analyze physiological-TET correlations'
)
parser.add_argument('--tet-data', 
                    default='results/tet/preprocessed/tet_preprocessed.csv')
parser.add_argument('--ecg-dir', default='results/ecg')
parser.add_argument('--eda-dir', default='results/eda')
parser.add_argument('--resp-dir', default='results/resp')
parser.add_argument('--output', default='results/tet/physio_correlation')
parser.add_argument('--n-cca-components', type=int, default=2)
parser.add_argument('--by-state', action='store_true', default=True,
                    help='Analyze RS and DMT separately')
parser.add_argument('--verbose', action='store_true')
```



## Interpretation Guidelines

### Correlation Interpretation

**Strength Guidelines** (Cohen, 1988):
- Small: |r| = 0.10 - 0.29
- Medium: |r| = 0.30 - 0.49
- Large: |r| ≥ 0.50

**Expected Patterns**:
1. **Arousal (emotional_intensity)**:
   - Moderate to strong positive correlations with all physio measures
   - Strongest with HR and SMNA_AUC
   - Weaker with RVT (respiratory less directly linked to arousal)

2. **Valence (pleasantness - unpleasantness)**:
   - Weak to moderate correlations
   - May show different patterns in RS vs DMT
   - Less consistent across physiological measures

3. **Anxiety**:
   - Strong positive correlations with autonomic activation
   - Similar pattern to arousal but potentially stronger

4. **Bliss/Pleasantness**:
   - Weak or negative correlations with autonomic activation
   - May show parasympathetic dominance (↓HR, ↓SC)

### Regression Interpretation

**R² Guidelines**:
- Small: R² = 0.01 - 0.08 (1-8% variance explained)
- Medium: R² = 0.09 - 0.24 (9-24% variance explained)
- Large: R² ≥ 0.25 (≥25% variance explained)

**Hypothesis Testing**:
- If arousal-PC1 R² significantly > valence-PC1 R²:
  - Supports hypothesis that autonomic signals primarily reflect arousal
  - Valence may be less directly coupled to peripheral physiology
  - Consistent with dimensional models of emotion

### CCA Interpretation

**Canonical Correlation Guidelines**:
- Weak: r < 0.30
- Moderate: r = 0.30 - 0.59
- Strong: r ≥ 0.60

**Canonical Variate Interpretation**:

**Typical Pattern for CV1** (strongest):
- Physio loadings: High on all measures (HR, SC, RVT)
- TET loadings: High on arousal-related dimensions
- Interpretation: "General autonomic arousal" dimension
- Expected r: 0.50 - 0.70

**Typical Pattern for CV2** (second):
- Physio loadings: Mixed or specific to one measure
- TET loadings: Valence or specific affective quality
- Interpretation: "Affective quality" dimension
- Expected r: 0.30 - 0.50

**State Differences** (RS vs DMT):
- RS: Weaker coupling (lower canonical correlations)
- DMT: Stronger coupling (higher canonical correlations)
- Interpretation: Psychedelic state enhances mind-body integration



## Validation

### 1. Temporal Alignment Validation
- Check alignment quality: % timepoints matched
- Verify no systematic time shifts (cross-correlation)
- Expected: >90% timepoints successfully aligned
- Visual inspection: plot physio and TET time series overlaid

### 2. Correlation Validity
- Verify correlations are within valid range [-1, 1]
- Check for outliers (|r| > 0.95 suggests data issues)
- Compare with literature values (expected ranges)
- Validate p-values: should be uniformly distributed under H0

### 3. Regression Diagnostics
- **Linearity**: Residual vs fitted plot (should be random scatter)
- **Homoscedasticity**: Breusch-Pagan test (p > 0.05)
- **Normality**: Q-Q plot of residuals (should be linear)
- **Independence**: Durbin-Watson statistic (1.5 - 2.5)
- **Multicollinearity**: VIF < 5 for predictors

### 4. CCA Validity
- Check canonical correlations are decreasing (r1 > r2)
- Verify Wilks' Lambda is between 0 and 1
- Validate loadings are within [-1, 1]
- Check for degenerate solutions (all loadings near zero)
- Compare with permutation test (shuffle Y, recompute CCA)

### 5. Cross-Validation
- Split data by subjects (leave-one-subject-out)
- Recompute correlations and regressions
- Check stability of effect sizes across folds
- Expected: correlation r should vary by <0.10 across folds

## Limitations and Caveats

1. **Temporal Resolution**:
   - TET sampled at 4-second bins (coarse)
   - Physiological signals have faster dynamics
   - Aggregation may miss rapid fluctuations

2. **Causality**:
   - Correlations do not imply causation
   - Bidirectional relationships possible
   - Third variables may drive both physio and TET

3. **Individual Differences**:
   - Within-subject z-scoring controls for scale usage
   - But individual physiology-affect coupling may vary
   - Group-level patterns may not apply to all individuals

4. **State Confounds**:
   - DMT induces both physiological and psychological changes
   - Difficult to separate drug effects from experience effects
   - RS provides baseline but limited affective range

5. **Measurement Error**:
   - Physiological signals contain noise and artifacts
   - TET ratings are subjective and may have recall bias
   - Both sources of error attenuate correlations

## Future Extensions

1. **Time-Lagged Correlations**:
   - Test whether physio predicts TET (or vice versa) with time lag
   - Identify temporal precedence

2. **Dynamic Connectivity**:
   - Compute time-varying correlations (sliding windows)
   - Identify periods of strong vs weak coupling

3. **Multivariate Patterns**:
   - Machine learning to predict TET from physio
   - Identify non-linear relationships

4. **Individual Differences**:
   - Cluster subjects by physio-affect coupling patterns
   - Relate to personality or drug sensitivity

5. **Mediation Analysis**:
   - Test whether physio mediates dose effects on TET
   - Identify causal pathways
