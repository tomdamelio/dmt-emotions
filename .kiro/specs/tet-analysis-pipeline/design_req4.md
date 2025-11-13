# Design Document: Requirement 4 - Linear Mixed Effects Modeling

## Overview

This design document describes the implementation of Linear Mixed Effects (LME) modeling for TET data. The system will fit LME models to test dose and state effects on each dimension, controlling for repeated measures within subjects.

## Statistical Model

### Model Formula

For each dimension, we fit the following LME model:

```
Y ~ State + Dose + Time_c + State:Dose + State:Time_c + Dose:Time_c + (1|Subject)
```

Where:
- **Y**: Z-scored dimension value
- **State**: Categorical (RS, DMT)
- **Dose**: Categorical (Alta, Baja) 
- **Time_c**: Continuous, centered time (t_bin - mean(t_bin))
- **State:Dose**: Interaction between state and dose
- **State:Time_c**: Interaction between state and time
- **Dose:Time_c**: Interaction between dose and time
- **(1|Subject)**: Random intercept for subject

### Analysis Window

- **Time bins**: 0-18 (corresponding to 0-540 seconds, 0-9 minutes)
- **Rationale**: Focus on acute effects period, consistent across RS and DMT

### Reference Levels

- **State**: RS (reference)
- **Dose**: Baja (reference)

This means:
- State coefficient = DMT effect relative to RS
- Dose coefficient = Alta effect relative to Baja

## Architecture

### Components

1. **TETLMEAnalyzer**: Fits LME models and extracts results
2. **TETContrastAnalyzer**: Computes dose contrasts within states
3. **Main Script**: Orchestrates analysis and exports results

### Data Flow

```
Preprocessed Data (tet_preprocessed.csv)
    ↓
Filter to 0-9 minutes (t_bin 0-18)
    ↓
Center time variable
    ↓
TETLMEAnalyzer
    ├→ For each dimension:
    │   ├→ Fit LME model
    │   ├→ Extract coefficients, CIs, p-values
    │   └→ Store results
    ├→ Apply FDR correction per effect
    └→ Export lme_results.csv
    
TETContrastAnalyzer
    ├→ Compute High vs Low dose contrast in DMT
    ├→ Compute High vs Low dose contrast in RS
    └→ Export lme_contrasts.csv
```

## Component Design

### 1. TETLMEAnalyzer

**Purpose**: Fit LME models and extract statistical results.

**Input**:
- Preprocessed TET data (DataFrame)
- List of dimensions to analyze (z-scored dimensions only)

**Processing**:

1. **Data Preparation**:
   - Filter to t_bin 0-18 (0-9 minutes)
   - Center time variable: `time_c = t_bin - mean(t_bin)`
   - Set reference levels: State='RS', Dose='Baja'

2. **Model Fitting** (for each dimension):
   - Fit LME using `statsmodels.formula.api.mixedlm`
   - Formula: `dimension ~ State + Dose + time_c + State:Dose + State:time_c + Dose:time_c`
   - Random effects: `groups=Subject`, `re_formula='1'`
   - Method: Maximum Likelihood (ML)

3. **Results Extraction**:
   - Coefficients (beta)
   - 95% Confidence intervals
   - P-values
   - Model diagnostics (AIC, BIC, log-likelihood)

4. **FDR Correction**:
   - Apply Benjamini-Hochberg FDR correction
   - Separate correction for each fixed effect
   - Family size: 15 dimensions (z-scored dimensions only, not composite indices)

**Output**:
- DataFrame with columns: `dimension`, `effect`, `beta`, `ci_lower`, `ci_upper`, `p_value`, `p_fdr`, `significant`

**Key Methods**:
- `prepare_data()`: Filter and center time
- `fit_lme(dimension)`: Fit model for one dimension
- `extract_results(model)`: Extract coefficients and statistics
- `apply_fdr_correction()`: Apply BH-FDR correction
- `fit_all_dimensions()`: Fit models for all dimensions
- `export_results(output_dir)`: Save results to CSV

### 2. TETContrastAnalyzer

**Purpose**: Compute dose contrasts within each state.

**Input**:
- Fitted LME models (from TETLMEAnalyzer)
- Preprocessed data

**Processing**:

For each dimension:

1. **DMT High vs Low Contrast**:
   - Estimate: β(Dose=Alta) + β(State:Dose)
   - This gives the dose effect specifically in DMT state
   - Compute SE and p-value using delta method

2. **RS High vs Low Contrast**:
   - Estimate: β(Dose=Alta)
   - This gives the dose effect specifically in RS state
   - SE and p-value from model

**Output**:
- DataFrame with columns: `dimension`, `contrast`, `estimate`, `se`, `ci_lower`, `ci_upper`, `p_value`, `p_fdr`

**Key Methods**:
- `compute_contrast(model, contrast_type)`: Compute one contrast
- `compute_all_contrasts()`: Compute contrasts for all dimensions
- `export_contrasts(output_dir)`: Save results to CSV

## Implementation Details

### Data Preparation

```python
# Filter to 0-9 minutes
data_lme = data[data['t_bin'] <= 18].copy()

# Center time
data_lme['time_c'] = data_lme['t_bin'] - data_lme['t_bin'].mean()

# Set reference levels
data_lme['state'] = pd.Categorical(data_lme['state'], categories=['RS', 'DMT'])
data_lme['dose'] = pd.Categorical(data_lme['dose'], categories=['Baja', 'Alta'])
```

### Model Fitting

```python
import statsmodels.formula.api as smf

# Fit LME model
formula = f"{dimension} ~ state + dose + time_c + state:dose + state:time_c + dose:time_c"
model = smf.mixedlm(
    formula=formula,
    data=data_lme,
    groups=data_lme['subject'],
    re_formula='1'  # Random intercept only
)
result = model.fit(method='ml')  # Maximum likelihood
```

### Results Extraction

```python
# Extract coefficients
params = result.params
conf_int = result.conf_int()
pvalues = result.pvalues

# Create results DataFrame
results = pd.DataFrame({
    'effect': params.index,
    'beta': params.values,
    'ci_lower': conf_int[0].values,
    'ci_upper': conf_int[1].values,
    'p_value': pvalues.values
})
```

### FDR Correction

```python
from statsmodels.stats.multitest import multipletests

# For each effect separately
for effect in effects:
    effect_results = results[results['effect'] == effect]
    _, p_fdr, _, _ = multipletests(
        effect_results['p_value'],
        alpha=0.05,
        method='fdr_bh'
    )
    results.loc[results['effect'] == effect, 'p_fdr'] = p_fdr
```

### Dose Contrasts

```python
# DMT High vs Low: beta_dose + beta_state:dose
dmt_contrast = result.params['dose[T.Alta]'] + result.params['state[T.DMT]:dose[T.Alta]']

# Compute SE using variance-covariance matrix
vcov = result.cov_params()
var_contrast = (
    vcov.loc['dose[T.Alta]', 'dose[T.Alta]'] +
    vcov.loc['state[T.DMT]:dose[T.Alta]', 'state[T.DMT]:dose[T.Alta]'] +
    2 * vcov.loc['dose[T.Alta]', 'state[T.DMT]:dose[T.Alta]']
)
se_contrast = np.sqrt(var_contrast)

# Compute p-value (Wald test)
z_stat = dmt_contrast / se_contrast
p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
```

## Data Schema

### LME Results Output

| Column | Type | Description |
|--------|------|-------------|
| dimension | str | Dimension name (z-scored) |
| effect | str | Fixed effect name |
| beta | float | Coefficient estimate |
| ci_lower | float | Lower 95% CI |
| ci_upper | float | Upper 95% CI |
| p_value | float | Uncorrected p-value |
| p_fdr | float | FDR-corrected p-value |
| significant | bool | p_fdr < 0.05 |

### Contrasts Output

| Column | Type | Description |
|--------|------|-------------|
| dimension | str | Dimension name |
| contrast | str | Contrast description |
| estimate | float | Contrast estimate |
| se | float | Standard error |
| ci_lower | float | Lower 95% CI |
| ci_upper | float | Upper 95% CI |
| p_value | float | Uncorrected p-value |
| p_fdr | float | FDR-corrected p-value |

## Fixed Effects

The model includes the following fixed effects:

1. **Intercept**: Baseline (RS, Baja dose, mean time)
2. **state[T.DMT]**: Main effect of DMT vs RS
3. **dose[T.Alta]**: Main effect of high vs low dose
4. **time_c**: Linear time trend
5. **state[T.DMT]:dose[T.Alta]**: State × Dose interaction
6. **state[T.DMT]:time_c**: State × Time interaction
7. **dose[T.Alta]:time_c**: Dose × Time interaction

## Dimensions to Analyze

**Z-scored Dimensions Only (15)**:
- pleasantness_z, unpleasantness_z, emotional_intensity_z
- elementary_imagery_z, complex_imagery_z
- auditory_z, interoception_z
- bliss_z, anxiety_z, entity_z
- selfhood_z, disembodiment_z
- salience_z, temporality_z, general_intensity_z

**Note**: Composite indices are NOT included in LME analysis (they are derived from the dimensions above).

## Error Handling

1. **Convergence Issues**: Log warning and skip dimension if model fails to converge
2. **Singular Matrix**: Handle cases where variance-covariance matrix is singular
3. **Missing Data**: Ensure complete cases for model fitting

## Output Files

```
results/tet/lme/
├── lme_results.csv           # Main LME results with FDR correction
├── lme_contrasts.csv         # Dose contrasts within states
└── lme_model_diagnostics.csv # Model fit statistics (AIC, BIC, etc.)
```

## Testing Strategy

1. **Unit Tests**:
   - Test data preparation (filtering, centering)
   - Test model fitting with synthetic data
   - Test FDR correction
   - Test contrast computation

2. **Integration Tests**:
   - Test complete pipeline with real data
   - Verify output file structure
   - Verify all dimensions processed

3. **Validation**:
   - Compare results with R lme4 package (if available)
   - Verify FDR correction: p_fdr >= p_value
   - Verify contrasts: DMT dose effect = beta_dose + beta_interaction

## Performance Considerations

- Model fitting can be slow (~1-2 seconds per dimension)
- Total time: ~15-30 seconds for 15 dimensions
- Consider parallel processing if needed
- Cache fitted models for contrast computation

## Dependencies

- **statsmodels**: LME model fitting
- **scipy**: Statistical functions
- **pandas**: Data manipulation
- **numpy**: Numerical computations

## Usage Example

```python
from tet.lme_analyzer import TETLMEAnalyzer
from tet.contrast_analyzer import TETContrastAnalyzer
import pandas as pd

# Load preprocessed data
data = pd.read_csv('results/tet/preprocessed/tet_preprocessed.csv')

# Fit LME models
lme_analyzer = TETLMEAnalyzer(data)
lme_results = lme_analyzer.fit_all_dimensions()
lme_analyzer.export_results('results/tet/lme')

# Compute contrasts
contrast_analyzer = TETContrastAnalyzer(lme_analyzer.models, data)
contrasts = contrast_analyzer.compute_all_contrasts()
contrast_analyzer.export_contrasts('results/tet/lme')
```

## Notes

- LME models account for repeated measures within subjects
- FDR correction controls for multiple comparisons across dimensions
- Contrasts allow interpretation of dose effects within each state
- Time is centered to improve model interpretability and convergence
- Maximum likelihood (ML) is used for parameter estimation
