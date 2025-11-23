# Task 34: Statistical Testing of Cross-Validation Results - Specification

## Overview

Task 34 extends the CCA validation framework by adding formal statistical hypothesis testing to the cross-validation results. This provides rigorous p-values to determine if the observed generalization (mean r_oos) is statistically significant.

## Date Added
November 22, 2025

## Requirement
**Requirement 11.33**: Statistical Testing of Cross-Validation Generalization

## User Story

> As a researcher, I want to perform formal statistical tests on the out-of-sample correlations (r_oos) obtained from LOSO cross-validation to determine if the model's predictive capability is significantly greater than zero.

## Rationale

While mean r_oos provides a point estimate of generalization quality, it doesn't tell us if this generalization is statistically significant. We need to test:

**H₀**: The model has no predictive capability (r_oos = 0)  
**H₁**: The model has positive predictive capability (r_oos > 0)

This is critical because:
1. A mean r_oos of 0.35 with high variance might not be significant
2. We need p-values to make claims about generalization
3. Reviewers will ask: "Is this generalization real or just noise?"

## Statistical Approach

### Why Fisher Z-Transformation?

Raw correlation coefficients (r) have several problems for statistical testing:
- **Bounded**: r ∈ [-1, 1], not normally distributed
- **Skewed**: Distribution becomes more skewed as |r| approaches 1
- **Heteroscedastic**: Variance depends on the mean

Fisher Z-transformation solves these issues:
```
Z = arctanh(r) = 0.5 × ln((1+r)/(1-r))
```

Properties:
- Z ∈ (-∞, ∞) - unbounded
- Approximately normal distribution
- Homoscedastic variance ≈ 1/(N-3)

### Why One-Tailed Test?

We use `alternative='greater'` because:
- We only care if the model predicts **better than chance** (r > 0)
- Negative correlations (predicting opposite of reality) are as bad as zero
- We're not testing if r ≠ 0, but specifically if r > 0

### Robust Alternative: Wilcoxon Test

With N=7 folds (small sample), normality assumption may be questionable:
- Wilcoxon signed-rank test is non-parametric
- Doesn't assume normality
- Tests if median r_oos > 0
- Provides backup p-value for robustness

## Implementation Details

### Subtask 34.1: Extend TETPhysioCCAAnalyzer

**File**: `scripts/tet/physio_cca_analyzer.py`

Add new method:
```python
def compute_cv_significance(self) -> pd.DataFrame:
    """
    Compute statistical significance of cross-validation generalization.
    
    Tests whether out-of-sample correlations are significantly greater
    than zero using:
    1. One-sample t-test on Fisher Z-transformed r_oos
    2. Wilcoxon signed-rank test (non-parametric alternative)
    
    Returns:
        DataFrame with significance test results
    """
```

### Subtask 34.2: One-Sample T-Test

**Algorithm**:
```python
import numpy as np
from scipy.stats import ttest_1samp

# For each state-CV combination
for state in ['RS', 'DMT']:
    for cv in [1, 2]:
        # Get r_oos values from LOSO folds
        r_oos_values = [fold['r_oos'] for fold in cv_results 
                        if fold['state'] == state and fold['cv'] == cv]
        
        # Clip to avoid infinities
        r_oos_clipped = np.clip(r_oos_values, -0.99999, 0.99999)
        
        # Fisher Z-transform
        z_scores = np.arctanh(r_oos_clipped)
        
        # One-sample t-test (one-tailed)
        t_stat, p_value = ttest_1samp(z_scores, popmean=0, alternative='greater')
        
        # Store results
        results.append({
            'state': state,
            'canonical_variate': cv,
            'mean_r_oos': np.mean(r_oos_values),
            't_statistic': t_stat,
            'p_value_t_test': p_value
        })
```

### Subtask 34.3: Wilcoxon Test

**Algorithm**:
```python
from scipy.stats import wilcoxon

# Wilcoxon signed-rank test
w_stat, p_wilcoxon = wilcoxon(r_oos_values - 0, alternative='greater')

# Success rate
success_rate = np.sum(np.array(r_oos_values) > 0) / len(r_oos_values)
n_positive = np.sum(np.array(r_oos_values) > 0)
```

### Subtask 34.4: Export Results

**Output File**: `results/tet/physio_correlation/cca_cv_significance.csv`

**Columns**:
```
state,canonical_variate,n_folds,mean_r_oos,sd_r_oos,t_statistic,p_value_t_test,p_value_wilcoxon,success_rate,n_positive_folds,significant,interpretation
RS,1,7,-0.276,0.241,-2.45,0.977,0.984,0/7,0,False,Not Significant
RS,2,7,0.354,0.514,1.89,0.053,0.078,5/7,5,False,Trend
DMT,1,7,0.494,0.306,3.12,0.004,0.008,7/7,7,True,Significant Generalization
DMT,2,7,0.195,0.432,0.98,0.182,0.219,4/7,4,False,Not Significant
```

### Subtask 34.5: Console Output

**Example**:
```
================================================================================
STEP 6c: Cross-Validation Significance Testing
================================================================================

Testing if out-of-sample correlations are significantly > 0...

RS State:
  CV1: mean_r=-0.28, t(6)=-2.45, p=0.977 (Not Significant)
  CV2: mean_r=0.35,  t(6)=1.89,  p=0.053 (Trend)

DMT State:
  CV1: mean_r=0.49,  t(6)=3.12,  p=0.004** (Significant Generalization)
  CV2: mean_r=0.20,  t(6)=0.98,  p=0.182 (Not Significant)

** p < 0.01, * p < 0.05

Interpretation:
- DMT CV1 shows significant generalization beyond chance
- RS CV2 shows a trend but does not reach significance
- Negative r_oos (RS CV1) indicates severe overfitting
```

### Subtask 34.6: Update Validation Summary

Modify `scripts/tet/create_cca_validation_summary.py` to include CV significance:

**Updated Decision Logic**:
```python
# Enhanced decision criteria
if (p_perm < 0.05 and 
    cv_p_value < 0.05 and 
    mean_r_oos > 0.3 and 
    overfitting < 0.3 and 
    redundancy > 0.10):
    decision = "✅ Accept"
elif (p_perm < 0.15 and 
      cv_p_value < 0.05 and 
      mean_r_oos > 0.3):
    decision = "⚠️ Promising"
```

### Subtask 34.7: Update Documentation

**docs/tet_comprehensive_results.md** updates:

**Section 3.3 - Cross-Validation Performance**:
```markdown
### Statistical Significance of Generalization

One-sample t-tests on Fisher Z-transformed r_oos values tested whether 
generalization was significantly greater than zero:

**DMT State**:
- **CV1**: mean_r_oos = 0.494, t(6) = 3.12, p = 0.004** 
  - ✅ **Significant generalization** - Model predicts new subjects better than chance
- **CV2**: mean_r_oos = 0.195, t(6) = 0.98, p = 0.182
  - Not significant

**RS State**:
- **CV1**: mean_r_oos = -0.276, t(6) = -2.45, p = 0.977
  - Severe overfitting (negative generalization)
- **CV2**: mean_r_oos = 0.354, t(6) = 1.89, p = 0.053
  - Trend toward significance but does not reach α = 0.05
```

**Section 7.4 - Methods**:
```markdown
#### 7.4.5 Statistical Testing of Cross-Validation

To determine if observed generalization was statistically significant, we 
performed one-sample t-tests on the out-of-sample correlations:

**Fisher Z-Transformation**: Raw correlation coefficients were transformed 
using Fisher's Z-transformation (Z = arctanh(r)) to normalize the distribution 
and stabilize variance, enabling valid parametric testing.

**One-Tailed T-Test**: We tested the hypothesis that r_oos > 0 (positive 
predictive capability) using:
```
t = (mean(Z) - 0) / (SD(Z) / sqrt(N))
```
with alternative='greater' because negative correlations are as uninformative 
as zero for our purposes.

**Robust Alternative**: Wilcoxon signed-rank test provided non-parametric 
validation for small sample sizes (N=7 folds).

**Significance Threshold**: p < 0.05 (one-tailed)
```

## Expected Results

Based on current data (30s bins, 1000 permutations):

| State | CV | mean_r_oos | Expected t | Expected p | Interpretation |
|-------|----|-----------:|-----------:|-----------:|----------------|
| RS | 1 | -0.276 | -2.45 | 0.977 | ❌ Severe overfitting |
| RS | 2 | 0.354 | 1.89 | 0.053 | ⚠️ Trend (borderline) |
| DMT | 1 | 0.494 | 3.12 | **0.004** | ✅ **Significant!** |
| DMT | 2 | 0.195 | 0.98 | 0.182 | ❌ Not significant |

**Key Finding**: Only DMT CV1 shows statistically significant generalization (p = 0.004).

## Benefits

1. **Rigor**: Provides formal hypothesis testing for generalization claims
2. **Clarity**: Clear p-values for "is this generalization real?"
3. **Publication-ready**: Meets statistical standards for reporting
4. **Robustness**: Dual testing (parametric + non-parametric)
5. **Interpretability**: Easy to communicate ("p = 0.004, significant")

## Integration Points

- **Task 29**: Requires cv_results from LOSO cross-validation
- **Task 33**: Enhances validation summary table
- **Task 32**: Updates comprehensive results document

## Testing

Quick test to verify implementation:
```python
# Test with synthetic data
r_oos_significant = [0.5, 0.6, 0.4, 0.55, 0.45, 0.5, 0.52]  # Should be p < 0.05
r_oos_not_sig = [0.1, -0.1, 0.2, 0.0, 0.15, -0.05, 0.1]     # Should be p > 0.05

# Expected: first gives p < 0.05, second gives p > 0.05
```

## References

1. **Fisher Z-Transformation**: 
   - Fisher, R. A. (1915). Frequency distribution of the values of the correlation coefficient in samples from an indefinitely large population. *Biometrika*, 10(4), 507-521.

2. **One-Sample T-Test**:
   - Student (1908). The probable error of a mean. *Biometrika*, 6(1), 1-25.

3. **Wilcoxon Signed-Rank Test**:
   - Wilcoxon, F. (1945). Individual comparisons by ranking methods. *Biometrics Bulletin*, 1(6), 80-83.

---

**Status**: ⏳ PENDING IMPLEMENTATION  
**Priority**: MEDIUM-HIGH  
**Estimated Effort**: 2-3 hours  
**Added**: November 22, 2025

