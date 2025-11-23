# CCA Validation Methods

## Overview

This document describes the validation methods used to assess the robustness and generalizability of Canonical Correlation Analysis (CCA) results in the physiological-TET integration analysis. These methods address critical concerns about overfitting, temporal autocorrelation, and subject-level dependencies that can inflate canonical correlations and lead to spurious findings.

## Table of Contents

1. [Why Validation is Critical](#why-validation-is-critical)
2. [Subject-Level Permutation Testing](#subject-level-permutation-testing)
3. [Leave-One-Subject-Out Cross-Validation](#leave-one-subject-out-cross-validation)
4. [Redundancy Index](#redundancy-index)
5. [Decision Criteria](#decision-criteria)
6. [References](#references)

---

## Why Validation is Critical

### The Problem with Naive CCA

Canonical Correlation Analysis identifies linear combinations of two variable sets that maximize their correlation. However, several factors can lead to artificially inflated canonical correlations:

1. **Temporal Autocorrelation**: Consecutive time points within a subject are not independent. Using raw time series data (e.g., 0.25 Hz sampling) creates thousands of pseudo-replicated observations, inflating sample size and canonical correlations.

2. **Subject-Level Dependencies**: Observations from the same subject are more similar to each other than to observations from different subjects. Standard CCA treats all observations as independent, violating this hierarchical structure.

3. **Overfitting**: With many variables (3 physiological + 6 TET dimensions), CCA can find spurious correlations that don't generalize to new data, especially with limited subjects.

4. **Multiple Comparisons**: Testing multiple canonical variates increases the risk of false positives.

### Why Row Shuffling is Inappropriate

A common but **incorrect** approach is to shuffle rows of the data matrix randomly:

```python
# ❌ WRONG: Row shuffling breaks temporal structure
permuted_indices = np.random.permutation(len(data))
X_perm = X[permuted_indices]
Y_perm = Y[permuted_indices]
```

**Why this is wrong:**
- Breaks within-subject temporal structure
- Destroys autocorrelation patterns that are part of the data
- Creates artificially independent observations
- Leads to overly conservative p-values
- Doesn't test the right null hypothesis

### The Correct Null Hypothesis

The appropriate null hypothesis for physiological-TET coupling is:

> **H₀**: There is no systematic relationship between physiological signals and affective TET dimensions **across subjects**, while preserving within-subject temporal structure.

This requires **subject-level permutation testing**.

---

## Subject-Level Permutation Testing

### Rationale

Subject-level permutation testing validates CCA significance by randomly pairing subjects' physiological and TET data while preserving within-subject temporal structure. This approach:

- Maintains temporal autocorrelation within each subject
- Tests cross-subject coupling (the scientifically meaningful hypothesis)
- Respects the hierarchical data structure
- Provides robust significance testing

### Algorithm

For each permutation iteration:

1. **Identify unique subjects**: Get list of all subjects in the dataset
2. **Create derangement**: Generate a permutation where no subject maps to itself (i ≠ j)
3. **Pair subjects**: For each subject i, pair their physiological data with subject j's TET data
4. **Preserve temporal order**: Keep all time points in their original sequence within each subject
5. **Fit CCA**: Compute canonical correlations on the permuted data
6. **Store results**: Save permuted canonical correlations

After N permutations (typically 1000):

7. **Compute empirical p-value**: 
   ```
   p = (count of r_perm ≥ r_observed + 1) / (N + 1)
   ```

### Implementation Details

```python
def _subject_level_shuffle(X, Y, subject_ids, random_state=None):
    """
    Shuffle subject pairings while preserving temporal structure.
    
    Args:
        X: Physiological matrix (n_obs × 3)
        Y: TET affective matrix (n_obs × 6)
        subject_ids: Subject identifier for each observation
        random_state: Random seed for reproducibility
    
    Returns:
        X_perm: Original X (unchanged)
        Y_perm: Shuffled Y with preserved temporal structure
    """
    unique_subjects = np.unique(subject_ids)
    
    # Create derangement (no subject maps to itself)
    permuted_subjects = create_derangement(unique_subjects)
    
    # Create mapping dictionary
    subject_mapping = dict(zip(unique_subjects, permuted_subjects))
    
    # Apply permutation to Y
    Y_perm = np.zeros_like(Y)
    for subject in unique_subjects:
        subject_mask = subject_ids == subject
        paired_subject = subject_mapping[subject]
        paired_mask = subject_ids == paired_subject
        
        # Copy paired subject's Y data (preserves temporal order)
        Y_perm[subject_mask] = Y[paired_mask]
    
    return X, Y_perm
```

### Interpretation

- **p < 0.05**: Canonical correlation is statistically significant
- **p ≥ 0.05**: No evidence for systematic physiological-affective coupling
- **p < 0.001**: Strong evidence for robust coupling

### Computational Considerations

- **Debugging**: Start with 100 permutations (~2 minutes) to catch bugs quickly
- **Publication**: Use 1000 permutations (~15 minutes) for final results
- **Reproducibility**: Always set `random_state` for reproducible results

---

## Leave-One-Subject-Out Cross-Validation

### Rationale

LOSO cross-validation assesses whether CCA canonical correlations generalize to new subjects. This addresses overfitting concerns by:

- Training on N-1 subjects
- Testing on the held-out subject
- Comparing in-sample vs out-of-sample correlations
- Quantifying generalization performance

### Algorithm

For each subject k:

1. **Split data**: 
   - Training set: All subjects except k
   - Test set: Subject k only

2. **Train CCA**: Fit CCA on training data
   ```python
   cca_train = CCA(n_components=2)
   cca_train.fit(X_train, Y_train)
   ```

3. **Extract weights**: Get canonical weights W_x and W_y

4. **Handle sign indeterminacy**: Align signs with global (full-sample) weights
   ```python
   # Sign flipping is fundamental to CCA
   # Align with global weights to avoid spurious sign changes
   for i in range(n_components):
       if np.dot(W_x_train[:, i], W_x_global[:, i]) < 0:
           W_x_train[:, i] *= -1
           W_y_train[:, i] *= -1
   ```

5. **Transform test data**: Apply weights to held-out subject
   ```python
   U_test = X_test @ W_x_train
   V_test = Y_test @ W_y_train
   ```

6. **Compute out-of-sample correlation**:
   ```python
   r_oos = np.corrcoef(U_test[:, i], V_test[:, i])[0, 1]
   ```

7. **Handle low variance**: Flag folds with insufficient variance as invalid

### Sign Indeterminacy

**Critical technical note**: CCA canonical weights are determined only up to sign. The weights (W_x, W_y) and (-W_x, -W_y) produce identical canonical correlations. This creates a problem in cross-validation:

- Different folds may flip signs arbitrarily
- Without alignment, correlations can be spuriously negative
- Must align signs with a reference (global weights)

**Solution**: Align each fold's weights with the global (full-sample) weights by checking the dot product and flipping if negative.

### Variance Filtering

Some subjects may have insufficient variance in their data, leading to unstable correlations. We flag folds as invalid if:

- Standard deviation of canonical variate < 0.1
- Correlation computation fails (NaN)
- Extreme outliers detected

Invalid folds are excluded from summary statistics.

### Summary Statistics

For each canonical variate, compute:

1. **Mean r_oos**: Average out-of-sample correlation
   - Use Fisher Z-transformation for proper averaging:
     ```python
     z_values = np.arctanh(r_oos_valid)
     mean_z = np.mean(z_values)
     mean_r_oos = np.tanh(mean_z)
     ```

2. **SD r_oos**: Standard deviation of out-of-sample correlations

3. **Overfitting index**: 
   ```
   overfitting_index = (r_in_sample - mean_r_oos) / r_in_sample
   ```

4. **Valid folds**: Number of subjects with valid correlations

### Interpretation

- **mean_r_oos > 0.3**: Moderate generalization (acceptable)
- **mean_r_oos > 0.5**: Strong generalization (excellent)
- **mean_r_oos < 0.2**: Weak generalization (concerning)
- **overfitting_index < 0.3**: Acceptable overfitting
- **overfitting_index > 0.5**: Severe overfitting (model doesn't generalize)

---

## Redundancy Index

### Rationale

Canonical correlations can be high even when the shared variance between variable sets is minimal. The redundancy index quantifies the **percentage of variance** in one variable set explained by the canonical variates from the other set.

### Formula

For canonical variate pair i:

**Redundancy of Y given X:**
```
Redundancy(Y|X) = r_c² × R²(Y|U_i)
```

**Redundancy of X given Y:**
```
Redundancy(X|Y) = r_c² × R²(X|V_i)
```

Where:
- `r_c` = canonical correlation for variate i
- `R²(Y|U_i)` = average R² from regressing each TET dimension on physiological canonical variate U_i
- `R²(X|V_i)` = average R² from regressing each physiological measure on TET canonical variate V_i
- `U_i = X @ W_x` (physiological canonical variate)
- `V_i = Y @ W_y` (TET canonical variate)

### Computation

1. **Transform to canonical variates**:
   ```python
   U, V = cca_model.transform(X, Y)
   ```

2. **For each canonical variate i**:
   
   a. Compute variance explained in TET by physio variate:
   ```python
   r_squared_values = []
   for j in range(n_tet_dims):
       r = np.corrcoef(Y[:, j], U[:, i])[0, 1]
       r_squared_values.append(r**2)
   var_explained_Y = np.mean(r_squared_values)
   ```
   
   b. Compute variance explained in physio by TET variate:
   ```python
   r_squared_values = []
   for k in range(n_physio_measures):
       r = np.corrcoef(X[:, k], V[:, i])[0, 1]
       r_squared_values.append(r**2)
   var_explained_X = np.mean(r_squared_values)
   ```
   
   c. Compute redundancy indices:
   ```python
   redundancy_Y_given_X = (r_c**2) * var_explained_Y
   redundancy_X_given_Y = (r_c**2) * var_explained_X
   ```

3. **Total redundancy**: Sum across all canonical variates

### Interpretation

- **Redundancy > 10%**: Meaningful shared variance (acceptable)
- **Redundancy 5-10%**: Weak relationship (borderline)
- **Redundancy < 5%**: Minimal shared variance (potential overfitting)

**Example interpretation:**

```
Canonical Variate 1:
- r_canonical = 0.65
- Redundancy(TET|Physio) = 18.2%
- Redundancy(Physio|TET) = 12.4%

Interpretation: Physiological signals explain 18.2% of variance in 
affective TET dimensions, indicating meaningful autonomic-affective coupling.
```

---

## Decision Criteria

### When to Trust CCA Results

Use the following decision tree to determine whether CCA results represent robust physiological-affective coupling:

#### 1. Permutation Test

**Question**: Is the canonical correlation statistically significant?

- ✅ **p < 0.05**: Proceed to next check
- ❌ **p ≥ 0.05**: **STOP** - No evidence for systematic coupling

#### 2. Cross-Validation

**Question**: Does the model generalize to new subjects?

- ✅ **mean_r_oos > 0.3 AND overfitting_index < 0.3**: Proceed to next check
- ⚠️ **mean_r_oos 0.2-0.3 OR overfitting_index 0.3-0.5**: Weak generalization (interpret cautiously)
- ❌ **mean_r_oos < 0.2 OR overfitting_index > 0.5**: **STOP** - Severe overfitting

#### 3. Redundancy Index

**Question**: Is there meaningful shared variance?

- ✅ **Redundancy > 10%**: **ACCEPT** - Robust physiological-affective coupling
- ⚠️ **Redundancy 5-10%**: Weak coupling (interpret cautiously)
- ❌ **Redundancy < 5%**: **REJECT** - Minimal shared variance (likely overfitting)

### Decision Matrix

| Permutation p | mean_r_oos | Overfitting | Redundancy | Decision |
|---------------|------------|-------------|------------|----------|
| < 0.05 | > 0.3 | < 0.3 | > 10% | ✅ **Accept** - Robust coupling |
| < 0.05 | > 0.3 | < 0.3 | 5-10% | ⚠️ **Caution** - Weak coupling |
| < 0.05 | 0.2-0.3 | 0.3-0.5 | > 10% | ⚠️ **Caution** - Borderline generalization |
| < 0.05 | < 0.2 | > 0.5 | Any | ❌ **Reject** - Overfitting |
| ≥ 0.05 | Any | Any | Any | ❌ **Reject** - Not significant |

### Example Interpretations

#### Example 1: Robust Coupling (DMT State)

```
Canonical Variate 1:
- Permutation p-value: 0.002
- mean_r_oos: 0.52 (SD: 0.18)
- Overfitting index: 0.20
- Redundancy (TET|Physio): 18.2%

Decision: ✅ ACCEPT
Interpretation: Strong evidence for robust physiological-affective 
coupling during DMT. The relationship generalizes well to new subjects 
and explains meaningful shared variance.
```

#### Example 2: Overfitting (RS State)

```
Canonical Variate 1:
- Permutation p-value: 0.048
- mean_r_oos: 0.15 (SD: 0.32)
- Overfitting index: 0.68
- Redundancy (TET|Physio): 4.2%

Decision: ❌ REJECT
Interpretation: Although nominally significant, the model shows severe 
overfitting (overfitting index = 0.68) and minimal shared variance 
(redundancy = 4.2%). Results likely reflect noise rather than true coupling.
```

#### Example 3: Weak Coupling (Borderline)

```
Canonical Variate 2:
- Permutation p-value: 0.012
- mean_r_oos: 0.28 (SD: 0.24)
- Overfitting index: 0.35
- Redundancy (TET|Physio): 7.8%

Decision: ⚠️ INTERPRET WITH CAUTION
Interpretation: Statistically significant but weak generalization and 
borderline shared variance. May represent a real but weak effect. 
Requires replication in independent sample.
```

---

## References

### Statistical Methods

1. **Canonical Correlation Analysis**:
   - Hotelling, H. (1936). Relations between two sets of variates. *Biometrika*, 28(3/4), 321-377.
   - Thompson, B. (1984). *Canonical correlation analysis: Uses and interpretation*. Sage Publications.

2. **Permutation Testing**:
   - Nichols, T. E., & Holmes, A. P. (2002). Nonparametric permutation tests for functional neuroimaging: A primer with examples. *Human Brain Mapping*, 15(1), 1-25.
   - Winkler, A. M., et al. (2014). Permutation inference for the general linear model. *NeuroImage*, 92, 381-397.

3. **Cross-Validation**:
   - Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The elements of statistical learning* (2nd ed.). Springer.
   - Varoquaux, G., et al. (2017). Assessing and tuning brain decoders: Cross-validation, caveats, and guidelines. *NeuroImage*, 145, 166-179.

4. **Redundancy Index**:
   - Stewart, D., & Love, W. (1968). A general canonical correlation index. *Psychological Bulletin*, 70(3), 160-163.
   - Lambert, Z. V., Wildt, A. R., & Durand, R. M. (1991). Approximating confidence intervals for factor loadings. *Multivariate Behavioral Research*, 26(3), 421-434.

### Temporal Autocorrelation

5. **Autocorrelation in Neuroimaging**:
   - Friston, K. J., et al. (1994). Statistical parametric maps in functional imaging: A general linear approach. *Human Brain Mapping*, 2(4), 189-210.
   - Woolrich, M. W., et al. (2001). Temporal autocorrelation in univariate linear modeling of FMRI data. *NeuroImage*, 14(6), 1370-1386.

6. **Subject-Level Dependencies**:
   - Chen, G., et al. (2013). Handling multiplicity in neuroimaging through Bayesian lenses with multilevel modeling. *Neuroinformatics*, 11(4), 483-493.
   - Mumford, J. A., & Nichols, T. (2009). Simple group fMRI modeling and inference. *NeuroImage*, 47(4), 1469-1475.

### Psychedelic Research Applications

7. **Physiological-Subjective Coupling**:
   - Carhart-Harris, R. L., et al. (2016). Neural correlates of the LSD experience revealed by multimodal neuroimaging. *PNAS*, 113(17), 4853-4858.
   - Tagliazucchi, E., et al. (2016). Enhanced repertoire of brain dynamical states during the psychedelic experience. *Human Brain Mapping*, 37(11), 3985-4000.

---

## Implementation Notes

### File Locations

- **Validator**: `scripts/tet/cca_data_validator.py`
- **CCA Analyzer**: `scripts/tet/physio_cca_analyzer.py`
- **Main Script**: `scripts/compute_physio_correlation.py`
- **Test Scripts**: `test/tet/test_cca_*.py`

### Usage Example

```python
from scripts.tet.physio_data_loader import TETPhysioDataLoader
from scripts.tet.cca_data_validator import CCADataValidator
from scripts.tet.physio_cca_analyzer import TETPhysioCCAAnalyzer

# Load and validate data
loader = TETPhysioDataLoader()
merged_data = loader.load_and_merge()

validator = CCADataValidator(merged_data)
validator.validate_temporal_resolution()
validator.validate_sample_size()
report = validator.generate_validation_report()

# Fit CCA
analyzer = TETPhysioCCAAnalyzer(merged_data)
analyzer.fit_cca(n_components=2)

# Validation analyses
perm_results = analyzer.permutation_test(n_permutations=1000)
cv_results = analyzer.loso_cross_validation('DMT')
redundancy_df = analyzer.compute_redundancy_index('DMT')

# Export results
analyzer.export_results('results/tet/physio_correlation')
```

### Computational Time

- **Data validation**: < 1 second
- **CCA fitting**: < 1 second
- **Permutation test (100 iterations)**: ~2 minutes
- **Permutation test (1000 iterations)**: ~15 minutes
- **LOSO cross-validation**: ~30 seconds
- **Redundancy index**: < 1 second

---

## Troubleshooting

### Common Issues

1. **All permutation p-values = 1.0**
   - **Cause**: Subject shuffling not working (subjects mapping to themselves)
   - **Solution**: Check derangement algorithm, ensure i ≠ j for all pairs

2. **Negative out-of-sample correlations**
   - **Cause**: Sign indeterminacy not handled
   - **Solution**: Align fold weights with global weights

3. **Many invalid CV folds**
   - **Cause**: Insufficient data per subject or low variance
   - **Solution**: Check data completeness, consider aggregating to longer time windows

4. **Redundancy index > 50%**
   - **Cause**: Likely computational error or perfect multicollinearity
   - **Solution**: Check for duplicate variables, verify standardization

### Debugging Checklist

- [ ] Data aggregated to 30-second bins (not raw 0.25 Hz)
- [ ] Sample size ≥ 100 observations
- [ ] Subject IDs correctly aligned across modalities
- [ ] No missing values in key variables
- [ ] Permutation creates valid derangement (i ≠ j)
- [ ] Sign alignment applied in cross-validation
- [ ] Fisher Z-transformation used for averaging correlations
- [ ] Random seed set for reproducibility

---

## Contact

For questions about CCA validation methods, contact the analysis team or refer to the comprehensive results document at `docs/tet_comprehensive_results.md`.

**Last updated**: 2024-01-XX


---

## Detailed Interpretation Guide

### Step-by-Step Interpretation Workflow

When interpreting CCA validation results, follow this systematic workflow:

#### Step 1: Check Permutation Test Results

**Load permutation results:**
```python
perm_df = pd.read_csv('results/tet/physio_correlation/cca_permutation_pvalues.csv')
print(perm_df)
```

**Expected output:**
```
   state  canonical_variate  observed_r  permutation_p_value  n_permutations
0     RS                  1       0.423                0.156            1000
1     RS                  2       0.287                0.342            1000
2    DMT                  1       0.651                0.002            1000
3    DMT                  2       0.412                0.028            1000
```

**Interpretation:**
- **DMT CV1**: p = 0.002 → **Significant** (strong evidence for coupling)
- **DMT CV2**: p = 0.028 → **Significant** (moderate evidence)
- **RS CV1**: p = 0.156 → **Not significant** (no evidence for coupling)
- **RS CV2**: p = 0.342 → **Not significant**

**Decision**: Proceed with DMT analysis only. RS shows no significant coupling.

---

#### Step 2: Examine Cross-Validation Performance

**Load CV summary:**
```python
cv_summary = pd.read_csv('results/tet/physio_correlation/cca_cross_validation_summary.csv')
print(cv_summary)
```

**Expected output:**
```
   state  canonical_variate  mean_r_oos  sd_r_oos  in_sample_r  overfitting_index  n_valid_folds
0    DMT                  1       0.521     0.182        0.651              0.200             18
1    DMT                  2       0.285     0.241        0.412              0.308             17
```

**Interpretation:**

**DMT CV1:**
- mean_r_oos = 0.521 → **Strong generalization** (> 0.5)
- overfitting_index = 0.200 → **Acceptable overfitting** (< 0.3)
- n_valid_folds = 18/19 → **Good data quality**
- **Conclusion**: Model generalizes well to new subjects

**DMT CV2:**
- mean_r_oos = 0.285 → **Weak generalization** (< 0.3)
- overfitting_index = 0.308 → **Borderline overfitting** (~0.3)
- n_valid_folds = 17/19 → **Acceptable data quality**
- **Conclusion**: Weak generalization, interpret cautiously

**Decision**: CV1 is robust, CV2 is borderline.

---

#### Step 3: Assess Redundancy Index

**Load redundancy results:**
```python
redundancy_df = pd.read_csv('results/tet/physio_correlation/cca_redundancy_indices.csv')
print(redundancy_df)
```

**Expected output:**
```
   state  canonical_variate  r_canonical  redundancy_Y_given_X  redundancy_X_given_Y
0    DMT                  1        0.651                 0.182                 0.124
1    DMT                  2        0.412                 0.068                 0.045
2    DMT              Total          NaN                 0.250                 0.169
```

**Interpretation:**

**DMT CV1:**
- Redundancy(TET|Physio) = 18.2% → **Meaningful shared variance** (> 10%)
- Redundancy(Physio|TET) = 12.4% → **Meaningful shared variance**
- **Conclusion**: Physiological signals explain 18.2% of variance in affective TET dimensions

**DMT CV2:**
- Redundancy(TET|Physio) = 6.8% → **Weak shared variance** (< 10%)
- Redundancy(Physio|TET) = 4.5% → **Weak shared variance**
- **Conclusion**: Minimal shared variance, likely noise

**Total Redundancy:**
- Total(TET|Physio) = 25.0% → **Substantial shared variance**
- Total(Physio|TET) = 16.9% → **Moderate shared variance**

**Decision**: CV1 shows meaningful coupling, CV2 does not.

---

#### Step 4: Synthesize Evidence

**Create evidence table:**

| Criterion | DMT CV1 | DMT CV2 | Decision |
|-----------|---------|---------|----------|
| Permutation p | 0.002 ✅ | 0.028 ✅ | Both significant |
| mean_r_oos | 0.521 ✅ | 0.285 ⚠️ | CV1 strong, CV2 weak |
| Overfitting | 0.200 ✅ | 0.308 ⚠️ | CV1 good, CV2 borderline |
| Redundancy | 18.2% ✅ | 6.8% ❌ | CV1 meaningful, CV2 weak |
| **Overall** | **✅ ACCEPT** | **⚠️ CAUTION** | CV1 robust, CV2 questionable |

**Final interpretation:**

> **Canonical Variate 1 (DMT)**: Strong evidence for robust physiological-affective coupling. The relationship is statistically significant (p = 0.002), generalizes well to new subjects (mean r_oos = 0.52), shows acceptable overfitting (20%), and explains meaningful shared variance (18.2%). This represents a genuine autonomic-affective coupling during the psychedelic state.

> **Canonical Variate 2 (DMT)**: Statistically significant (p = 0.028) but weak generalization (mean r_oos = 0.29), borderline overfitting (31%), and minimal shared variance (6.8%). This likely represents noise or a very weak effect. Interpret with caution and do not emphasize in main results.

---

### Threshold Definitions

#### Permutation P-Value

| Range | Interpretation | Action |
|-------|----------------|--------|
| p < 0.001 | Very strong evidence | ✅ Highly confident |
| p < 0.01 | Strong evidence | ✅ Confident |
| p < 0.05 | Moderate evidence | ✅ Accept with standard confidence |
| p < 0.10 | Weak evidence | ⚠️ Trend, interpret cautiously |
| p ≥ 0.10 | No evidence | ❌ Reject |

#### Out-of-Sample Correlation (mean_r_oos)

| Range | Interpretation | Action |
|-------|----------------|--------|
| r > 0.5 | Strong generalization | ✅ Excellent |
| r > 0.3 | Moderate generalization | ✅ Acceptable |
| r > 0.2 | Weak generalization | ⚠️ Borderline |
| r ≤ 0.2 | Poor generalization | ❌ Reject |

#### Overfitting Index

| Range | Interpretation | Action |
|-------|----------------|--------|
| < 0.2 | Minimal overfitting | ✅ Excellent |
| < 0.3 | Acceptable overfitting | ✅ Good |
| 0.3-0.5 | Moderate overfitting | ⚠️ Borderline |
| > 0.5 | Severe overfitting | ❌ Reject |

**Formula:**
```
overfitting_index = (r_in_sample - mean_r_oos) / r_in_sample
```

**Interpretation:**
- 0.0 = Perfect generalization (no overfitting)
- 0.2 = 20% drop from in-sample to out-of-sample (acceptable)
- 0.5 = 50% drop (severe overfitting)
- 1.0 = Complete failure to generalize

#### Redundancy Index

| Range | Interpretation | Action |
|-------|----------------|--------|
| > 20% | Strong shared variance | ✅ Excellent |
| > 10% | Meaningful shared variance | ✅ Acceptable |
| 5-10% | Weak shared variance | ⚠️ Borderline |
| < 5% | Minimal shared variance | ❌ Likely noise |

**Context:**
- In social sciences, redundancy > 10% is considered meaningful
- In neuroscience, redundancy > 15% is considered strong
- Total redundancy (sum across all CVs) provides overall coupling strength

---

### Common Interpretation Scenarios

#### Scenario A: Ideal Results

```
Permutation p: 0.001
mean_r_oos: 0.58
Overfitting: 0.15
Redundancy: 22.4%
```

**Interpretation**: ✅ **Excellent** - All criteria met. Strong evidence for robust physiological-affective coupling. Suitable for main results and emphasis in discussion.

**Reporting**: "Canonical correlation analysis revealed strong physiological-affective coupling (r = 0.68, permutation p = 0.001), which generalized well to new subjects (mean r_oos = 0.58, overfitting index = 0.15) and explained substantial shared variance (redundancy = 22.4%)."

---

#### Scenario B: Significant but Weak Generalization

```
Permutation p: 0.032
mean_r_oos: 0.24
Overfitting: 0.42
Redundancy: 7.2%
```

**Interpretation**: ⚠️ **Caution** - Statistically significant but poor generalization and weak shared variance. Likely overfitting or very weak effect.

**Reporting**: "Although canonical correlation reached statistical significance (p = 0.032), the model showed poor generalization to new subjects (mean r_oos = 0.24, overfitting index = 0.42) and minimal shared variance (redundancy = 7.2%), suggesting limited practical significance."

**Action**: Report in supplementary materials, do not emphasize in main text.

---

#### Scenario C: Strong Generalization but Not Significant

```
Permutation p: 0.082
mean_r_oos: 0.45
Overfitting: 0.18
Redundancy: 14.8%
```

**Interpretation**: ⚠️ **Trend** - Good generalization and meaningful shared variance, but not statistically significant. May represent a real but weak effect requiring larger sample.

**Reporting**: "Canonical correlation showed a trend toward significance (p = 0.082) with moderate generalization (mean r_oos = 0.45) and meaningful shared variance (redundancy = 14.8%), suggesting a potential effect that requires replication in a larger sample."

**Action**: Report as exploratory finding, recommend replication.

---

#### Scenario D: Significant but Contradictory Metrics

```
Permutation p: 0.018
mean_r_oos: 0.52
Overfitting: 0.22
Redundancy: 4.1%
```

**Interpretation**: ⚠️ **Ambiguous** - Significant with good generalization, but minimal shared variance. Possible issues:
1. Redundancy computation error
2. Canonical correlation captures noise rather than meaningful variance
3. Relationship exists but is weak in practical terms

**Action**: 
1. Verify redundancy computation
2. Examine canonical loadings to understand what the variate represents
3. Check for outliers or data quality issues
4. Report with caution, emphasize low redundancy

---

### Reporting Guidelines

#### Main Text (for robust findings)

**Template:**
> "Canonical correlation analysis revealed [strength] physiological-affective coupling during [state] (r = [value], permutation p = [value]). The relationship generalized well to new subjects (mean r_oos = [value], overfitting index = [value]) and explained [percentage]% of variance in affective TET dimensions (redundancy index = [value]%)."

**Example:**
> "Canonical correlation analysis revealed strong physiological-affective coupling during DMT (r = 0.65, permutation p = 0.002). The relationship generalized well to new subjects (mean r_oos = 0.52, overfitting index = 0.20) and explained 18.2% of variance in affective TET dimensions (redundancy index = 18.2%)."

#### Supplementary Materials (for borderline findings)

**Template:**
> "Canonical Variate [N] showed [significance level] (r = [value], permutation p = [value]) but [generalization quality] (mean r_oos = [value], overfitting index = [value]) and [shared variance quality] (redundancy = [value]%). This suggests [interpretation]."

**Example:**
> "Canonical Variate 2 showed statistical significance (r = 0.41, permutation p = 0.028) but weak generalization (mean r_oos = 0.29, overfitting index = 0.31) and minimal shared variance (redundancy = 6.8%). This suggests a weak or spurious relationship that should not be emphasized."

#### Methods Section

**Required information:**
1. Number of permutations (e.g., 1000)
2. Permutation strategy (subject-level shuffling)
3. Cross-validation approach (LOSO)
4. Sign alignment procedure
5. Redundancy index formula
6. Significance thresholds

**Example:**
> "CCA significance was assessed using subject-level permutation testing (1000 iterations), which preserves within-subject temporal structure while testing cross-subject coupling. Generalization was evaluated using leave-one-subject-out cross-validation with sign alignment to handle canonical weight indeterminacy. Redundancy indices quantified the percentage of variance in each variable set explained by the canonical variates. We considered results robust if permutation p < 0.05, mean out-of-sample correlation > 0.3, overfitting index < 0.3, and redundancy > 10%."

---

### Visualization Interpretation

#### Permutation Null Distribution Plot

**What to look for:**
1. **Observed correlation position**: Should be in the far right tail
2. **Separation from null**: Clear gap between observed and permuted values
3. **Null distribution shape**: Should be approximately normal
4. **Rejection region**: Observed value should exceed 95th percentile

**Red flags:**
- Observed value near center of null distribution (not significant)
- Bimodal null distribution (permutation error)
- Very wide null distribution (high variability, unstable)

#### Cross-Validation Distribution Plot

**What to look for:**
1. **Mean r_oos**: Should be positive and substantial (> 0.3)
2. **Variability**: Narrow distribution indicates stable generalization
3. **Outliers**: Few extreme negative values acceptable
4. **Comparison to in-sample**: Should be lower but not drastically

**Red flags:**
- Many negative r_oos values (sign flipping error)
- Bimodal distribution (subgroups with different patterns)
- Mean r_oos near zero (no generalization)

#### Redundancy Index Bar Chart

**What to look for:**
1. **CV1 redundancy**: Should be highest (> 10%)
2. **CV2 redundancy**: Typically lower than CV1
3. **Bidirectional redundancy**: Both directions should be similar
4. **Total redundancy**: Sum should be meaningful (> 15%)

**Red flags:**
- All redundancy values < 5% (no meaningful coupling)
- Huge asymmetry (e.g., 30% one direction, 2% other) (computational error)
- Redundancy > 50% (perfect multicollinearity or error)

---

### Troubleshooting Interpretation Issues

#### Issue 1: Significant permutation test but poor CV performance

**Possible causes:**
1. Overfitting to specific subjects
2. Outlier subjects driving the effect
3. Heterogeneous subject population

**Solutions:**
1. Examine per-subject CV results
2. Check for outliers in canonical variates
3. Consider subgroup analyses (e.g., high vs low responders)

#### Issue 2: Good CV performance but low redundancy

**Possible causes:**
1. Canonical correlation captures noise rather than meaningful variance
2. Redundancy computation error
3. Variables have low communality

**Solutions:**
1. Verify redundancy computation
2. Examine canonical loadings (which variables contribute?)
3. Check variable correlations within each set

#### Issue 3: Inconsistent results across states (RS vs DMT)

**Possible causes:**
1. True state-dependent effect (expected)
2. Different sample sizes or data quality
3. Different variance structures

**Solutions:**
1. Compare sample sizes and completeness rates
2. Check variance of canonical variates in each state
3. Examine state-specific canonical loadings

#### Issue 4: All CV folds invalid

**Possible causes:**
1. Insufficient data per subject
2. Low variance in canonical variates
3. Sign flipping error

**Solutions:**
1. Check observations per subject (need > 10)
2. Examine variance of canonical variates
3. Verify sign alignment code

---

### Decision Tree Flowchart

```
START
  |
  v
Is permutation p < 0.05?
  |
  +-- NO --> REJECT (not significant)
  |
  +-- YES
      |
      v
    Is mean_r_oos > 0.3?
      |
      +-- NO --> Is mean_r_oos > 0.2?
      |           |
      |           +-- NO --> REJECT (poor generalization)
      |           |
      |           +-- YES --> CAUTION (weak generalization)
      |
      +-- YES
          |
          v
        Is overfitting_index < 0.3?
          |
          +-- NO --> Is overfitting_index < 0.5?
          |           |
          |           +-- NO --> REJECT (severe overfitting)
          |           |
          |           +-- YES --> CAUTION (moderate overfitting)
          |
          +-- YES
              |
              v
            Is redundancy > 10%?
              |
              +-- NO --> Is redundancy > 5%?
              |           |
              |           +-- NO --> REJECT (minimal shared variance)
              |           |
              |           +-- YES --> CAUTION (weak shared variance)
              |
              +-- YES --> ACCEPT (robust coupling)
```

---

### Summary Checklist

Before finalizing CCA interpretation, verify:

- [ ] Permutation test performed with subject-level shuffling (not row shuffling)
- [ ] Number of permutations ≥ 1000 for publication
- [ ] Cross-validation used LOSO approach
- [ ] Sign alignment applied in CV
- [ ] Fisher Z-transformation used for averaging correlations
- [ ] Redundancy index computed for all canonical variates
- [ ] All three validation criteria checked (permutation, CV, redundancy)
- [ ] Decision criteria applied systematically
- [ ] Borderline results flagged for caution
- [ ] Robust results clearly distinguished from weak results
- [ ] Methods section includes all validation procedures
- [ ] Results section reports all validation metrics
- [ ] Discussion acknowledges limitations of borderline findings

---

## Quick Reference Card

### Accept CCA Results If:
✅ Permutation p < 0.05  
✅ mean_r_oos > 0.3  
✅ Overfitting index < 0.3  
✅ Redundancy > 10%

### Interpret with Caution If:
⚠️ Permutation p < 0.10 (trend)  
⚠️ mean_r_oos 0.2-0.3 (weak generalization)  
⚠️ Overfitting index 0.3-0.5 (moderate overfitting)  
⚠️ Redundancy 5-10% (weak shared variance)

### Reject CCA Results If:
❌ Permutation p ≥ 0.10  
❌ mean_r_oos < 0.2  
❌ Overfitting index > 0.5  
❌ Redundancy < 5%

### Key Formulas:
```
Overfitting Index = (r_in_sample - mean_r_oos) / r_in_sample
Redundancy(Y|X) = r_c² × R²(Y|U)
Permutation p = (count of r_perm ≥ r_obs + 1) / (N + 1)
```

---

**End of Interpretation Guide**
