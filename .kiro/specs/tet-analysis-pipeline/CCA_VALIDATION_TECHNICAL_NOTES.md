# CCA Validation: Critical Technical Notes

## Overview

This document highlights three critical technical issues in CCA cross-validation that can cause spurious results if not handled properly. These are common pitfalls in multivariate analysis that must be addressed for publication-quality validation.

---

## A. Sign Indeterminacy Problem (Task 29.3)

### The Problem

**CCA canonical variates have arbitrary signs** - this is a fundamental mathematical property, not a bug.

**Example Scenario:**
- **Fold 1** (trained on 6 subjects): Finds "↑HR predicts ↑Anxiety" (r = +0.8)
- **Fold 2** (trained on different 6 subjects): Finds "↓HR predicts ↓Anxiety" (r = +0.8)
  - Mathematically equivalent relationship, but signs are flipped

**What Goes Wrong:**
- If you average the canonical weights or projections across folds without aligning signs
- The positive and negative versions cancel out: (+0.8 + -0.8) / 2 = 0
- Result: Appears like there's no relationship when there actually is!

### The Solution

**Before applying training weights to test data, align signs:**

```python
# For each canonical variate
for i in range(n_components):
    # Compare train weights to global weights
    sign_check = np.corrcoef(W_x_train[:, i], W_x_global[:, i])[0, 1]
    
    # If negative correlation, flip both weight vectors
    if sign_check < 0:
        W_x_train[:, i] *= -1
        W_y_train[:, i] *= -1
```

**Why This Works:**
- Aligns all folds to have consistent sign convention
- Uses global (full-sample) weights as reference
- Preserves the magnitude of relationships while fixing direction

**Implementation Location:**
- Task 29.3: `_compute_oos_correlation()` method
- Design document: Section 3.2

---

## B. Averaging Correlations Problem (Task 29.4)

### The Problem

**Correlation coefficients (r) are not normally distributed**, especially with small sample sizes.

**Mathematical Issue:**
- Correlations are bounded [-1, +1]
- Distribution is skewed near boundaries
- Simple arithmetic mean is biased

**Example:**
```
Fold correlations: [0.7, 0.8, 0.9]
Simple mean: 0.80
Fisher Z mean: 0.82 (more accurate)
```

### The Solution

**Use Fisher Z-transformation for averaging:**

```python
# 1. Remove NaN values (failed folds)
r_valid = r_oos_array[~np.isnan(r_oos_array)]

# 2. Transform to z-space (unbounded, normally distributed)
z_values = np.arctanh(r_valid)

# 3. Compute statistics in z-space
mean_z = np.mean(z_values)
sd_z = np.std(z_values)

# 4. Transform back to correlation space
mean_r_oos = np.tanh(mean_z)
```

**Why This Works:**
- Fisher Z-transformation: z = arctanh(r) = 0.5 * ln((1+r)/(1-r))
- Converts bounded correlation to unbounded z
- z is approximately normally distributed
- Statistically more rigorous than simple averaging

**Acceptable Alternative:**
- Simple averaging: `mean_r_oos = np.nanmean(r_oos_array)`
- Acceptable for first pass, but less rigorous
- Recommend Fisher Z for publication

**Implementation Location:**
- Task 29.4: `_summarize_cv_results()` method
- Design document: Section 3.3

---

## C. Low Variance / NaN Handling (Task 29.3)

### The Problem

**Single-subject test sets may have zero or near-zero variance** in some variables.

**Example Scenario:**
- Test subject's respiratory rate was extremely stable during that session
- Variance ≈ 0 in RVT dimension
- Correlation calculation: r = cov(X,Y) / (std(X) * std(Y))
- Division by zero → NaN or error

**What Goes Wrong:**
- Correlation computation fails
- Code crashes or returns NaN
- If not handled, contaminates all downstream statistics

### The Solution

**Wrap correlation computation in error handling:**

```python
def _compute_oos_correlation(self, X_test, Y_test, W_x, W_y, subject_id):
    try:
        # Transform test data
        U_test = X_test @ W_x
        V_test = Y_test @ W_y
        
        # Compute correlation
        r_oos = np.corrcoef(U_test.T, V_test.T)[0, 1]
        
        # Check for NaN (can happen with zero variance)
        if np.isnan(r_oos):
            logger.warning(f"NaN correlation for subject {subject_id} - low variance")
            return np.nan
            
        return r_oos
        
    except (ValueError, RuntimeWarning) as e:
        logger.warning(f"Correlation failed for subject {subject_id}: {e}")
        return np.nan
```

**In summarization (Task 29.4):**

```python
# Count valid folds
n_valid_folds = np.sum(~np.isnan(r_oos_array))
n_excluded_folds = np.sum(np.isnan(r_oos_array))

# Only average valid folds
r_valid = r_oos_array[~np.isnan(r_oos_array)]

# Report both counts
results['n_valid_folds'] = n_valid_folds
results['n_excluded_folds'] = n_excluded_folds
```

**Interpretation:**
- If n_valid_folds < 70% of total subjects → concerning
- May indicate data quality issues
- Report in validation summary

**Implementation Location:**
- Task 29.3: `_compute_oos_correlation()` method
- Task 29.4: `_summarize_cv_results()` method
- Design document: Sections 3.2 and 3.3

---

## D. Temporal Resolution Validation (Task 27.2)

### Confirmation: Use 30-Second Bins

**Critical Decision:** The spec correctly enforces 30-second bins.

**Why This Matters:**

**Raw data (0.25 Hz, 4-second bins):**
- N ≈ 1350 points per subject
- Extreme temporal autocorrelation (consecutive points nearly identical)
- Inflates canonical correlations artificially
- Violates independence assumption
- p-values are spurious

**Aggregated data (30-second bins):**
- N ≈ 126 points per subject (7 subjects × 18 bins)
- Reduced autocorrelation
- More appropriate for statistical inference
- Aligns with phenomenological resolution

**Validation Check (Task 27.2):**
```python
# Compute modal time difference
time_diffs = df.groupby(['subject', 'session'])['t_sec'].diff()
modal_diff = time_diffs.mode()[0]

if modal_diff < 20:  # Less than 20 seconds
    raise ValueError(
        f"Data appears to be raw (modal diff = {modal_diff}s). "
        "Must use 30-second bins to avoid inflated N."
    )
```

**Expected Values:**
- Modal time difference: 30 seconds (±2 seconds tolerance)
- Bins per 9-minute session: 18
- Total observations: ~504 (7 subjects × 4 sessions × 18 bins)

---

## Execution Strategy: Efficient Implementation Order

### ⚡ Phase 1: Data Validation First (Task 27) 🛑 STOP POINT

**Execute Task 27 completely and review validation report before proceeding.**

**Why this matters:**
- Validation takes ~1 minute
- Permutations take ~10-15 minutes
- If data validation fails (wrong resolution, missing subjects), permutations are wasted computation
- Catch issues early!

**Action:**
1. Implement and run all Task 27 subtasks
2. Review `data_validation_report.txt` carefully
3. **STOP if validation fails** - fix data issues first
4. Only proceed to Task 28 if validation passes ✓

### ⚡ Phase 2: Start Permutations with n=100 (Task 28)

**Use n_permutations=100 for initial debugging, then scale to 1000.**

**Why this matters:**
- n=100: ~1-2 minutes (fast iteration)
- n=1000: ~10-15 minutes (publication quality)
- Catch code bugs in 2 minutes instead of 15 minutes

**Action:**
1. Set default: `n_permutations=100` in method signature
2. Implement Task 28 and test with n=100
3. Verify null distributions look reasonable
4. **After validation**, change to n=1000 for final results

**Code pattern:**
```python
def permutation_test(self, n_permutations: int = 100, random_state: int = 42):
    """
    Perform subject-level permutation test.
    
    Args:
        n_permutations: Number of permutations
            - Use 100 for debugging (fast)
            - Use 1000 for publication (rigorous)
    """
```

### ⚡ Phase 3: Cross-Validation and Redundancy (Tasks 29-30)

**These are fast (~5 seconds each), proceed normally.**

---

## Summary: Implementation Checklist

When implementing CCA validation (Tasks 27-30), ensure:

### Execution Order
- [ ] **Task 27 FIRST**: Data validation → Review report → Fix issues
- [ ] **Task 28 with n=100**: Permutation test → Verify → Scale to n=1000
- [ ] **Task 29**: LOSO cross-validation
- [ ] **Task 30**: Redundancy indices

### Task 27.2 (Temporal Resolution)
- [ ] Validate modal time difference = 30 seconds
- [ ] Verify bins per session = 18
- [ ] Raise error if raw data detected (modal diff ≈ 4s)
- [ ] Document actual N in validation report
- [ ] **STOP and review before proceeding to Task 28**

### Task 28.1 (Permutation Setup)
- [ ] Set default n_permutations=100 (not 1000)
- [ ] Add parameter documentation about debugging vs publication
- [ ] Test with n=100 first
- [ ] Scale to n=1000 only after validation

### Task 29.3 (Out-of-Sample Correlation)
- [ ] Implement sign alignment before applying weights to test data
- [ ] Compare train weights to global weights
- [ ] Flip signs if correlation is negative
- [ ] Wrap correlation computation in try-except
- [ ] Return NaN for failed folds (low variance)
- [ ] Log warnings with subject IDs

### Task 29.4 (CV Statistics)
- [ ] Use Fisher Z-transformation for averaging (recommended)
- [ ] Alternative: Simple averaging with np.nanmean (acceptable)
- [ ] Count valid vs excluded folds
- [ ] Report both counts in results
- [ ] Flag if n_valid < 70% of subjects

---

## References

1. **Sign Indeterminacy in Multivariate Methods:**
   - Bro, R., Kjeldahl, K., Smilde, A. K., & Kiers, H. A. (2008). Cross-validation of component models: A critical look at current methods. *Analytical and Bioanalytical Chemistry*, 390(5), 1241-1251.

2. **Fisher Z-Transformation:**
   - Fisher, R. A. (1915). Frequency distribution of the values of the correlation coefficient in samples from an indefinitely large population. *Biometrika*, 10(4), 507-521.
   - Silver, N. C., & Dunlap, W. P. (1987). Averaging correlation coefficients: Should Fisher's z transformation be used? *Journal of Applied Psychology*, 72(1), 146.

3. **CCA Cross-Validation:**
   - Bilenko, N. Y., & Gallant, J. L. (2016). Pyrcca: Regularized kernel canonical correlation analysis in Python and its applications to neuroimaging. *Frontiers in Neuroinformatics*, 10, 49.

4. **Temporal Autocorrelation in Psychophysiology:**
   - Ebner-Priemer, U. W., & Sawitzki, G. (2007). Ambulatory assessment of affective instability in borderline personality disorder. *European Journal of Psychological Assessment*, 23(4), 238-247.

---

## Contact

For questions about these technical details, refer to:
- Tasks document: `.kiro/specs/tet-analysis-pipeline/tasks.md` (Tasks 27-30)
- Design document: `.kiro/specs/tet-analysis-pipeline/design_req11.md` (Sections 3.2-3.3)
- Requirements: `.kiro/specs/tet-analysis-pipeline/requirements.md` (11.23-11.32)
