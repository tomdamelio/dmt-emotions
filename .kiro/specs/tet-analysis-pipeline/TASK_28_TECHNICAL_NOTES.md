# Task 28: Technical Notes on Subject-Level Permutation Testing

## Methodological Rationale

### Why Subject-Level Permutation?

Traditional permutation tests that shuffle individual observations violate the independence assumption when data have hierarchical structure (observations nested within subjects). Subject-level permutation testing addresses this by:

1. **Preserving Within-Subject Dependencies**: Each subject's temporal sequence remains intact
2. **Breaking Cross-Subject Coupling**: Randomly pairs subjects' physiological and TET data
3. **Accounting for Subject-Level Variance**: Null distribution reflects true sampling variability

### Derangement Algorithm

The implementation ensures no subject is paired with themselves (i ≠ j) using a derangement algorithm:

```python
# Simple approach: shuffle until valid
while not valid_permutation:
    np.random.shuffle(permuted_subjects)
    if not np.any(unique_subjects == permuted_subjects):
        valid_permutation = True

# Fallback: swap-based derangement
for i in range(n_subjects):
    if unique_subjects[i] == permuted_subjects[i]:
        # Find another position to swap with
        for j in range(i + 1, n_subjects):
            if unique_subjects[j] != permuted_subjects[i] and \
               unique_subjects[i] != permuted_subjects[j]:
                permuted_subjects[i], permuted_subjects[j] = \
                    permuted_subjects[j], permuted_subjects[i]
                break
```

This ensures the null hypothesis (no physiological-affective coupling) is properly tested.

## Statistical Properties

### Empirical P-Value Formula

```
p = (count of r_perm ≥ r_obs + 1) / (n_permutations + 1)
```

The +1 in numerator and denominator accounts for the observed value being part of the permutation distribution (Phipson & Smyth, 2010).

### One-Tailed vs Two-Tailed

The implementation uses a **one-tailed test** because:
- We have a directional hypothesis (positive physiological-affective coupling)
- CCA canonical correlations are always positive by construction
- We test whether observed r is larger than expected by chance

### Multiple Comparison Correction

Permutation p-values can be corrected for multiple comparisons using:
- Benjamini-Hochberg FDR (across canonical variates)
- Bonferroni correction (conservative)
- Max-T permutation (family-wise error rate)

Currently, no correction is applied within the permutation test itself, but p-values can be corrected downstream.

## Computational Considerations

### Memory Efficiency

The implementation stores only:
- Final permuted correlations (n_permutations × n_components)
- Summary statistics (observed r, p-values)

Full permutation distributions are recomputed for export/visualization to save memory during computation.

### Parallelization Potential

The permutation loop is embarrassingly parallel:
```python
# Current: sequential
for perm_idx in range(n_permutations):
    permuted_corrs[perm_idx] = self._fit_permuted_cca(...)

# Future: parallel (using joblib or multiprocessing)
from joblib import Parallel, delayed
permuted_corrs = Parallel(n_jobs=-1)(
    delayed(self._fit_permuted_cca)(...) 
    for perm_idx in range(n_permutations)
)
```

This could reduce runtime from ~15 min to ~2-3 min for n=1000 on multi-core systems.

### Random Seed Management

Each permutation uses a unique seed:
```python
perm_seed = random_state + perm_idx if random_state is not None else None
```

This ensures:
- Reproducibility when random_state is set
- Different permutations when random_state is None
- Consistent results across runs

## Validation Strategy

### Synthetic Data Tests

The test suite uses synthetic data with known structure:
- Subject-specific offsets create between-subject variance
- Correlated physio-TET relationships create signal
- Random noise creates realistic variability

Expected behavior:
- Observed r should be higher than most permuted r
- P-values should be < 0.05 for strong correlations
- P-values should be > 0.05 for weak correlations

### Real Data Validation

When testing with real data, check:
1. **Observed r > mean(permuted r)**: Signal exists
2. **P-value < 0.05**: Signal is significant
3. **Permutation distribution is smooth**: Sufficient permutations
4. **No outliers in permutation distribution**: No bugs in shuffling

## Comparison with Parametric Tests

### Wilks' Lambda Test (Parametric)

The standard CCA significance test uses Wilks' Lambda:
```
Λ = ∏(1 - r_i²)
χ² = -n * ln(Λ)
p = P(χ² > observed | df = p × q)
```

**Assumptions**:
- Multivariate normality
- Independent observations
- Homoscedasticity

### Permutation Test (Non-Parametric)

**Advantages**:
- No distributional assumptions
- Accounts for subject-level dependencies
- Robust to outliers
- Exact test (not asymptotic)

**Disadvantages**:
- Computationally intensive
- Requires sufficient permutations (≥1000)
- May be conservative with small samples

## Interpretation Guidelines

### P-Value Thresholds

- **p < 0.001**: Very strong evidence of coupling
- **p < 0.01**: Strong evidence of coupling
- **p < 0.05**: Moderate evidence of coupling
- **p > 0.05**: Insufficient evidence of coupling

### Effect Size Interpretation

Canonical correlations (r):
- **r > 0.7**: Strong coupling
- **0.5 < r < 0.7**: Moderate coupling
- **0.3 < r < 0.5**: Weak coupling
- **r < 0.3**: Negligible coupling

### Null Distribution Inspection

Visual inspection of permutation distributions reveals:
- **Narrow distribution**: Low variability, precise estimate
- **Wide distribution**: High variability, uncertain estimate
- **Observed r in tail**: Significant result
- **Observed r in center**: Non-significant result

## References

1. Phipson, B., & Smyth, G. K. (2010). Permutation P-values should never be zero: calculating exact P-values when permutations are randomly drawn. *Statistical Applications in Genetics and Molecular Biology*, 9(1).

2. Winkler, A. M., et al. (2014). Permutation inference for the general linear model. *NeuroImage*, 92, 381-397.

3. Nichols, T. E., & Holmes, A. P. (2002). Nonparametric permutation tests for functional neuroimaging: a primer with examples. *Human Brain Mapping*, 15(1), 1-25.

4. Anderson, M. J., & Robinson, J. (2001). Permutation tests for linear models. *Australian & New Zealand Journal of Statistics*, 43(1), 75-88.

## Implementation Notes

### Code Quality
- ✅ Type hints for all parameters
- ✅ Comprehensive docstrings
- ✅ Error handling for edge cases
- ✅ Logging for progress tracking
- ✅ Validation of inputs

### Testing
- ✅ Unit tests for shuffling algorithm
- ✅ Integration tests for full pipeline
- ✅ Synthetic data validation
- ⏳ Real data validation (pending)

### Documentation
- ✅ Method docstrings with examples
- ✅ Usage instructions in main script
- ✅ Technical notes (this document)
- ⏳ User guide section (pending)

## Future Enhancements

1. **Parallel Processing**: Use joblib for multi-core execution
2. **Adaptive Permutations**: Stop early if p-value is clearly significant/non-significant
3. **Stratified Permutation**: Preserve state (RS/DMT) during shuffling
4. **Block Permutation**: Shuffle entire sessions instead of subjects
5. **Cluster-Based Correction**: Extend to multiple canonical variates with cluster correction
