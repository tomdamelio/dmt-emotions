# Task 29: LOSO Cross-Validation - Technical Notes

## Critical Implementation Details

### 1. Sign Indeterminacy in CCA

#### Problem
Canonical Correlation Analysis produces canonical variates with **arbitrary signs**. For the same data, different runs or different train/test splits can produce canonical weights with opposite signs, even though they represent the same underlying relationship.

#### Why This Matters for LOSO CV
When averaging out-of-sample correlations across folds, sign inconsistency can cause **spurious cancellation**:
- Fold 1: r_oos = +0.8 (positive sign)
- Fold 2: r_oos = -0.8 (negative sign, but same relationship)
- Naive average: (0.8 + (-0.8)) / 2 = 0.0 ← WRONG!

#### Solution: Sign Flipping
Before computing OOS correlations, we align the signs of the fold-specific weights with the global (full-sample) weights:

```python
# For each canonical component
for i in range(n_components):
    # Check if fold weights align with global weights
    corr_x = np.corrcoef(W_x[:, i], W_x_global[:, i])[0, 1]
    corr_y = np.corrcoef(W_y[:, i], W_y_global[:, i])[0, 1]
    
    # If anti-correlated, flip signs
    if corr_x < 0:
        W_x[:, i] = -W_x[:, i]
    if corr_y < 0:
        W_y[:, i] = -W_y[:, i]
```

#### Verification
After sign flipping, all fold-specific weights should have positive correlation with global weights, ensuring consistent sign convention across folds.

---

### 2. Fisher Z-Transformation for Averaging Correlations

#### Problem
Correlation coefficients (r) have a **non-normal distribution**, especially when:
- True correlation is far from zero
- Sample size is small
- Correlations are near ±1

Simple averaging of correlations is **biased** and can produce incorrect estimates.

#### Solution: Fisher Z-Transformation
Transform correlations to Fisher z-scores before averaging:

```python
# Transform to Fisher z
z_values = np.arctanh(r_oos_valid)

# Compute mean in z-space
mean_z = np.mean(z_values)

# Transform back to correlation
mean_r_oos = np.tanh(mean_z)
```

#### Why This Works
- Fisher z has approximately normal distribution
- Variance is approximately constant across z-values
- Unbiased estimator of population correlation

#### Mathematical Justification
Fisher's z-transformation:
- z = arctanh(r) = 0.5 * ln((1 + r) / (1 - r))
- Inverse: r = tanh(z) = (e^(2z) - 1) / (e^(2z) + 1)

For large N, z ~ Normal(arctanh(ρ), 1/(N-3))
where ρ is the population correlation.

---

### 3. Low Variance Handling

#### Problem
Some subjects may have **zero or near-zero variance** in their canonical variates, especially for:
- Weaker canonical components (CV2, CV3, ...)
- Subjects with limited data
- Edge cases in data collection

#### Consequences
- Correlation computation fails (division by zero)
- Returns NaN or undefined values
- Can crash the analysis pipeline

#### Solution: Graceful Degradation
```python
# Check for zero/near-zero variance
if np.std(U_test[:, i]) < 1e-10 or np.std(V_test[:, i]) < 1e-10:
    r_oos[i] = np.nan
    continue

# Compute correlation with error handling
try:
    r_oos[i] = np.corrcoef(U_test[:, i], V_test[:, i])[0, 1]
    
    # Check for NaN
    if np.isnan(r_oos[i]):
        continue
        
except Exception:
    r_oos[i] = np.nan
```

#### Reporting
- Count valid vs excluded folds
- Log warnings for problematic subjects
- Exclude NaN values from averaging
- Report n_valid_folds and n_excluded_folds

---

### 4. Overfitting Index

#### Definition
```
overfitting_index = (in_sample_r - mean_r_oos) / in_sample_r
```

#### Interpretation
- **0.0**: Perfect generalization (OOS = in-sample)
- **< 0.0**: Better OOS than in-sample (unusual, suggests noise)
- **0.0 - 0.1**: Excellent generalization (< 10% drop)
- **0.1 - 0.3**: Good generalization (10-30% drop)
- **> 0.3**: Poor generalization (> 30% drop, potential overfitting)

#### Example
- In-sample r = 0.80
- Mean OOS r = 0.72
- Overfitting index = (0.80 - 0.72) / 0.80 = 0.10 (10% drop)
- Interpretation: Good generalization

#### Threshold
We use **10% (0.1)** as a reference threshold in the bar chart visualization. Values above this suggest the model may not generalize well.

---

### 5. LOSO vs K-Fold Cross-Validation

#### Why LOSO?
For subject-level data with temporal dependencies:
- **LOSO**: Leave-One-Subject-Out
  - Ensures complete separation of subjects
  - No data leakage from same subject
  - More conservative estimate
  - Appropriate for small N (< 50 subjects)

- **K-Fold**: Random splits
  - Can split same subject across folds
  - Violates independence assumption
  - Overly optimistic estimates
  - Not appropriate for hierarchical data

#### Trade-offs
- LOSO: Higher variance, lower bias
- K-Fold: Lower variance, higher bias (for subject data)

For physiological-TET data with ~20 subjects, LOSO is the appropriate choice.

---

### 6. Computational Complexity

#### Time Complexity
For N subjects, M observations per subject, K components:
- Fold generation: O(N)
- CCA fitting per fold: O(M² * K)
- Total: O(N * M² * K)

#### Example
- N = 20 subjects
- M = 18 observations per subject
- K = 2 components
- Estimated time: ~5-10 seconds

#### Optimization
- Vectorized operations (NumPy)
- Efficient matrix operations (sklearn CCA)
- Minimal data copying

---

### 7. Diagnostic Plots

#### Plot 1: Box Plots
- **Purpose**: Show distribution of OOS correlations
- **Interpretation**: 
  - Tight boxes → Consistent across subjects
  - Wide boxes → High variability
  - Outliers → Problematic subjects

#### Plot 2: Scatter Plot
- **Purpose**: Compare in-sample vs OOS
- **Interpretation**:
  - Points on identity line → Perfect generalization
  - Points below line → Overfitting
  - Points above line → Unusual (noise or model mismatch)

#### Plot 3: Overfitting Index
- **Purpose**: Quantify generalization gap
- **Interpretation**:
  - Bars below 0.1 → Good generalization
  - Bars above 0.1 → Potential overfitting
  - Negative bars → Better OOS (unusual)

---

### 8. Edge Cases and Error Handling

#### Case 1: All Folds Invalid
```python
if n_valid == 0:
    # Return NaN for all statistics
    summary_records.append({
        'mean_r_oos': np.nan,
        'sd_r_oos': np.nan,
        # ...
    })
```

#### Case 2: Single Valid Fold
```python
# Use ddof=1 for unbiased estimate
sd_r_oos = np.std(r_oos_valid, ddof=1) if n_valid > 1 else 0.0
```

#### Case 3: CCA Convergence Failure
```python
try:
    cca_train.fit(X_train, Y_train)
except Exception as e:
    self.logger.warning(f"Failed to fit CCA: {e}")
    return np.full(n_components, np.nan)
```

---

### 9. Validation Checklist

Before accepting LOSO CV results, verify:

- [ ] All subjects have at least one valid fold
- [ ] Sign flipping is working (no spurious cancellations)
- [ ] Fisher Z-transformation is applied
- [ ] Overfitting indices are reasonable (< 0.5)
- [ ] In-sample and OOS correlations are in same direction
- [ ] No systematic bias across folds
- [ ] Diagnostic plots look reasonable

---

### 10. Common Pitfalls

#### Pitfall 1: Forgetting Sign Flipping
**Symptom**: Mean OOS correlation near zero despite strong in-sample correlation
**Solution**: Implement sign alignment with global weights

#### Pitfall 2: Simple Averaging of Correlations
**Symptom**: Biased estimates, especially for strong correlations
**Solution**: Use Fisher Z-transformation

#### Pitfall 3: Not Handling Low Variance
**Symptom**: NaN errors, crashes, or invalid correlations
**Solution**: Check variance before computing correlations

#### Pitfall 4: Data Leakage
**Symptom**: Overly optimistic OOS correlations
**Solution**: Ensure complete subject separation in folds

#### Pitfall 5: Ignoring Invalid Folds
**Symptom**: Biased estimates from excluding problematic subjects
**Solution**: Report n_valid_folds and n_excluded_folds

---

### 11. Future Enhancements

#### Potential Improvements
1. **Stratified LOSO**: Balance folds by state/dose
2. **Nested CV**: Tune hyperparameters within CV loop
3. **Bootstrap CI**: Confidence intervals for OOS correlations
4. **Permutation Test**: Test if OOS > chance
5. **Subject-Level Metrics**: Identify problematic subjects

#### Research Questions
1. Does overfitting vary by state (RS vs DMT)?
2. Are certain canonical components more stable?
3. Do specific subjects drive the results?
4. How does sample size affect generalization?

---

## References

### Fisher Z-Transformation
- Fisher, R. A. (1915). "Frequency distribution of the values of the correlation coefficient in samples from an indefinitely large population". Biometrika, 10(4), 507-521.

### CCA Sign Indeterminacy
- Hotelling, H. (1936). "Relations between two sets of variates". Biometrika, 28(3/4), 321-377.

### Cross-Validation
- Stone, M. (1974). "Cross-validatory choice and assessment of statistical predictions". Journal of the Royal Statistical Society, 36(2), 111-147.

### LOSO for Hierarchical Data
- Varoquaux, G., et al. (2017). "Assessing and tuning brain decoders: Cross-validation, caveats, and guidelines". NeuroImage, 145, 166-179.

---

## Conclusion

The LOSO cross-validation implementation addresses three critical challenges:

1. **Sign indeterminacy**: Solved with sign flipping
2. **Non-normal distributions**: Solved with Fisher Z-transformation
3. **Low variance edge cases**: Solved with graceful degradation

These technical details ensure robust, unbiased estimates of CCA generalization performance.

---

**Document Version**: 1.0
**Date**: November 21, 2025
**Author**: TET Analysis Pipeline
