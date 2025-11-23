# Task 30: Redundancy Index - Technical Implementation Notes

## Mathematical Foundation

### Redundancy Index Formula

The redundancy index quantifies the proportion of variance in one variable set explained by canonical variates from another set.

**For TET variables (Y) given Physiological variables (X):**
```
Redundancy(Y|X) = r_c² × R²(Y|U)
```

Where:
- `r_c` = canonical correlation between U and V
- `U` = canonical variate for X (physio): U = X @ W_x
- `V` = canonical variate for Y (TET): V = Y @ W_y
- `R²(Y|U)` = average R² from regressing each y_j on U

**For Physiological variables (X) given TET variables (Y):**
```
Redundancy(X|Y) = r_c² × R²(X|V)
```

### Variance Explained Computation

For a canonical variate U and variable set Y:

```
R²(Y|U) = (1/p) × Σ[j=1 to p] r²(y_j, U)
```

Where:
- p = number of variables in Y
- r(y_j, U) = Pearson correlation between variable y_j and canonical variate U

**Simplification**: For simple linear regression, R² = r²

## Implementation Details

### Method: `compute_redundancy_index(state)`

**Input**:
- `state`: 'RS' or 'DMT'

**Process**:
1. Retrieve fitted CCA model for the state
2. Prepare standardized matrices X (physio) and Y (TET)
3. Transform to canonical variates: U, V = cca.transform(X, Y)
4. For each canonical variate pair (U_i, V_i):
   - Compute R²(Y|U_i) using `_compute_variance_explained(Y, U_i)`
   - Compute R²(X|V_i) using `_compute_variance_explained(X, V_i)`
   - Compute redundancy: r_c² × R²
5. Sum redundancy across all variates for total redundancy

**Output**: DataFrame with per-variate and total redundancy

### Method: `_compute_variance_explained(Y, U)`

**Input**:
- `Y`: Variable matrix (n_obs × n_vars)
- `U`: Canonical variate (n_obs,)

**Process**:
1. For each variable y_j in Y:
   - Compute r = corr(y_j, U)
   - Compute R² = r²
2. Average R² across all variables

**Output**: Scalar average R²

### Method: `plot_redundancy_indices(output_dir)`

**Visualization Design**:
- **Chart Type**: Grouped bar chart
- **X-axis**: Canonical variates (CV1, CV2, ...)
- **Y-axis**: Redundancy index (%)
- **Bars**:
  - Blue: Physio → TET (redundancy_Y_given_X)
  - Red: TET → Physio (redundancy_X_given_Y)
- **Reference Line**: Horizontal line at 10% (meaningful threshold)
- **Annotations**: Exact percentage values on each bar
- **Layout**: Separate panels for RS and DMT states

**Output**: PNG file at `{output_dir}/cca_redundancy_indices.png`

### Method: `_interpret_redundancy(row)`

**Interpretation Thresholds**:
```python
if redundancy > 0.15:    # 15%
    return 'High'
elif redundancy > 0.10:  # 10%
    return 'Moderate'
elif redundancy > 0.05:  # 5%
    return 'Low'
else:
    return 'Very Low'
```

**Rationale**:
- **High (> 15%)**: Strong shared variance, meaningful relationship
- **Moderate (10-15%)**: Meaningful but not dominant relationship
- **Low (5-10%)**: Weak relationship, limited practical utility
- **Very Low (< 5%)**: Minimal shared variance, potential overfitting

These thresholds are based on:
1. Stewart & Love (1968) canonical correlation literature
2. Practical experience with CCA in psychophysiology
3. Conservative approach to avoid over-interpretation

## Integration with Export

The `export_results()` method was updated to automatically compute and export redundancy indices:

**New Files Created**:
1. `cca_redundancy_indices.csv`: Raw redundancy values
2. `cca_redundancy_indices_interpreted.csv`: With interpretation column

**CSV Schema**:
```
state, canonical_variate, r_canonical, var_explained_Y_by_U, 
var_explained_X_by_V, redundancy_Y_given_X, redundancy_X_given_Y, 
interpretation
```

## Edge Cases Handled

1. **Zero Variance**: If a canonical variate has zero variance, R² = 0
2. **Missing Data**: Already handled by `prepare_matrices()` (complete cases only)
3. **Single Component**: Works correctly with n_components=1
4. **Total Row**: Special handling for 'Total' canonical_variate in interpretation

## Performance Considerations

**Computational Complexity**:
- `_compute_variance_explained()`: O(n × p) where n = observations, p = variables
- `compute_redundancy_index()`: O(k × n × p) where k = n_components
- Overall: Linear in data size, very fast even for large datasets

**Memory Usage**:
- Minimal additional memory beyond CCA model storage
- Redundancy DataFrame is small (k rows × 7 columns)

## Validation Strategy

The implementation was validated through:

1. **Synthetic Data Test**: Created correlated physio-TET data with known structure
2. **Range Validation**: Verified 0 ≤ redundancy ≤ 1 for all variates
3. **Mathematical Consistency**: Verified total = sum of individual variates
4. **Visualization Check**: Confirmed plots render correctly
5. **Export Verification**: Confirmed all files created with correct schema

## Example Interpretation

**Scenario**: DMT state, CV1
```
r_canonical = 0.866
var_explained_Y_by_U = 0.382 (38.2%)
var_explained_X_by_V = 0.558 (55.8%)
redundancy_Y_given_X = 0.866² × 0.382 = 0.287 (28.7%)
redundancy_X_given_Y = 0.866² × 0.558 = 0.419 (41.9%)
interpretation = 'High'
```

**Meaning**:
- The first canonical variate captures strong shared variance
- 28.7% of TET variance is explained by the physiological canonical variate
- 41.9% of physiological variance is explained by the TET canonical variate
- This indicates a robust physiological-affective coupling during DMT

## Future Enhancements

Potential extensions (not currently implemented):

1. **Confidence Intervals**: Bootstrap CIs for redundancy indices
2. **Permutation Testing**: Test significance of redundancy values
3. **Variable-Level Redundancy**: Compute redundancy for individual variables
4. **Comparative Analysis**: Statistical tests comparing redundancy across states

## References

### Primary Literature
- Stewart, D., & Love, W. (1968). A general canonical correlation index. *Psychological Bulletin*, 70(3), 160-163.
- Lambert, Z. V., Wildt, A. R., & Durand, R. M. (1991). Approximating confidence intervals for factor loadings. *Multivariate Behavioral Research*, 26(3), 421-434.

### Implementation References
- scikit-learn CCA documentation
- Requirement 11.29-11.31 in TET analysis pipeline specification

---

**Document Version**: 1.0  
**Last Updated**: November 21, 2025  
**Author**: Kiro AI Assistant
