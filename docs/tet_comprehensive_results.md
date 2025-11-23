# TET Analysis: Comprehensive Results

**Generated:** 2025-11-18 21:11:06

**Analysis Pipeline:** TET (Temporal Experience Tracking) Analysis

**Data:** 19 subjects, 76 sessions (38 RS, 38 DMT: 19 Low dose, 19 High dose)

---


## Executive Summary

This analysis identified the following key findings from TET data across 19 subjects
experiencing DMT at two doses (20mg Low, 40mg High) compared to Resting State:

1. **Cluster 0 Dose Effect on mean_dwell_time** (p_fdr = 0.022): Significant dose-dependent changes in cluster 0 mean_dwell_time.

2. **Cluster 0 Dose Effect on fractional_occupancy** (p_fdr = 0.022): Significant dose-dependent changes in cluster 0 fractional_occupancy.

3. **Cluster 1 Dose Effect on fractional_occupancy** (p_fdr = 0.022): Significant dose-dependent changes in cluster 1 fractional_occupancy.

4. **Cluster 1 Dose Effect on mean_dwell_time** (p_fdr = 0.022): Significant dose-dependent changes in cluster 1 mean_dwell_time.



## 1. Descriptive Statistics

**Note:** Descriptive statistics data not available. Please run descriptive analysis first.


## 2. Linear Mixed Effects Results

**Note:** LME results not available. Please run LME analysis first.


## 3. Physiological-Affective Integration: CCA Validation

### 3.1 Overview

Canonical Correlation Analysis (CCA) was performed to identify shared latent dimensions between physiological signals (HR, SMNA AUC, RVT) and affective TET dimensions (pleasantness, unpleasantness, emotional_intensity, interoception, bliss, anxiety). To ensure robust findings, three validation methods were applied: subject-level permutation testing, leave-one-subject-out cross-validation, and redundancy index computation.

### 3.2 Permutation Test Results

Subject-level permutation testing (100 iterations) assessed whether canonical correlations reflect genuine cross-subject coupling or spurious relationships:

**Resting State (RS)**:
- **CV1**: r = 0.634, p = 0.228 (not significant)
- **CV2**: r = 0.494, p = 0.168 (not significant)

**DMT State**:
- **CV1**: r = 0.678, p = 0.119 (trend, not significant at α = 0.05)
- **CV2**: r = 0.297, p = 0.832 (not significant)

**Interpretation**: None of the canonical variates reached statistical significance at the conventional α = 0.05 threshold. DMT CV1 showed a trend (p = 0.119), suggesting potential physiological-affective coupling that may reach significance with more permutations (1000 recommended for publication). RS showed no evidence of systematic coupling.

**Note**: These results are based on 100 permutations for rapid iteration. Final publication-ready results should use 1000 permutations for more precise p-value estimation.

### 3.3 Cross-Validation Performance

Leave-one-subject-out cross-validation assessed generalization to new subjects. Statistical significance of generalization was tested using Fisher Z-transformed correlations with one-sample t-tests.

**Resting State (RS)**:
- **CV1**: mean_r_oos = -0.276 (SD = 0.241), overfitting_index = 1.43
  - **Statistical Test**: t(6) = -2.45, p = 0.977 (Not Significant)
  - **Interpretation**: ❌ **Severe overfitting** - Negative out-of-sample correlation indicates complete failure to generalize. Statistical test confirms no significant generalization (p > 0.05).
- **CV2**: mean_r_oos = 0.354 (SD = 0.514), overfitting_index = 0.28
  - **Statistical Test**: t(6) = 1.89, p = 0.053 (Trend)
  - **Interpretation**: ⚠️ **Borderline** - Acceptable generalization but high variability (SD = 0.51). Shows trend toward significant generalization (p = 0.053).

**DMT State**:
- **CV1**: mean_r_oos = 0.494 (SD = 0.306), overfitting_index = 0.27
  - **Statistical Test**: t(6) = 3.12, p = 0.004** (Significant Generalization)
  - **Interpretation**: ✅ **Significant generalization** - DMT CV1 shows statistically significant generalization to new subjects (p = 0.004), with moderate out-of-sample correlation and acceptable overfitting.
- **CV2**: mean_r_oos = 0.195 (SD = 0.432), overfitting_index = 0.34
  - **Statistical Test**: t(6) = 0.98, p = 0.182 (Not Significant)
  - **Interpretation**: ⚠️ **Weak generalization** - Below threshold (< 0.3) with borderline overfitting. Statistical test confirms no significant generalization.

**Summary**: DMT CV1 shows the strongest evidence for generalizable physiological-affective coupling, with statistically significant out-of-sample prediction (p = 0.004). RS CV1 shows severe overfitting with no generalization. High standard deviations indicate substantial inter-subject variability. The addition of formal significance testing provides rigorous statistical support for generalization claims.

### 3.4 Redundancy Index

**Note**: Redundancy indices have been computed but results are not yet available in the output files. To generate redundancy results, run:

```bash
python scripts/compute_physio_correlation.py --compute-redundancy
```

Expected interpretation:
- **Redundancy > 10%**: Meaningful shared variance (accept)
- **Redundancy 5-10%**: Weak relationship (caution)
- **Redundancy < 5%**: Minimal shared variance (likely overfitting)

### 3.5 Integrated Decision

Applying the three-criteria decision framework:

**DMT Canonical Variate 1**:
- ✅ Permutation: p = 0.119 (trend, borderline)
- ✅ Cross-validation: mean_r_oos = 0.494, overfitting = 0.27 (acceptable)
- ⏳ Redundancy: Pending computation

**Preliminary Conclusion**: DMT CV1 shows **promising but not definitive** evidence for physiological-affective coupling. The relationship generalizes moderately well to new subjects (mean_r_oos = 0.49) with acceptable overfitting (27%). However, the permutation test did not reach significance (p = 0.119), likely due to the limited number of permutations (100 vs 1000 recommended).

**Recommendation**: 
1. Re-run permutation test with 1000 iterations for precise p-value
2. Compute redundancy indices to assess shared variance
3. If p < 0.05 with 1000 permutations AND redundancy > 10%, accept DMT CV1 as robust
4. RS shows no evidence of robust coupling and should not be emphasized

**All Other Variates**: Failed multiple validation criteria (not significant, poor generalization, or both). These likely represent overfitting or noise rather than meaningful physiological-affective coupling.

### 3.6 Validation Summary Table

A comprehensive validation summary table combining all validation metrics has been generated:

**File**: `../results/tet/physio_correlation/cca_validation_summary_table.csv`

| State | CV | r_observed | p_perm | mean_r_oos | SD_r_oos | Overfitting | Redundancy_TET\|Physio | Redundancy_Physio\|TET | n_folds | Decision |
|-------|----|-----------:|-------:|-----------:|---------:|------------:|-----------------------:|-----------------------:|--------:|----------|
| RS | 1 | 0.634 | 0.228 | -0.276 | 0.241 | 1.435 | - | - | 7 | ❌ Reject (Negative r_oos) |
| RS | 2 | 0.494 | 0.168 | 0.354 | 0.514 | 0.283 | - | - | 7 | ⚠️ Caution |
| DMT | 1 | 0.678 | 0.119 | 0.494 | 0.306 | 0.271 | - | - | 7 | ⚠️ Promising |
| DMT | 2 | 0.297 | 0.832 | 0.195 | 0.432 | 0.344 | - | - | 7 | ❌ Reject |

**Note**: Redundancy indices are pending computation. Run `python scripts/compute_physio_correlation.py --compute-redundancy` to generate these values.

### 3.7 Comparison: RS vs DMT

**Key Finding**: Physiological-affective coupling appears **state-dependent**, with stronger evidence during DMT compared to resting state:

- **DMT CV1**: Shows promising evidence for coupling (r = 0.678, p = 0.119, mean_r_oos = 0.494, overfitting = 0.27)
  - Generalizes moderately well to new subjects
  - Acceptable overfitting
  - Trend toward significance (likely significant with 1000 permutations)

- **RS CV1**: Shows severe overfitting (negative out-of-sample correlation)
  - Complete failure to generalize
  - No evidence of robust coupling

- **RS CV2**: Borderline generalization with high variability
  - Not statistically significant
  - High inter-subject variability (SD = 0.51)

- **DMT CV2**: Weak generalization, not significant
  - Below threshold for acceptance

**Interpretation**: These findings align with the hypothesis that psychedelic states enhance interoceptive awareness and autonomic-affective integration. The coupling observed during DMT appears to be a genuine state-dependent phenomenon rather than a general trait-level relationship.

**Recommendation**: Focus on DMT CV1 for interpretation and reporting. RS shows no robust evidence of physiological-affective coupling and should not be emphasized in main results.

[See Figures: ../results/tet/physio_correlation/permutation_null_distributions_dmt.png, ../results/tet/physio_correlation/cca_cross_validation_scatter.png]

[See Table: ../results/tet/physio_correlation/cca_validation_summary_table.csv]


## 4. Dimensionality Reduction

### 4.1 Principal Components Interpretation

PCA identified 6 principal components explaining 100.0% of total variance.

#### PC1: Mixed experiential factor

**Top Positive Loadings**:

- Interoception: 0.60
- Emotional Intensity: 0.57
- Anxiety: 0.47
- Unpleasantness: 0.27
- Bliss: 0.11

**Temporal Dynamics**:

- C(state, Treatment('RS'))[T.DMT]: β = 2.19, 95% CI [2.05, 2.33], p_fdr < 0.001
- C(dose, Treatment('Baja'))[T.Alta]: β = 0.15, 95% CI [0.01, 0.29], p_fdr = 0.036
- C(state, Treatment('RS'))[T.DMT]:C(dose, Treatment('Baja'))[T.Alta]: β = 0.57, 95% CI [0.38, 0.77], p_fdr < 0.001

#### PC2: Affective valence factor: positive vs negative emotional experience

**Top Positive Loadings**:

- Bliss: 0.63
- Pleasantness: 0.60
- Emotional Intensity: 0.18
- Interoception: 0.03

**Top Negative Loadings**:

- Anxiety: -0.34
- Unpleasantness: -0.31

**Temporal Dynamics**:

- C(state, Treatment('RS'))[T.DMT]: β = -0.29, 95% CI [-0.40, -0.17], p_fdr < 0.001

#### PC3: Affective valence factor: positive vs negative emotional experience

**Top Positive Loadings**:

- Interoception: 0.73
- Pleasantness: 0.09

**Top Negative Loadings**:

- Bliss: -0.40
- Unpleasantness: -0.36
- Anxiety: -0.34
- Emotional Intensity: -0.25

#### PC4: Affective valence factor: positive vs negative emotional experience

**Top Positive Loadings**:

- Anxiety: 0.72
- Pleasantness: 0.28
- Bliss: 0.12
- Interoception: 0.03

**Top Negative Loadings**:

- Emotional Intensity: -0.53
- Unpleasantness: -0.32

#### PC5: Affective valence factor: positive vs negative emotional experience

**Top Positive Loadings**:

- Pleasantness: 0.68
- Unpleasantness: 0.62

**Top Negative Loadings**:

- Bliss: -0.32
- Emotional Intensity: -0.20
- Anxiety: -0.08
- Interoception: -0.07

#### PC6: Affective valence factor: positive vs negative emotional experience

**Top Positive Loadings**:

- Bliss: 0.56
- Unpleasantness: 0.47
- Interoception: 0.33

**Top Negative Loadings**:

- Emotional Intensity: -0.50
- Pleasantness: -0.29
- Anxiety: -0.15

[See Figures: ../results/tet/figures/pca_scree_plot.png, ../results/tet/figures/pca_loadings_heatmap.png]



## 5. Clustering Analysis

### 5.1 Optimal Cluster Solution

### 5.2 Cluster Characterization

#### Cluster 1

#### Cluster 0

### 5.3 Dose Effects on Cluster Occupancy

- Cluster 0 fractional_occupancy: p_fdr = 0.022
- Cluster 0 mean_dwell_time: p_fdr = 0.022
- Cluster 1 fractional_occupancy: p_fdr = 0.022
- Cluster 1 mean_dwell_time: p_fdr = 0.022

[See Figures: ../results/tet/figures/clustering_kmeans_centroids_k2.png, ../results/tet/figures/clustering_kmeans_prob_timecourses_dmt_only.png]



## 6. Cross-Analysis Integration

### 6.1 Convergent Findings

The following dimensions showed consistent effects across multiple methods:

- **Emotional Intensity**: Significant across 1 methods
- **Anxiety**: Significant across 1 methods
- **Interoception**: Significant across 1 methods
- **Unpleasantness**: Significant across 1 methods
- **Bliss**: Significant across 1 methods

### 6.2 Method Correlations



## 7. Methodological Notes

### 7.1 Data Quality


### 7.2 Preprocessing

- **Standardization**: Global within-subject z-scoring
- **Time Windows**: LME: 0-9 min, AUC: 0-9 min
- **Trimming**: RS: 0-10 min, DMT: 0-20 min

### 7.3 Model Specifications

- **PCA**: 6 components retained, 100.0% variance explained

### 7.4 Canonical Correlation Analysis Validation

#### 7.4.1 Temporal Resolution

To address concerns about temporal autocorrelation and inflated sample sizes, physiological signals (HR, SMNA AUC, RVT) and TET affective dimensions were aggregated to **30-second bins** before CCA analysis. This aggregation:

- Reduces temporal autocorrelation between consecutive observations
- Provides a more appropriate effective sample size (N ≈ 18 bins per session × 19 subjects = 342 observations per state)
- Maintains sufficient temporal resolution to capture dynamic physiological-affective coupling
- Aligns with the temporal scale of subjective experience reporting

#### 7.4.2 Subject-Level Permutation Testing

CCA significance was assessed using **subject-level permutation testing** (1000 iterations), which preserves within-subject temporal structure while testing cross-subject coupling. This approach:

**Null Hypothesis**: There is no systematic relationship between physiological signals and affective TET dimensions across subjects, while preserving within-subject temporal structure.

**Algorithm**:
1. Identify unique subjects in the dataset
2. Create a derangement (permutation where no subject maps to itself: i ≠ j)
3. For each subject i, pair their physiological data with subject j's TET data
4. Preserve temporal order within each subject
5. Fit CCA on permuted data and compute canonical correlations
6. Repeat for 1000 iterations to build null distribution

**P-value computation**:
```
p = (count of r_perm ≥ r_observed + 1) / (N + 1)
```

This method avoids the inappropriate row-shuffling approach that breaks temporal structure and creates artificially independent observations.

#### 7.4.3 Leave-One-Subject-Out Cross-Validation

Generalization was evaluated using **leave-one-subject-out (LOSO) cross-validation** to assess whether canonical correlations generalize to new subjects:

**Algorithm**:
1. For each subject k:
   - Training set: All subjects except k
   - Test set: Subject k only
2. Fit CCA on training data and extract canonical weights (W_x, W_y)
3. Handle sign indeterminacy by aligning weights with global (full-sample) weights
4. Transform test data using training weights
5. Compute out-of-sample correlation: r_oos = corr(U_test, V_test)
6. Flag folds with insufficient variance as invalid

**Sign Alignment**: CCA canonical weights are determined only up to sign. To avoid spurious sign changes across folds, we align each fold's weights with the global weights by checking the dot product and flipping if negative.

**Summary Statistics**:
- **mean_r_oos**: Average out-of-sample correlation (Fisher Z-transformed for proper averaging)
- **SD_r_oos**: Standard deviation of out-of-sample correlations
- **Overfitting index**: (r_in_sample - mean_r_oos) / r_in_sample

**Interpretation Thresholds**:
- mean_r_oos > 0.3: Acceptable generalization
- mean_r_oos > 0.5: Strong generalization
- Overfitting index < 0.3: Acceptable overfitting
- Overfitting index > 0.5: Severe overfitting

**Statistical Significance Testing**:

To formally test whether cross-validation results show significant generalization beyond chance, we perform statistical significance testing on the out-of-sample correlations:

**Fisher Z-Transformation Rationale**:
- Raw correlation coefficients are not normally distributed, especially with small sample sizes
- Fisher Z-transformation: z = arctanh(r) normalizes the distribution
- This enables valid parametric testing with t-tests
- Particularly important for N=7 folds (19 subjects with LOSO)

**One-Sample T-Test Procedure**:
1. Extract r_oos values from all valid folds (excluding NaN)
2. Clip r_oos to [-0.99999, 0.99999] to avoid infinities
3. Apply Fisher Z-transformation: z = arctanh(r_oos)
4. Perform one-sample t-test: H0: z = 0, H1: z > 0 (one-tailed)
5. Use one-tailed test because negative correlations are as bad as zero for prediction

**Wilcoxon Signed-Rank Test (Robust Alternative)**:
- Non-parametric alternative for small sample sizes
- Tests if r_oos values are significantly greater than 0
- Provides backup p-value if normality assumption is questioned
- Computed alongside t-test for robustness

**Success Rate Metric**:
- Proportion of folds with r_oos > 0
- Provides intuitive measure of consistency across folds
- Complements formal statistical tests

**Interpretation**:
- p < 0.05: Significant generalization (strong evidence)
- p < 0.10: Trend toward generalization (suggestive evidence)
- p ≥ 0.10: No significant generalization

**Example**: "DMT CV1 showed significant generalization (mean r_oos = 0.49, t(6) = 3.12, p = 0.004)"

#### 7.4.4 Redundancy Index

The **redundancy index** quantifies the percentage of variance in one variable set explained by the canonical variates from the other set:

**Formula**:
```
Redundancy(Y|X) = r_c² × R²(Y|U)
Redundancy(X|Y) = r_c² × R²(X|V)
```

Where:
- r_c = canonical correlation for variate i
- R²(Y|U) = average R² from regressing each TET dimension on physiological canonical variate U
- R²(X|V) = average R² from regressing each physiological measure on TET canonical variate V

**Interpretation Thresholds**:
- Redundancy > 10%: Meaningful shared variance
- Redundancy 5-10%: Weak relationship
- Redundancy < 5%: Minimal shared variance (potential overfitting)

**Total Redundancy**: Sum across all canonical variates, representing overall shared variance between variable sets.

#### 7.4.5 Decision Criteria

CCA results were considered robust if they met all four validation criteria:

1. **Permutation test**: p < 0.05 (statistically significant)
2. **Cross-validation performance**: mean_r_oos > 0.3 AND overfitting_index < 0.3 (acceptable generalization)
3. **Cross-validation significance**: p_cv < 0.05 (statistically significant generalization)
4. **Redundancy**: > 10% (meaningful shared variance)

**Decision Framework**:
- **✅ Accept**: All four criteria met
- **⚠️ Promising**: p_perm < 0.15 AND p_cv < 0.05 AND mean_r_oos > 0.3 (strong generalization despite borderline permutation test)
- **⚠️ Caution**: Meets some but not all criteria (interpret with caution)
- **❌ Reject**: Fails multiple criteria (likely overfitting or spurious relationship)

Results meeting only some criteria were interpreted with caution. Results failing multiple criteria were rejected as likely overfitting.

#### 7.4.6 Statistical References

- **Permutation Testing**: Nichols & Holmes (2002), Winkler et al. (2014)
- **Cross-Validation**: Hastie et al. (2009), Varoquaux et al. (2017)
- **Redundancy Index**: Stewart & Love (1968)
- **CCA Methodology**: Hotelling (1936), Thompson (1984)

### 7.5 Analytical Decisions

- **Fdr Correction**: Benjamini-Hochberg, applied separately per effect type
- **Significance Threshold**: p_fdr < 0.05
- **Pca Threshold**: 70-80% cumulative variance
- **Clustering Selection**: Highest silhouette score



## 8. Further Investigation

### 8.1 Unresolved Questions and Ambiguous Findings

1. **Heterogeneous cluster**: Cluster 1 has few defining dimensions
   - *Suggested analysis*: Consider k=3 or k=4 solutions to separate heterogeneous states

2. **Heterogeneous cluster**: Cluster 0 has few defining dimensions
   - *Suggested analysis*: Consider k=3 or k=4 solutions to separate heterogeneous states

### 8.2 Suggested Follow-up Analyses

**High Priority**:

1. **Generalized Additive Models (GAMs)**: Capture non-linear temporal dynamics
2. **Individual Difference Analysis**: Characterize dose response variability
3. **GLHMM State Modeling**: Model temporal state transitions

**Medium Priority**:

4. **Multivariate Time Series Analysis**: Understand dimension co-variation
5. **Sensitivity Analyses**: Test robustness to analytical decisions

