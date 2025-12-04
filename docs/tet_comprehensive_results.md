# TET Analysis: Comprehensive Results

**Generated:** 2025-12-03 20:13:55

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


## 4. Dimensionality Reduction

### 4.1 Principal Components Interpretation

PCA identified 5 principal components explaining 95.1% of total variance.

#### PC1: Mixed experiential factor

**Top Positive Loadings**:

- Interoception: 0.52
- Emotional Intensity: 0.52
- Anxiety: 0.52
- Unpleasantness: 0.43
- Bliss: 0.04

**Temporal Dynamics**:

- C(state, Treatment('RS'))[T.DMT]: β = 2.31, 95% CI [2.16, 2.45], p_fdr < 0.001
- C(state, Treatment('RS'))[T.DMT]:C(dose, Treatment('Baja'))[T.Alta]: β = 0.68, 95% CI [0.47, 0.89], p_fdr < 0.001

#### PC2: Affective valence factor: positive vs negative emotional experience

**Top Positive Loadings**:

- Pleasantness: 0.63
- Bliss: 0.63
- Emotional Intensity: 0.25
- Interoception: 0.13

**Top Negative Loadings**:

- Unpleasantness: -0.31
- Anxiety: -0.20

**Temporal Dynamics**:

- Intercept: β = -0.82, 95% CI [-1.16, -0.48], p_fdr < 0.001
- Group Var: β = 0.56, 95% CI [0.17, 0.95], p_fdr = 0.005

#### PC3: Affective valence factor: positive vs negative emotional experience

**Top Positive Loadings**:

- Unpleasantness: 0.76
- Bliss: 0.34
- Pleasantness: 0.12

**Top Negative Loadings**:

- Interoception: -0.53
- Anxiety: -0.11
- Emotional Intensity: -0.02

#### PC4: Affective valence factor: positive vs negative emotional experience

**Top Positive Loadings**:

- Pleasantness: 0.59
- Unpleasantness: 0.30
- Interoception: 0.30

**Top Negative Loadings**:

- Bliss: -0.56
- Anxiety: -0.36
- Emotional Intensity: -0.17

#### PC5: Affective valence factor: positive vs negative emotional experience

**Top Positive Loadings**:

- Anxiety: 0.74
- Pleasantness: 0.47

**Top Negative Loadings**:

- Emotional Intensity: -0.36
- Interoception: -0.28
- Unpleasantness: -0.13
- Bliss: -0.09

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



## 6. Physiological-Affective Integration

### 6.1 Canonical Correlation Analysis (CCA)

**Canonical Correlations:**

- **RS State, CV1**: r = 0.634*, p = p_fdr < 0.001
- **RS State, CV2**: r = 0.494*, p = p_fdr < 0.001
- **DMT State, CV1**: r = 0.678*, p = p_fdr < 0.001
- **DMT State, CV2**: r = 0.297*, p = p_fdr < 0.001

### 6.2 CCA Validation: Permutation Testing

**Subject-level permutation test results** (1000 iterations):

- **RS State, CV1**: r_obs = 0.634, p_perm = p_fdr = 0.228
- **RS State, CV2**: r_obs = 0.494, p_perm = p_fdr = 0.168
- **DMT State, CV1**: r_obs = 0.678, p_perm = p_fdr = 0.119
- **DMT State, CV2**: r_obs = 0.297, p_perm = p_fdr = 0.832

**Interpretation:** No canonical correlations survived permutation testing, suggesting potential overfitting or weak coupling.

### 6.3 CCA Validation: Cross-Validation

**Leave-One-Subject-Out (LOSO) cross-validation:**

- **RS State, CV1**:
  - Out-of-sample r: -0.276 ± 0.241
  - In-sample r: 0.634
  - Overfitting index: 1.435
- **RS State, CV2**:
  - Out-of-sample r: 0.354 ± 0.514
  - In-sample r: 0.494
  - Overfitting index: 0.283
- **DMT State, CV1**:
  - Out-of-sample r: 0.494 ± 0.306
  - In-sample r: 0.678
  - Overfitting index: 0.271
- **DMT State, CV2**:
  - Out-of-sample r: 0.195 ± 0.432
  - In-sample r: 0.297
  - Overfitting index: 0.344

**Interpretation:** High overfitting (mean index = 0.583), indicating poor generalization and potential model instability.

### 6.4 CCA Validation: Redundancy Analysis

**Redundancy indices** (percentage of variance explained):

- **RS State, CV1**:
  - TET variance explained by physio: 3.2%
  - Physio variance explained by TET: 5.5%
  - Interpretation: N/A
- **RS State, CV2**:
  - TET variance explained by physio: 0.8%
  - Physio variance explained by TET: 2.0%
  - Interpretation: N/A
- **DMT State, CV1**:
  - TET variance explained by physio: 3.2%
  - Physio variance explained by TET: 10.3%
  - Interpretation: N/A
- **DMT State, CV2**:
  - TET variance explained by physio: 0.1%
  - Physio variance explained by TET: 0.2%
  - Interpretation: N/A

**Interpretation:** Low shared variance (mean redundancy = 3.2%), indicating that CCA may be capturing noise rather than meaningful coupling.

### 6.5 CCA Validation Summary

**Conclusion:** CCA results show signs of **potential overfitting** or weak coupling:

- No canonical correlations survived permutation testing
- High overfitting in cross-validation (index = 0.583)
- Low redundancy indices (mean = 3.2%)

**Recommendation:** Interpret CCA results with caution. Consider alternative approaches or larger sample sizes.



## 7. Cross-Analysis Integration

### 7.1 Convergent Findings

The following dimensions showed consistent effects across multiple methods:

- **Interoception**: Significant across 1 methods
- **Anxiety**: Significant across 1 methods
- **Bliss**: Significant across 1 methods
- **Emotional Intensity**: Significant across 1 methods
- **Unpleasantness**: Significant across 1 methods

### 7.2 Method Correlations



## 8. Methodological Notes

### 8.1 Data Quality


### 8.2 Preprocessing

- **Standardization**: Global within-subject z-scoring
- **Time Windows**: LME: 0-9 min, AUC: 0-9 min
- **Trimming**: RS: 0-10 min, DMT: 0-20 min

### 8.3 Model Specifications

- **PCA**: 5 components retained, 95.1% variance explained

### 8.4 Analytical Decisions

- **Fdr Correction**: Benjamini-Hochberg, applied separately per effect type
- **Significance Threshold**: p_fdr < 0.05
- **Pca Threshold**: 70-80% cumulative variance
- **Clustering Selection**: Highest silhouette score



## 9. Further Investigation

### 9.1 Unresolved Questions and Ambiguous Findings

1. **Heterogeneous cluster**: Cluster 1 has few defining dimensions
   - *Suggested analysis*: Consider k=3 or k=4 solutions to separate heterogeneous states

2. **Heterogeneous cluster**: Cluster 0 has few defining dimensions
   - *Suggested analysis*: Consider k=3 or k=4 solutions to separate heterogeneous states

### 9.2 Suggested Follow-up Analyses

**High Priority**:

1. **Generalized Additive Models (GAMs)**: Capture non-linear temporal dynamics
2. **Individual Difference Analysis**: Characterize dose response variability
3. **GLHMM State Modeling**: Model temporal state transitions

**Medium Priority**:

4. **Multivariate Time Series Analysis**: Understand dimension co-variation
5. **Sensitivity Analyses**: Test robustness to analytical decisions

