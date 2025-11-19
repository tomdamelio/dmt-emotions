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

### 7.4 Analytical Decisions

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

