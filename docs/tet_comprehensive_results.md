# TET Analysis: Comprehensive Results

**Generated:** 2025-11-14 16:04:37

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


## 3. Peak and AUC Analysis

**Note:** Peak/AUC results not available. Please run peak/AUC analysis first.


## 4. Dimensionality Reduction

### 4.1 Principal Components Interpretation

PCA identified 5 principal components explaining 76.6% of total variance.

#### PC1: Imagery intensity factor

**Top Positive Loadings**:

- Elementary Imagery: 0.39
- General Intensity: 0.38
- Temporality: 0.33
- Complex Imagery: 0.32
- Emotional Intensity: 0.31

**Temporal Dynamics**:

- Intercept: β = -2.42, 95% CI [-2.92, -1.93], p_fdr < 0.001
- C(state, Treatment('RS'))[T.DMT]: β = 3.70, 95% CI [3.49, 3.91], p_fdr < 0.001
- C(state, Treatment('RS'))[T.DMT]:C(dose, Treatment('Baja'))[T.Alta]: β = 1.80, 95% CI [1.51, 2.10], p_fdr < 0.001
- Group Var: β = 0.54, 95% CI [0.17, 0.92], p_fdr = 0.005

#### PC2: Affective valence factor: positive vs negative emotional experience

**Top Positive Loadings**:

- Bliss: 0.61
- Pleasantness: 0.57
- Salience: 0.22
- Emotional Intensity: 0.19
- Auditory: 0.09

**Top Negative Loadings**:

- Anxiety: -0.30
- Unpleasantness: -0.29
- Temporality: -0.15
- General Intensity: -0.06
- Disembodiment: -0.04

**Temporal Dynamics**:

- Intercept: β = -0.56, 95% CI [-0.85, -0.27], p_fdr < 0.001
- C(state, Treatment('RS'))[T.DMT]: β = -0.32, 95% CI [-0.44, -0.20], p_fdr < 0.001
- Group Var: β = 0.57, 95% CI [0.18, 0.96], p_fdr = 0.005

#### PC3: Imagery intensity factor

**Top Positive Loadings**:

- Auditory: 0.53
- Elementary Imagery: 0.41
- Complex Imagery: 0.29
- Bliss: 0.15
- Anxiety: 0.12

**Top Negative Loadings**:

- Salience: -0.51
- Interoception: -0.21
- General Intensity: -0.18
- Entity: -0.17
- Selfhood: -0.15

#### PC4: Imagery intensity factor

**Top Positive Loadings**:

- Temporality: 0.48
- Interoception: 0.30
- General Intensity: 0.28
- Pleasantness: 0.18
- Elementary Imagery: 0.13

**Top Negative Loadings**:

- Disembodiment: -0.38
- Salience: -0.36
- Entity: -0.33
- Selfhood: -0.29
- Complex Imagery: -0.26

#### PC5: Mixed experiential factor

**Top Positive Loadings**:

- Anxiety: 0.49
- Interoception: 0.48
- Selfhood: 0.33
- Emotional Intensity: 0.15
- Pleasantness: 0.14

**Top Negative Loadings**:

- Temporality: -0.41
- Complex Imagery: -0.28
- Entity: -0.26
- General Intensity: -0.15
- Disembodiment: -0.08

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

- **Temporality**: Significant across 1 methods
- **Elementary Imagery**: Significant across 1 methods
- **Emotional Intensity**: Significant across 1 methods
- **Complex Imagery**: Significant across 1 methods
- **General Intensity**: Significant across 1 methods

### 6.2 Method Correlations



## 7. Methodological Notes

### 7.1 Data Quality


### 7.2 Preprocessing

- **Standardization**: Global within-subject z-scoring
- **Time Windows**: LME: 0-9 min, AUC: 0-9 min
- **Trimming**: RS: 0-10 min, DMT: 0-20 min

### 7.3 Model Specifications

- **PCA**: 5 components retained, 76.6% variance explained

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

