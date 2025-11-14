# Clustering Analysis Summary Report

**Generated**: 2025-11-13 16:47:22

**Significance Level**: α = 0.05

---

## Executive Summary

**Optimal KMeans Model**: k = 2 clusters (silhouette = 0.380)

**KMeans Stability**: highly stable (mean ARI = 0.995)


**Significant Dose Effects**: 4

**Significant Interactions**: 4

## Model Evaluation

### KMeans Clustering

| k | Silhouette Score | Interpretation |
|---|------------------|----------------|
| **2** | **0.380** | **Weak structure** |

## Bootstrap Stability

| Method | n_states | Mean ARI | 95% CI | Interpretation |
|--------|----------|----------|--------|----------------|
| KMeans | 2 | 0.995 | [0.990, 0.999] | Highly stable |

**Interpretation**: ARI (Adjusted Rand Index) measures clustering similarity. Higher values indicate more stable, reproducible clustering.

## State Occupancy Metrics

### KMeans

| Cluster/State | Fractional Occupancy | Number of Visits | Mean Dwell Time (bins) |
|---------------|---------------------|------------------|------------------------|
| 0 | 0.253 ± 0.303 | 0.9 ± 1.0 | 74.8 ± 90.4 |
| 1 | 0.747 ± 0.303 | 1.3 ± 0.7 | 139.9 ± 55.9 |

**Note**: Values shown as mean ± SD across all subject-sessions. Dwell times are in bins (multiply by 4 for seconds at 0.25 Hz sampling).

## Dose Effects

### Significant Effects (Permutation Tests, p < 0.05)

| Metric | Method | State | Observed Δ | p-value | Direction |
|--------|--------|-------|------------|---------|-----------|
| Fractional Occupancy | KMeans | 0 | 0.2320 | 0.0000 | High > Low |
| Mean Dwell Time | KMeans | 0 | 73.8611 | 0.0000 | High > Low |
| Fractional Occupancy | KMeans | 1 | -0.2320 | 0.0000 | High < Low |
| Mean Dwell Time | KMeans | 1 | -72.1389 | 0.0000 | High < Low |

**Summary**: 4 significant dose effects detected.

### Classical t-test Results (p < 0.05)

| Metric | Method | State | Mean Diff | t-statistic | p-value | 95% CI |
|--------|--------|-------|-----------|-------------|---------|--------|
| Fractional Occupancy | KMeans | 0 | 0.2320 | 2.80 | 0.0218 | [0.0570, 0.4071] |
| Mean Dwell Time | KMeans | 0 | 73.8611 | 3.08 | 0.0218 | [23.3017, 124.4205] |
| Fractional Occupancy | KMeans | 1 | -0.2320 | -2.80 | 0.0218 | [-0.4071, -0.0570] |
| Mean Dwell Time | KMeans | 1 | -72.1389 | -2.72 | 0.0218 | [-128.0879, -16.1898] |

**Summary**: 4 significant effects (classical tests).

## Interaction Effects (State × Dose)

### Significant Interactions (p < 0.05)

| Metric | Method | State | Interaction | DMT Δ | RS Δ | p-value | Interpretation |
|--------|--------|-------|-------------|-------|------|---------|----------------|
| Fractional Occupancy | KMeans | 0 | 0.2320 | 0.2320 | 0.0000 | 0.0000 | Stronger in DMT |
| Mean Dwell Time | KMeans | 0 | 73.8611 | 73.8611 | 0.0000 | 0.0000 | Stronger in DMT |
| Fractional Occupancy | KMeans | 1 | -0.2320 | -0.2320 | 0.0000 | 0.0000 | Weaker in DMT |
| Mean Dwell Time | KMeans | 1 | -72.1389 | -72.1389 | 0.0000 | 0.0000 | Weaker in DMT |

**Summary**: 4 significant State × Dose interactions.

**Interpretation**: Positive interaction indicates dose effect is stronger in DMT than RS. Negative interaction indicates dose effect is weaker in DMT than RS.

## Key Findings and Interpretations

1. **Experiential State Structure**: KMeans clustering identified 2 distinct experiential states with 
moderate cluster separation, suggesting some overlap between experiential states.

2. **Clustering Stability**: KMeans clustering showed 
good stability (mean ARI = 0.995), indicating robust state identification across bootstrap samples.

3. **Dose Effects**: Significant dose effects were observed for 4 state occupancy metrics. 

   - Fractional Occupancy: 2 significant effect(s)

   - Mean Dwell Time: 2 significant effect(s)

4. **State × Dose Interactions**: Significant interactions were found for 4 metrics, indicating that dose effects differ between DMT and resting state conditions. This suggests that the drug modulates dose sensitivity.

## Generated Figures

The following figures were generated from the clustering analysis:

- **clustering_kmeans_centroids_k2.png**: Centroid profile plots showing characteristic dimension patterns for each cluster (replicates Fig. 3.5)
- **clustering_kmeans_prob_timecourses_dmt_only.png**: Time-course cluster probability plots showing temporal dynamics for DMT sessions (replicates Fig. 3.6)

**Note**: GLHMM-related figures (state probability time courses and KMeans-GLHMM correspondence) were not generated because the GLHMM library is not installed. To generate these figures, install GLHMM with: `pip install git+https://github.com/vidaurre/glhmm` and rerun the analysis.

Refer to these figures for visual interpretation of the clustering results.