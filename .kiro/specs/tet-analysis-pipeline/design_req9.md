# Design Document: Requirement 9 - Comprehensive Results Synthesis and Reporting

## Overview

This document describes the design for implementing an automated comprehensive results synthesis system that aggregates, interprets, and documents findings from all TET analysis components. The system SHALL generate a structured markdown document at `docs/tet_comprehensive_results.md` that synthesizes statistical results, identifies patterns across analyses, and highlights areas requiring further investigation.

## Requirement Summary

**Requirement 9**: Generate a comprehensive results document synthesizing all TET analysis findings.

**Key Acceptance Criteria**:
- 9.1: Generate comprehensive results document at `docs/tet_comprehensive_results.md`
- 9.2: Include Executive Summary with 3-5 most important findings
- 9.3: Include Descriptive Statistics section
- 9.4: Include LME Results section with significant effects
- 9.5: Include Peak and AUC Analysis section
- 9.6: Include Dimensionality Reduction section
- 9.7: Include Clustering Analysis section
- 9.8: Include Cross-Analysis Integration section
- 9.9: Include Methodological Notes section
- 9.10: Include Further Investigation section
- 9.11: Format statistical results consistently
- 9.12: Reference specific figures and tables
- 9.13: Organize findings hierarchically
- 9.14: Generate automatically after all analyses complete
- 9.15: Update when any analysis component is re-run

## Architecture

### Component Overview

```
TET Results Synthesis System
├── Data Aggregator
│   ├── Load all analysis outputs
│   ├── Validate data completeness
│   └── Extract key metrics
├── Results Analyzer
│   ├── Identify significant findings
│   ├── Compute effect size rankings
│   ├── Detect convergent patterns
│   └── Flag ambiguous results
├── Report Generator
│   ├── Executive Summary builder
│   ├── Section-specific formatters
│   ├── Statistical notation formatter
│   └── Cross-reference manager
└── Document Writer
    ├── Markdown template engine
    ├── Figure/table linker
    └── Version control integration
```


### Design Principles

1. **Automation**: Document generation triggered automatically after analysis completion
2. **Consistency**: Standardized statistical notation and formatting throughout
3. **Interpretability**: Clear hierarchical organization with actionable insights
4. **Traceability**: All findings linked to source data and figures
5. **Reproducibility**: Document regenerates consistently from same inputs
6. **Actionability**: Explicit identification of areas requiring further investigation

## Detailed Design

### 9.1: Document Structure and Location

**Purpose**: Provide a centralized, accessible results document.

**Implementation**:
- **Output Path**: `docs/tet_comprehensive_results.md`
- **Format**: Markdown with hierarchical sections
- **Version Control**: Track changes via git
- **Accessibility**: Human-readable, version-controllable, easily shareable

**Document Template Structure**:
```markdown
# TET Analysis: Comprehensive Results

## Executive Summary
[3-5 key findings with effect sizes]

## 1. Descriptive Statistics
### 1.1 Temporal Dynamics Overview
### 1.2 Dimension-Specific Patterns
[For each dimension: peak timing, trajectories, baseline comparisons]

## 2. Linear Mixed Effects Results
### 2.1 State Effects
### 2.2 Dose Effects
### 2.3 Interaction Effects
### 2.4 Temporal Effects
[Organized by effect type, then by dimension]

## 3. Peak and AUC Analysis
### 3.1 Peak Value Comparisons
### 3.2 Time to Peak Analysis
### 3.3 Area Under Curve Analysis
### 3.4 Dose Sensitivity Rankings

## 4. Dimensionality Reduction
### 4.1 Principal Components Interpretation
### 4.2 Variance Explained
### 4.3 Component Temporal Dynamics
### 4.4 Loading Patterns

## 5. Clustering Analysis
### 5.1 Optimal Cluster Solutions
### 5.2 Cluster Characterization
### 5.3 Temporal Prevalence Patterns
### 5.4 Dose Effects on Cluster Occupancy
### 5.5 Stability Metrics

## 6. Cross-Analysis Integration
### 6.1 Convergent Findings
### 6.2 Dimension Rankings Across Methods
### 6.3 Consistent Dose Effects
### 6.4 Temporal Pattern Concordance

## 7. Methodological Notes
### 7.1 Data Quality
### 7.2 Model Assumptions
### 7.3 Limitations
### 7.4 Analytical Decisions

## 8. Further Investigation
### 8.1 Unresolved Questions
### 8.2 Ambiguous Findings
### 8.3 Contradictory Results
### 8.4 Suggested Follow-up Analyses

## Appendix
### A. Statistical Notation Guide
### B. Figure and Table Index
### C. Analysis Parameters
```


### 9.2: Executive Summary Generation

**Purpose**: Highlight the most important findings for rapid comprehension.

**Implementation**:
- **Module**: `scripts/generate_comprehensive_report.py`
- **Class**: `TETResultsSynthesizer`
- **Method**: `generate_executive_summary()`

**Selection Criteria for Key Findings**:
1. **Effect Size Magnitude**: Prioritize largest standardized effects (|β| or r)
2. **Statistical Significance**: Focus on p_fdr < 0.01 (highly significant)
3. **Consistency**: Findings replicated across multiple analysis methods
4. **Theoretical Relevance**: Effects on core dimensions (pleasantness, anxiety, imagery)
5. **Dose Sensitivity**: Clear High vs Low dose differences

**Ranking Algorithm**:
```python
def rank_findings(all_results):
    """Rank findings by importance score."""
    findings = []
    
    # LME State effects
    for dim, beta, p_fdr in lme_state_effects:
        if p_fdr < 0.01:
            score = abs(beta) * (1 - p_fdr) * 2.0  # Weight State effects highly
            findings.append({
                'type': 'LME_State',
                'dimension': dim,
                'effect_size': beta,
                'p_fdr': p_fdr,
                'score': score
            })
    
    # Peak/AUC dose effects
    for dim, r, p_fdr in peak_dose_effects:
        if p_fdr < 0.05:
            score = abs(r) * (1 - p_fdr) * 1.5
            findings.append({
                'type': 'Peak_Dose',
                'dimension': dim,
                'effect_size': r,
                'p_fdr': p_fdr,
                'score': score
            })
    
    # Clustering dose effects
    for cluster, stat, p_fdr in cluster_dose_effects:
        if p_fdr < 0.05:
            score = abs(stat) * (1 - p_fdr) * 1.2
            findings.append({
                'type': 'Cluster_Dose',
                'cluster': cluster,
                'effect_size': stat,
                'p_fdr': p_fdr,
                'score': score
            })
    
    # Sort by score and return top 5
    findings.sort(key=lambda x: x['score'], reverse=True)
    return findings[:5]
```

**Output Format**:
```markdown
## Executive Summary

This analysis identified the following key findings from TET data across 19 subjects 
experiencing DMT at two doses (20mg Low, 40mg High) compared to Resting State:

1. **Strong DMT State Effect on Complex Imagery** (β = 2.34, 95% CI [1.98, 2.70], 
   p_fdr < 0.001): Complex imagery showed the strongest increase during DMT compared 
   to RS, with effects emerging within 2 minutes and persisting throughout the session.
   [See Figure: timeseries_complex_imagery.png]

2. **Dose-Dependent Anxiety Reduction** (r = -0.52, 95% CI [-0.71, -0.28], 
   p_fdr = 0.003): High dose DMT showed significantly lower peak anxiety compared 
   to Low dose, suggesting anxiolytic effects at higher doses.
   [See Figure: peak_auc_dose_comparison.png]

[... 3-5 total findings ...]
```


### 9.3: Descriptive Statistics Section

**Purpose**: Summarize temporal dynamics and basic patterns for each dimension.

**Implementation**:
- **Input Data**: `results/tet/descriptive/timecourse_*.csv`, `results/tet/descriptive/session_summaries.csv`
- **Method**: `generate_descriptive_section()`

**Content for Each Dimension**:
1. **Peak Timing**: Mean and range of time_to_peak across subjects
2. **Trajectory Shape**: Onset speed, plateau duration, offset pattern
3. **Dose Comparison**: Mean difference in raw scores between High and Low
4. **Baseline Comparison**: DMT vs RS mean values
5. **Inter-Subject Variability**: Coefficient of variation

**Output Format**:
```markdown
## 1. Descriptive Statistics

### 1.1 Temporal Dynamics Overview

Across all dimensions, DMT sessions showed characteristic temporal profiles with:
- Rapid onset (mean time to peak: 3.2 ± 1.4 minutes)
- Sustained plateau (mean duration: 4.8 ± 2.1 minutes)
- Gradual offset (return to baseline by 15-18 minutes)

### 1.2 Dimension-Specific Patterns

#### Complex Imagery
- **Peak Timing**: 2.8 ± 1.2 minutes (range: 1.0-5.5 min)
- **Peak Intensity**: High dose = 8.2 ± 1.1, Low dose = 7.4 ± 1.3 (raw scale)
- **Baseline**: RS = 1.2 ± 0.8
- **Trajectory**: Steep onset (0-3 min), sustained plateau (3-10 min), gradual decline
- **Variability**: CV = 0.31 (moderate inter-subject consistency)

[Repeat for all 15 dimensions...]
```

**Data Extraction**:
```python
def extract_descriptive_stats(dimension):
    """Extract key descriptive statistics for a dimension."""
    # Load time course data
    tc = pd.read_csv(f'results/tet/descriptive/timecourse_{dimension}.csv')
    
    # Load session summaries
    summ = pd.read_csv('results/tet/descriptive/session_summaries.csv')
    summ_dim = summ[summ['dimension'] == dimension]
    
    stats = {
        'peak_time_mean': summ_dim['time_to_peak'].mean(),
        'peak_time_std': summ_dim['time_to_peak'].std(),
        'peak_time_range': (summ_dim['time_to_peak'].min(), 
                           summ_dim['time_to_peak'].max()),
        'peak_high': summ_dim[summ_dim['dose']=='High']['peak_value'].mean(),
        'peak_low': summ_dim[summ_dim['dose']=='Low']['peak_value'].mean(),
        'baseline_rs': tc[tc['state']=='RS'][dimension].mean(),
        'cv': summ_dim['peak_value'].std() / summ_dim['peak_value'].mean()
    }
    
    return stats
```


### 9.4: LME Results Section

**Purpose**: Report significant fixed effects from linear mixed effects models.

**Implementation**:
- **Input Data**: `results/tet/lme/lme_coefficients_*.csv`
- **Method**: `generate_lme_section()`

**Organization**: Group by effect type, then list dimensions showing significant effects.

**Content Structure**:
1. **State Effects**: Dimensions with significant DMT vs RS differences
2. **Dose Effects**: Dimensions with significant High vs Low differences
3. **State:Dose Interactions**: Dimensions where dose effect differs by state
4. **Temporal Effects**: Dimensions with significant time trends

**Significance Threshold**: p_fdr < 0.05 (FDR-corrected)

**Output Format**:
```markdown
## 2. Linear Mixed Effects Results

### 2.1 State Effects (DMT vs RS)

Significant State effects (p_fdr < 0.05) were observed for 12 of 15 dimensions:

**Strong Effects (|β| > 1.5)**:
- **Complex Imagery**: β = 2.34, 95% CI [1.98, 2.70], p_fdr < 0.001
- **Elementary Imagery**: β = 2.12, 95% CI [1.81, 2.43], p_fdr < 0.001
- **Disembodiment**: β = 1.87, 95% CI [1.52, 2.22], p_fdr < 0.001

**Moderate Effects (0.8 < |β| < 1.5)**:
- **Pleasantness**: β = 1.23, 95% CI [0.94, 1.52], p_fdr < 0.001
- **Bliss**: β = 1.18, 95% CI [0.87, 1.49], p_fdr < 0.001
[...]

**Weak Effects (|β| < 0.8)**:
- **Anxiety**: β = 0.45, 95% CI [0.18, 0.72], p_fdr = 0.012
[...]

**Non-Significant**:
- Arousal: β = 0.12, 95% CI [-0.15, 0.39], p_fdr = 0.342
[...]

### 2.2 Dose Effects (High vs Low)

Significant main Dose effects (p_fdr < 0.05) were observed for 4 dimensions:

- **Complex Imagery**: β = 0.67, 95% CI [0.38, 0.96], p_fdr = 0.002
- **Disembodiment**: β = 0.54, 95% CI [0.26, 0.82], p_fdr = 0.008
- **Anxiety**: β = -0.48, 95% CI [-0.76, -0.20], p_fdr = 0.015 (negative = less anxiety at high dose)
- **Unpleasantness**: β = -0.42, 95% CI [-0.69, -0.15], p_fdr = 0.028

### 2.3 State:Dose Interaction Effects

Significant State:Dose interactions (p_fdr < 0.05) indicate that dose effects differ 
between DMT and RS conditions for 3 dimensions:

- **Complex Imagery**: β = 0.89, 95% CI [0.52, 1.26], p_fdr = 0.003
  (Dose effect stronger during DMT than RS)
- **Disembodiment**: β = 0.71, 95% CI [0.38, 1.04], p_fdr = 0.009
- **Anxiety**: β = -0.63, 95% CI [-1.01, -0.25], p_fdr = 0.018
  (Anxiolytic dose effect specific to DMT)

### 2.4 Temporal Effects

Significant Time effects (p_fdr < 0.05) were observed for 8 dimensions, indicating 
linear trends over the 0-9 minute analysis window:

[...]

[See Figure: lme_coefficients_forest.png for complete coefficient visualization]
```


### 9.5: Peak and AUC Analysis Section

**Purpose**: Report dose comparisons for intensity and duration metrics.

**Implementation**:
- **Input Data**: `results/tet/peak_auc/peak_comparison_results.csv`, `results/tet/peak_auc/auc_comparison_results.csv`
- **Method**: `generate_peak_auc_section()`

**Content Structure**:
1. **Peak Value Comparisons**: High vs Low dose peak intensity
2. **Time to Peak Analysis**: Dose effects on onset timing
3. **AUC Analysis**: Dose effects on cumulative experience
4. **Dose Sensitivity Rankings**: Dimensions ordered by effect size

**Output Format**:
```markdown
## 3. Peak and AUC Analysis

### 3.1 Peak Value Comparisons (High vs Low Dose)

Wilcoxon signed-rank tests comparing peak values within DMT sessions revealed 
significant dose effects (p_fdr < 0.05) for 6 dimensions:

**Increased at High Dose**:
- **Complex Imagery**: r = 0.58, 95% CI [0.32, 0.76], p_fdr = 0.002
- **Disembodiment**: r = 0.51, 95% CI [0.24, 0.71], p_fdr = 0.008
- **Elementary Imagery**: r = 0.47, 95% CI [0.19, 0.68], p_fdr = 0.015

**Decreased at High Dose**:
- **Anxiety**: r = -0.52, 95% CI [-0.71, -0.28], p_fdr = 0.003
- **Unpleasantness**: r = -0.44, 95% CI [-0.66, -0.18], p_fdr = 0.021
- **Confusion**: r = -0.39, 95% CI [-0.62, -0.12], p_fdr = 0.042

### 3.2 Time to Peak Analysis

No significant dose effects on time_to_peak were observed (all p_fdr > 0.05), 
indicating that onset timing is similar across doses despite differences in peak intensity.

### 3.3 Area Under Curve Analysis (0-9 minutes)

Significant dose effects on AUC (p_fdr < 0.05) for 5 dimensions:

- **Complex Imagery**: r = 0.61, 95% CI [0.36, 0.78], p_fdr = 0.001
- **Disembodiment**: r = 0.54, 95% CI [0.27, 0.73], p_fdr = 0.005
- **Anxiety**: r = -0.49, 95% CI [-0.69, -0.24], p_fdr = 0.009 (lower AUC at high dose)
- **Elementary Imagery**: r = 0.46, 95% CI [0.18, 0.67], p_fdr = 0.018
- **Unpleasantness**: r = -0.41, 95% CI [-0.64, -0.14], p_fdr = 0.035

### 3.4 Dose Sensitivity Rankings

Dimensions ranked by peak value effect size (|r|):

1. Complex Imagery (r = 0.58)
2. Anxiety (r = -0.52, decreased)
3. Disembodiment (r = 0.51)
4. Elementary Imagery (r = 0.47)
5. Unpleasantness (r = -0.44, decreased)

[See Figure: peak_auc_dose_comparison.png]
```

**Data Extraction**:
```python
def generate_peak_auc_section():
    """Generate Peak and AUC analysis section."""
    # Load results
    peak_results = pd.read_csv('results/tet/peak_auc/peak_comparison_results.csv')
    auc_results = pd.read_csv('results/tet/peak_auc/auc_comparison_results.csv')
    
    # Filter significant results
    sig_peak = peak_results[peak_results['p_fdr'] < 0.05].sort_values('effect_r', 
                                                                       key=abs, 
                                                                       ascending=False)
    sig_auc = auc_results[auc_results['p_fdr'] < 0.05].sort_values('effect_r', 
                                                                     key=abs, 
                                                                     ascending=False)
    
    # Generate formatted text
    section = format_peak_auc_findings(sig_peak, sig_auc)
    
    return section
```


### 9.6: Dimensionality Reduction Section

**Purpose**: Interpret principal components and their temporal dynamics.

**Implementation**:
- **Input Data**: `results/tet/pca/pca_loadings.csv`, `results/tet/pca/pca_variance_explained.csv`, `results/tet/pca/pca_lme_results.csv`
- **Method**: `generate_pca_section()`

**Content Structure**:
1. **Components Retained**: Number and cumulative variance explained
2. **PC1 Interpretation**: Dominant loading pattern and meaning
3. **PC2 Interpretation**: Secondary loading pattern and meaning
4. **Temporal Dynamics**: LME results for PC scores
5. **Dimension Contributions**: Top contributors to each PC

**Output Format**:
```markdown
## 4. Dimensionality Reduction

### 4.1 Principal Components Interpretation

PCA on z-scored experiential dimensions identified 3 principal components explaining 
76.4% of total variance:

- **PC1**: 42.3% variance
- **PC2**: 21.8% variance
- **PC3**: 12.3% variance

#### PC1: General Psychedelic Intensity

**Top Positive Loadings**:
- Complex Imagery: 0.42
- Elementary Imagery: 0.39
- Disembodiment: 0.37
- Bliss: 0.34

**Top Negative Loadings**:
- Anxiety: -0.28
- Unpleasantness: -0.24

**Interpretation**: PC1 represents a general psychedelic intensity factor, with high 
scores indicating strong imagery, ego dissolution, and positive affect, while low 
scores indicate anxiety and unpleasantness. This component captures the primary 
dimension of variation in DMT experiences.

#### PC2: Affective Valence

**Top Positive Loadings**:
- Pleasantness: 0.51
- Bliss: 0.48
- Insight: 0.36

**Top Negative Loadings**:
- Anxiety: -0.47
- Unpleasantness: -0.44
- Confusion: -0.32

**Interpretation**: PC2 represents affective valence, contrasting positive emotional 
experiences with negative affect and confusion. This component is orthogonal to 
intensity, indicating that positive and negative experiences can occur at any 
intensity level.

### 4.2 Variance Explained

[See Figure: pca_scree_plot.png]

The first two components explain 64.1% of variance, suggesting that experiential 
variation can be largely captured by intensity and valence dimensions.

### 4.3 Component Temporal Dynamics

LME models on PC scores revealed:

**PC1 (Intensity)**:
- State effect: β = 2.87, 95% CI [2.45, 3.29], p_fdr < 0.001
- Dose effect: β = 0.72, 95% CI [0.41, 1.03], p_fdr = 0.002
- State:Dose interaction: β = 0.94, 95% CI [0.55, 1.33], p_fdr = 0.001

**PC2 (Valence)**:
- State effect: β = 0.54, 95% CI [0.21, 0.87], p_fdr = 0.015
- Dose effect: β = -0.38, 95% CI [-0.67, -0.09], p_fdr = 0.032 (more positive at high dose)
- State:Dose interaction: β = -0.61, 95% CI [-0.98, -0.24], p_fdr = 0.012

### 4.4 Loading Patterns

[See Figure: pca_loadings_heatmap.png]

Dimensions cluster into three groups:
1. **Imagery/Dissolution**: Complex imagery, elementary imagery, disembodiment
2. **Positive Affect**: Pleasantness, bliss, insight
3. **Negative Affect**: Anxiety, unpleasantness, confusion
```


### 9.7: Clustering Analysis Section

**Purpose**: Characterize discrete experiential states and their dose sensitivity.

**Implementation**:
- **Input Data**: `results/tet/clustering/clustering_kmeans_assignments.csv`, `results/tet/clustering/clustering_metrics.csv`, `results/tet/clustering/state_occupancy.csv`
- **Method**: `generate_clustering_section()`

**Content Structure**:
1. **Optimal Solution**: Selected k value and quality metrics
2. **Cluster Characterization**: Dimension profiles for each cluster
3. **Temporal Prevalence**: When each cluster dominates
4. **Dose Effects**: Occupancy differences between High and Low
5. **Stability**: Bootstrap ARI scores

**Output Format**:
```markdown
## 5. Clustering Analysis

### 5.1 Optimal Cluster Solutions

**KMeans**: k = 2 selected based on highest silhouette score (0.42)
- k=2: silhouette = 0.42
- k=3: silhouette = 0.35
- k=4: silhouette = 0.29

**Stability**: Bootstrap ARI = 0.78 ± 0.09 (1000 iterations), indicating good stability.

### 5.2 Cluster Characterization

#### Cluster 0: "Baseline/Low Intensity State"

**Elevated Dimensions** (z > 0.5):
- Arousal: z = 0.82
- Selfhood: z = 0.61

**Suppressed Dimensions** (z < -0.5):
- Complex Imagery: z = -1.24
- Elementary Imagery: z = -1.18
- Disembodiment: z = -0.97
- Bliss: z = -0.73

**Interpretation**: This cluster represents baseline or low-intensity experiential 
states characterized by normal wakefulness and self-awareness with minimal imagery 
or altered states. Predominantly observed during RS and early DMT onset.

#### Cluster 1: "Peak Psychedelic State"

**Elevated Dimensions** (z > 0.5):
- Complex Imagery: z = 1.87
- Elementary Imagery: z = 1.72
- Disembodiment: z = 1.45
- Bliss: z = 1.23
- Pleasantness: z = 0.94
- Insight: z = 0.78

**Suppressed Dimensions** (z < -0.5):
- Anxiety: z = -0.68
- Selfhood: z = -0.54

**Interpretation**: This cluster represents peak psychedelic experiences with intense 
imagery, ego dissolution, and positive affect. Predominantly observed during DMT 
plateau phase (3-10 minutes).

[See Figure: clustering_kmeans_centroids_k2.png]

### 5.3 Temporal Prevalence Patterns

**Cluster 0 (Baseline)**:
- RS: 94.2% occupancy
- DMT Low (0-3 min): 78.3% → (3-10 min): 32.1% → (10-20 min): 61.4%
- DMT High (0-3 min): 71.2% → (3-10 min): 18.7% → (10-20 min): 54.8%

**Cluster 1 (Peak)**:
- RS: 5.8% occupancy
- DMT Low (0-3 min): 21.7% → (3-10 min): 67.9% → (10-20 min): 38.6%
- DMT High (0-3 min): 28.8% → (3-10 min): 81.3% → (10-20 min): 45.2%

**Pattern**: Cluster 1 dominates during DMT plateau (3-10 min), with higher occupancy 
at high dose. Return to Cluster 0 during offset phase (10-20 min).

[See Figure: clustering_kmeans_prob_timecourses_dmt_only.png]

### 5.4 Dose Effects on Cluster Occupancy

Paired t-tests comparing High vs Low dose occupancy within DMT sessions:

**Cluster 1 (Peak State)**:
- Fractional Occupancy: t(18) = 3.42, p_fdr = 0.006, d = 0.78
  (High dose: 0.58 ± 0.14, Low dose: 0.46 ± 0.16)
- Number of Visits: t(18) = 1.23, p_fdr = 0.234 (n.s.)
- Mean Dwell Time: t(18) = 2.87, p_fdr = 0.018, d = 0.66
  (High dose: 4.2 ± 1.8 min, Low dose: 3.1 ± 1.4 min)

**Interpretation**: High dose increases both the proportion of time spent in peak 
states and the duration of individual peak state episodes, but not the number of 
transitions between states.

### 5.5 Stability Metrics

Bootstrap resampling (1000 iterations) yielded:
- Mean ARI: 0.78
- SD: 0.09
- 95% CI: [0.62, 0.91]

This indicates good clustering stability, with consistent state assignments across 
resampled datasets.
```


### 9.8: Cross-Analysis Integration Section

**Purpose**: Identify convergent findings across multiple analytical approaches.

**Implementation**:
- **Method**: `generate_integration_section()`
- **Approach**: Compare dimension rankings and effect patterns across LME, Peak/AUC, PCA, and Clustering

**Content Structure**:
1. **Convergent Findings**: Effects replicated across methods
2. **Dimension Rankings**: Consistency in dose sensitivity
3. **Temporal Patterns**: Agreement on onset/offset dynamics
4. **Discrepancies**: Findings unique to specific methods

**Output Format**:
```markdown
## 6. Cross-Analysis Integration

### 6.1 Convergent Findings

The following findings were consistently observed across multiple analysis methods:

#### Complex Imagery: Strongest DMT Effect

- **LME**: Largest State effect (β = 2.34, p_fdr < 0.001)
- **Peak Analysis**: Highest peak values in DMT (8.2 vs 1.2 in RS)
- **PCA**: Highest loading on PC1 (0.42)
- **Clustering**: Strongest discriminator between Cluster 0 and 1 (Δz = 3.11)

**Conclusion**: Complex imagery is the most robust marker of DMT effects across all 
analytical approaches.

#### Dose-Dependent Anxiety Reduction

- **LME**: Negative Dose effect (β = -0.48, p_fdr = 0.015)
- **Peak Analysis**: Lower peaks at high dose (r = -0.52, p_fdr = 0.003)
- **AUC Analysis**: Lower cumulative anxiety at high dose (r = -0.49, p_fdr = 0.009)
- **Clustering**: Cluster 1 (low anxiety) more prevalent at high dose

**Conclusion**: High dose DMT consistently shows anxiolytic effects compared to low dose.

#### Rapid Onset, Sustained Plateau Pattern

- **Descriptive**: Peak at 2-4 minutes, plateau 3-10 minutes
- **LME**: Significant Time effects during 0-9 minute window
- **Clustering**: Transition to Cluster 1 by 3 minutes, sustained until 10 minutes

**Conclusion**: Temporal dynamics are consistent across analysis methods.

### 6.2 Dimension Rankings Across Methods

Dimensions ranked by effect size across four methods:

| Rank | LME State β | Peak Dose r | PCA PC1 Loading | Cluster Δz |
|------|-------------|-------------|-----------------|------------|
| 1    | Complex Imagery | Complex Imagery | Complex Imagery | Complex Imagery |
| 2    | Elementary Imagery | Anxiety* | Elementary Imagery | Elementary Imagery |
| 3    | Disembodiment | Disembodiment | Disembodiment | Disembodiment |
| 4    | Pleasantness | Elementary Imagery | Bliss | Bliss |
| 5    | Bliss | Unpleasantness* | Pleasantness | Pleasantness |

*Negative effects (decreased at high dose)

**Spearman Correlation** between ranking methods:
- LME vs Peak: ρ = 0.82, p < 0.001
- LME vs PCA: ρ = 0.78, p < 0.001
- LME vs Clustering: ρ = 0.85, p < 0.001

**Conclusion**: High consistency in dimension importance across methods.

### 6.3 Consistent Dose Effects

Five dimensions showed significant dose effects in at least 3 of 4 methods:

1. **Complex Imagery**: 4/4 methods (LME, Peak, AUC, Clustering)
2. **Disembodiment**: 4/4 methods
3. **Anxiety** (decreased): 4/4 methods
4. **Elementary Imagery**: 3/4 methods (LME n.s.)
5. **Unpleasantness** (decreased): 3/4 methods (Clustering n.s.)

### 6.4 Temporal Pattern Concordance

All methods agree on:
- **Onset**: 0-3 minutes (rapid increase)
- **Peak**: 2-4 minutes (maximum intensity)
- **Plateau**: 3-10 minutes (sustained effects)
- **Offset**: 10-20 minutes (gradual return to baseline)
- **Dose Effect Timing**: Present throughout plateau, strongest at 4-8 minutes
```

**Implementation**:
```python
def compute_cross_method_rankings():
    """Compute dimension rankings across methods."""
    # Load results from each method
    lme = load_lme_state_effects()
    peak = load_peak_dose_effects()
    pca = load_pca_loadings()
    clustering = load_cluster_discriminability()
    
    # Rank dimensions by effect size
    rankings = {
        'LME': lme.sort_values('beta', key=abs, ascending=False)['dimension'].tolist(),
        'Peak': peak.sort_values('effect_r', key=abs, ascending=False)['dimension'].tolist(),
        'PCA': pca.sort_values('pc1_loading', key=abs, ascending=False)['dimension'].tolist(),
        'Clustering': clustering.sort_values('delta_z', key=abs, ascending=False)['dimension'].tolist()
    }
    
    # Compute rank correlations
    correlations = compute_rank_correlations(rankings)
    
    return rankings, correlations
```


### 9.9: Methodological Notes Section

**Purpose**: Document data quality, assumptions, limitations, and analytical decisions.

**Implementation**:
- **Method**: `generate_methodological_notes()`
- **Sources**: Validation reports, analysis logs, model diagnostics

**Content Structure**:
```markdown
## 7. Methodological Notes

### 7.1 Data Quality

**Sample**: 19 subjects (S01-S20, excluding S14), 76 sessions total
- RS sessions: 38 (2 per subject)
- DMT sessions: 38 (2 per subject: 1 Low, 1 High)

**Data Completeness**: 
- Missing data: < 2% of time points (isolated missing ratings)
- Clamped values: 0.3% of ratings (outside 0-10 range)
- All subjects completed all 4 sessions

**Temporal Resolution**: 0.25 Hz (1 sample per 4 seconds)
- RS: 150 time points (0-10 minutes)
- DMT: 300 time points (0-20 minutes)

### 7.2 Model Assumptions

**Within-Subject Standardization**:
- Z-scores computed globally across all dimensions and sessions per subject
- Controls for individual differences in scale usage
- Assumes dimensions are commensurable within subject

**LME Models**:
- Random intercepts for subjects (no random slopes due to limited sessions)
- Assumes linear time trends (0-9 minute window)
- Residuals approximately normal (verified via Q-Q plots)
- Homoscedasticity assumption met for most dimensions

**Clustering**:
- KMeans assumes spherical clusters
- Euclidean distance metric
- No temporal dependencies modeled (static clustering)

### 7.3 Limitations

1. **Sample Size**: N=19 limits power for detecting small effects
2. **Session Count**: Only 2 sessions per condition limits random effects modeling
3. **Temporal Resolution**: 4-second sampling may miss rapid fluctuations
4. **Linear Time Models**: May not capture non-linear temporal dynamics
5. **Static Clustering**: Does not model state transitions or temporal dependencies
6. **Multiple Comparisons**: FDR correction applied, but family-wise error rate not controlled

### 7.4 Analytical Decisions

**Time Window Selection**:
- LME models: 0-9 minutes (captures onset and plateau)
- AUC: 0-9 minutes (excludes offset phase)
- Rationale: Focus on acute effects, avoid floor effects during offset

**PCA Retention**:
- Threshold: 70-80% cumulative variance
- Retained 3 components (76.4% variance)
- Rationale: Balance interpretability and variance explained

**Clustering k Selection**:
- Selected k=2 based on silhouette score
- Rationale: Parsimony and interpretability
- k=3 and k=4 solutions available for sensitivity analyses

**FDR Correction**:
- Applied separately for each effect type (State, Dose, Interaction)
- Family: 15 dimensions + 3 composite indices
- Rationale: Balance Type I and Type II error rates
```


### 9.10: Further Investigation Section

**Purpose**: Identify unresolved questions and suggest follow-up analyses.

**Implementation**:
- **Method**: `generate_further_investigation()`
- **Approach**: Systematic review of non-significant trends, ambiguous patterns, and unexplored questions

**Content Structure**:
```markdown
## 8. Further Investigation

### 8.1 Unresolved Questions

1. **Non-Linear Temporal Dynamics**
   - **Issue**: LME models assume linear time trends, but visual inspection suggests 
     non-linear patterns (rapid onset, plateau, gradual offset)
   - **Suggested Analysis**: Fit generalized additive models (GAMs) with smooth time 
     terms to capture non-linear trajectories
   - **Expected Insight**: Better characterization of onset speed and offset dynamics

2. **State Transition Dynamics**
   - **Issue**: KMeans provides static clustering without modeling temporal dependencies
   - **Suggested Analysis**: Implement GLHMM (Gaussian Linear Hidden Markov Models) to 
     model state transitions and dwell times
   - **Expected Insight**: Identify transition probabilities between experiential states 
     and dose effects on transition rates

3. **Individual Differences in Dose Response**
   - **Issue**: Current analyses focus on group-level effects; individual variability 
     not characterized
   - **Suggested Analysis**: Compute individual dose effect sizes and correlate with 
     baseline traits or session characteristics
   - **Expected Insight**: Identify predictors of dose sensitivity

4. **Dimension Interactions**
   - **Issue**: Analyses treat dimensions independently; potential interactions not explored
   - **Suggested Analysis**: Fit multivariate models or network analyses to identify 
     dimension co-variation patterns
   - **Expected Insight**: Understand how dimensions influence each other dynamically

### 8.2 Ambiguous Findings

1. **Arousal Effects**
   - **Observation**: Arousal showed non-significant State effect (β = 0.12, p_fdr = 0.342)
   - **Ambiguity**: Unclear if arousal is truly unaffected by DMT or if measurement is insensitive
   - **Suggested Analysis**: 
     - Review arousal rating instructions and subject interpretations
     - Correlate with physiological arousal measures (heart rate, EDA) if available
     - Consider non-linear arousal patterns (initial increase, then decrease)

2. **Time to Peak Null Results**
   - **Observation**: No dose effects on time_to_peak despite peak value differences
   - **Ambiguity**: Suggests onset timing is dose-independent, but low power possible
   - **Suggested Analysis**:
     - Increase temporal resolution around onset (0-5 minutes)
     - Use survival analysis methods for time-to-event modeling
     - Examine individual subject onset curves

3. **Cluster 0 Heterogeneity**
   - **Observation**: Cluster 0 includes both RS baseline and DMT offset periods
   - **Ambiguity**: May conflate distinct low-intensity states
   - **Suggested Analysis**:
     - Test k=3 solution to separate RS baseline, DMT onset, and DMT offset
     - Compare dimension profiles of Cluster 0 time points by session phase

### 8.3 Contradictory Results

1. **Pleasantness vs Bliss Dose Effects**
   - **Observation**: Bliss shows trend toward dose effect (p_fdr = 0.08) but 
     pleasantness does not (p_fdr = 0.21), despite high correlation (r = 0.82)
   - **Contradiction**: Conceptually similar dimensions show different dose sensitivity
   - **Suggested Analysis**:
     - Examine temporal profiles separately (may differ in timing)
     - Test if difference is due to measurement reliability
     - Consider affect_index_z as composite measure

2. **PCA vs Clustering Valence Patterns**
   - **Observation**: PC2 represents valence dimension, but clustering does not 
     separate positive/negative affect states
   - **Contradiction**: Suggests valence is secondary to intensity in state definition
   - **Suggested Analysis**:
     - Test k=3 or k=4 clustering to see if valence-based clusters emerge
     - Examine PC2 scores within each cluster
     - Consider hierarchical clustering (intensity first, then valence)

### 8.4 Suggested Follow-up Analyses

#### High Priority

1. **Generalized Additive Models (GAMs)**
   - **Rationale**: Capture non-linear temporal dynamics
   - **Implementation**: Fit GAMs with smooth time terms for each dimension
   - **Expected Effort**: 2-3 days
   - **Dependencies**: mgcv R package or pygam Python package

2. **GLHMM State Modeling**
   - **Rationale**: Model temporal state transitions
   - **Implementation**: Fit GLHMM with 2-4 states, compare to KMeans
   - **Expected Effort**: 1-2 weeks (includes learning GLHMM framework)
   - **Dependencies**: glhmm Python package, across-sessions-within-subject permutation tests

3. **Individual Difference Analysis**
   - **Rationale**: Characterize dose response variability
   - **Implementation**: Compute individual effect sizes, correlate with traits
   - **Expected Effort**: 3-5 days
   - **Dependencies**: Baseline trait data (if available)

#### Medium Priority

4. **Multivariate Time Series Analysis**
   - **Rationale**: Understand dimension co-variation
   - **Implementation**: Vector autoregression (VAR) or dynamic factor models
   - **Expected Effort**: 1 week
   - **Dependencies**: statsmodels or specialized time series packages

5. **Sensitivity Analyses**
   - **Rationale**: Test robustness to analytical decisions
   - **Implementation**: Vary time windows, FDR thresholds, clustering k values
   - **Expected Effort**: 2-3 days
   - **Dependencies**: None (re-run existing analyses with different parameters)

6. **Physiological Correlates**
   - **Rationale**: Validate subjective reports with objective measures
   - **Implementation**: Correlate TET dimensions with heart rate, EDA, EEG
   - **Expected Effort**: 1-2 weeks
   - **Dependencies**: Physiological data availability and preprocessing

#### Low Priority

7. **Network Analysis**
   - **Rationale**: Visualize dimension relationships
   - **Implementation**: Compute partial correlations, construct network graphs
   - **Expected Effort**: 3-5 days
   - **Dependencies**: networkx, graph visualization tools

8. **Machine Learning Prediction**
   - **Rationale**: Predict dose or state from dimension patterns
   - **Implementation**: Train classifiers (SVM, random forest) on TET data
   - **Expected Effort**: 1 week
   - **Dependencies**: scikit-learn

### 8.5 Data Collection Recommendations

For future studies, consider:

1. **Increased Temporal Resolution**: 1 Hz sampling (every second) during onset phase (0-5 min)
2. **Additional Dimensions**: Measure dimensions not captured (e.g., time perception, body sensations)
3. **Qualitative Reports**: Post-session interviews to contextualize quantitative ratings
4. **Physiological Measures**: Concurrent heart rate, EDA, pupillometry
5. **Baseline Traits**: Personality, trait anxiety, prior psychedelic experience
6. **More Dose Levels**: 3-4 dose levels to characterize dose-response curves
7. **Longer Follow-up**: Extend rating period to 30-40 minutes to capture full offset
```


## Implementation Details

### Module Structure

```
scripts/
├── generate_comprehensive_report.py    # Main script
└── tet/
    ├── results_synthesizer.py          # Core synthesis class
    ├── results_formatter.py             # Markdown formatting utilities
    └── results_analyzer.py              # Statistical analysis utilities
```

### Core Class: TETResultsSynthesizer

```python
class TETResultsSynthesizer:
    """Synthesize TET analysis results into comprehensive report."""
    
    def __init__(self, results_dir: str = 'results/tet'):
        self.results_dir = Path(results_dir)
        self.output_path = Path('docs/tet_comprehensive_results.md')
        
        # Data containers
        self.descriptive_data = None
        self.lme_results = None
        self.peak_auc_results = None
        self.pca_results = None
        self.clustering_results = None
        
    def load_all_results(self):
        """Load all analysis outputs."""
        self.descriptive_data = self._load_descriptive()
        self.lme_results = self._load_lme()
        self.peak_auc_results = self._load_peak_auc()
        self.pca_results = self._load_pca()
        self.clustering_results = self._load_clustering()
        
    def generate_report(self):
        """Generate comprehensive results document."""
        sections = []
        
        # Header
        sections.append(self._generate_header())
        
        # Executive Summary (9.2)
        sections.append(self.generate_executive_summary())
        
        # Descriptive Statistics (9.3)
        sections.append(self.generate_descriptive_section())
        
        # LME Results (9.4)
        sections.append(self.generate_lme_section())
        
        # Peak and AUC (9.5)
        sections.append(self.generate_peak_auc_section())
        
        # PCA (9.6)
        sections.append(self.generate_pca_section())
        
        # Clustering (9.7)
        sections.append(self.generate_clustering_section())
        
        # Cross-Analysis Integration (9.8)
        sections.append(self.generate_integration_section())
        
        # Methodological Notes (9.9)
        sections.append(self.generate_methodological_notes())
        
        # Further Investigation (9.10)
        sections.append(self.generate_further_investigation())
        
        # Appendix
        sections.append(self._generate_appendix())
        
        # Write document
        report_text = '\n\n'.join(sections)
        self.output_path.write_text(report_text, encoding='utf-8')
        
        return self.output_path
```

### Statistical Notation Formatter

```python
class StatisticalFormatter:
    """Format statistical results with consistent notation."""
    
    @staticmethod
    def format_lme_result(beta, ci_lower, ci_upper, p_fdr):
        """Format LME coefficient: β = X.XX, 95% CI [X.XX, X.XX], p_fdr = X.XXX"""
        return f"β = {beta:.2f}, 95% CI [{ci_lower:.2f}, {ci_upper:.2f}], p_fdr = {p_fdr:.3f}"
    
    @staticmethod
    def format_effect_size(r, ci_lower, ci_upper, p_fdr):
        """Format effect size: r = X.XX, 95% CI [X.XX, X.XX], p_fdr = X.XXX"""
        return f"r = {r:.2f}, 95% CI [{ci_lower:.2f}, {ci_upper:.2f}], p_fdr = {p_fdr:.3f}"
    
    @staticmethod
    def format_p_value(p):
        """Format p-value with appropriate precision."""
        if p < 0.001:
            return "p < 0.001"
        else:
            return f"p = {p:.3f}"
    
    @staticmethod
    def format_figure_reference(figure_name):
        """Format figure reference with relative path."""
        return f"[See Figure: ../results/tet/figures/{figure_name}]"
```

### Automation and Triggers

**Trigger Conditions**:
1. **After Full Pipeline**: When all analysis components complete
2. **After Component Update**: When any analysis is re-run
3. **Manual Trigger**: Via command-line script

**Implementation**:
```python
# In main analysis pipeline script
def run_tet_pipeline():
    """Run complete TET analysis pipeline."""
    # ... run all analyses ...
    
    # Generate comprehensive report
    synthesizer = TETResultsSynthesizer()
    synthesizer.load_all_results()
    report_path = synthesizer.generate_report()
    
    print(f"Comprehensive results report generated: {report_path}")
```

**Manual Trigger**:
```bash
# Regenerate report from existing results
python scripts/generate_comprehensive_report.py --results-dir results/tet --output docs/tet_comprehensive_results.md
```


## Data Flow

```
Analysis Outputs (CSV files)
├── descriptive/
│   ├── timecourse_*.csv
│   └── session_summaries.csv
├── lme/
│   └── lme_coefficients_*.csv
├── peak_auc/
│   ├── peak_comparison_results.csv
│   └── auc_comparison_results.csv
├── pca/
│   ├── pca_loadings.csv
│   ├── pca_variance_explained.csv
│   └── pca_lme_results.csv
└── clustering/
    ├── clustering_kmeans_assignments.csv
    ├── clustering_metrics.csv
    └── state_occupancy.csv
    
    ↓
    
TETResultsSynthesizer
├── Load all results
├── Validate completeness
├── Extract key findings
├── Rank by importance
├── Identify patterns
└── Detect ambiguities
    
    ↓
    
Markdown Document Generation
├── Format statistical notation
├── Create hierarchical sections
├── Link to figures/tables
└── Write to docs/tet_comprehensive_results.md
    
    ↓
    
Output: docs/tet_comprehensive_results.md
```

## Testing Strategy

### Unit Tests

```python
# tests/test_results_synthesizer.py

def test_load_all_results():
    """Test loading all analysis outputs."""
    synthesizer = TETResultsSynthesizer('tests/fixtures/results')
    synthesizer.load_all_results()
    
    assert synthesizer.descriptive_data is not None
    assert synthesizer.lme_results is not None
    assert synthesizer.peak_auc_results is not None
    assert synthesizer.pca_results is not None
    assert synthesizer.clustering_results is not None

def test_executive_summary_generation():
    """Test executive summary contains 3-5 findings."""
    synthesizer = TETResultsSynthesizer('tests/fixtures/results')
    synthesizer.load_all_results()
    
    summary = synthesizer.generate_executive_summary()
    
    # Check structure
    assert '## Executive Summary' in summary
    assert 'key findings' in summary.lower()
    
    # Check number of findings (should be 3-5)
    finding_count = summary.count('\n1.') + summary.count('\n2.') + summary.count('\n3.')
    assert 3 <= finding_count <= 5

def test_statistical_notation_consistency():
    """Test consistent statistical notation formatting."""
    formatter = StatisticalFormatter()
    
    # LME result
    lme_str = formatter.format_lme_result(beta=2.34, ci_lower=1.98, ci_upper=2.70, p_fdr=0.0001)
    assert 'β = 2.34' in lme_str
    assert '95% CI [1.98, 2.70]' in lme_str
    assert 'p_fdr' in lme_str
    
    # Effect size
    es_str = formatter.format_effect_size(r=0.52, ci_lower=0.28, ci_upper=0.71, p_fdr=0.003)
    assert 'r = 0.52' in es_str
    assert '95% CI [0.28, 0.71]' in es_str

def test_figure_references():
    """Test figure references use correct paths."""
    formatter = StatisticalFormatter()
    
    ref = formatter.format_figure_reference('timeseries_pleasantness.png')
    assert '../results/tet/figures/timeseries_pleasantness.png' in ref
    assert '[See Figure:' in ref

def test_report_regeneration():
    """Test report regenerates consistently from same inputs."""
    synthesizer = TETResultsSynthesizer('tests/fixtures/results')
    
    # Generate twice
    report1 = synthesizer.generate_report()
    report2 = synthesizer.generate_report()
    
    # Should be identical
    assert report1.read_text() == report2.read_text()
```

### Integration Tests

```python
def test_full_report_generation():
    """Test complete report generation from real results."""
    synthesizer = TETResultsSynthesizer('results/tet')
    synthesizer.load_all_results()
    
    report_path = synthesizer.generate_report()
    
    # Check file exists
    assert report_path.exists()
    
    # Check file size (should be substantial)
    assert report_path.stat().st_size > 10000  # At least 10KB
    
    # Check all required sections present
    content = report_path.read_text()
    assert '## Executive Summary' in content
    assert '## 1. Descriptive Statistics' in content
    assert '## 2. Linear Mixed Effects Results' in content
    assert '## 3. Peak and AUC Analysis' in content
    assert '## 4. Dimensionality Reduction' in content
    assert '## 5. Clustering Analysis' in content
    assert '## 6. Cross-Analysis Integration' in content
    assert '## 7. Methodological Notes' in content
    assert '## 8. Further Investigation' in content

def test_missing_data_handling():
    """Test graceful handling of missing analysis outputs."""
    # Create results dir with only some outputs
    test_dir = 'tests/fixtures/incomplete_results'
    
    synthesizer = TETResultsSynthesizer(test_dir)
    
    # Should not crash
    synthesizer.load_all_results()
    
    # Should generate report with available data
    report_path = synthesizer.generate_report()
    assert report_path.exists()
    
    # Should note missing analyses
    content = report_path.read_text()
    assert 'not available' in content.lower() or 'missing' in content.lower()
```

### Validation Checklist

Manual review checklist for generated reports:

- [ ] All 8 main sections present (Executive Summary through Further Investigation)
- [ ] Executive Summary contains 3-5 key findings with effect sizes
- [ ] Statistical notation consistent throughout (β, r, p_fdr, CI)
- [ ] All figure references use correct relative paths
- [ ] Dimension names formatted consistently (no '_z' suffixes in text)
- [ ] P-values formatted correctly (< 0.001 vs exact values)
- [ ] Effect sizes include confidence intervals
- [ ] Hierarchical organization clear (sections, subsections, bullets)
- [ ] Cross-references between sections accurate
- [ ] Further Investigation section actionable and specific
- [ ] No broken links to figures or tables
- [ ] Markdown syntax valid (renders correctly)


## Dependencies

### Python Libraries

```python
# Core data processing
pandas >= 1.3.0
numpy >= 1.21.0

# Statistical utilities
scipy >= 1.7.0  # For rank correlations, effect size computations

# Path handling
pathlib  # Standard library

# Optional (for enhanced formatting)
markdown >= 3.3.0  # For markdown validation
```

### Input Data Requirements

**Required Files**:
- `results/tet/descriptive/timecourse_*.csv` (one per dimension)
- `results/tet/descriptive/session_summaries.csv`
- `results/tet/lme/lme_coefficients_*.csv`
- `results/tet/peak_auc/peak_comparison_results.csv`
- `results/tet/peak_auc/auc_comparison_results.csv`
- `results/tet/pca/pca_loadings.csv`
- `results/tet/pca/pca_variance_explained.csv`
- `results/tet/clustering/clustering_kmeans_assignments.csv`
- `results/tet/clustering/clustering_metrics.csv`

**Optional Files** (gracefully handled if missing):
- `results/tet/pca/pca_lme_results.csv`
- `results/tet/clustering/state_occupancy.csv`
- `results/tet/clustering/clustering_glhmm_*.csv` (GLHMM results)

### Output Requirements

**Output File**: `docs/tet_comprehensive_results.md`
- Format: UTF-8 encoded markdown
- Size: Typically 50-100 KB
- Line endings: Unix-style (LF)

## Error Handling

### Missing Input Files

**Strategy**: Graceful degradation with informative messages

```python
def _load_lme(self):
    """Load LME results with error handling."""
    lme_path = self.results_dir / 'lme' / 'lme_coefficients_all_dimensions.csv'
    
    if not lme_path.exists():
        logger.warning(f"LME results not found: {lme_path}")
        return None
    
    try:
        return pd.read_csv(lme_path)
    except Exception as e:
        logger.error(f"Error loading LME results: {e}")
        return None

def generate_lme_section(self):
    """Generate LME section with missing data handling."""
    if self.lme_results is None:
        return """
## 2. Linear Mixed Effects Results

**Note**: LME analysis results are not available. This section will be populated 
when LME models are fitted.
"""
    
    # ... normal section generation ...
```

### Invalid Data

**Strategy**: Validation with informative error messages

```python
def _validate_lme_results(self, df):
    """Validate LME results dataframe."""
    required_cols = ['dimension', 'effect', 'beta', 'ci_lower', 'ci_upper', 'p_fdr']
    
    missing_cols = set(required_cols) - set(df.columns)
    if missing_cols:
        raise ValueError(f"LME results missing columns: {missing_cols}")
    
    # Check for NaN values
    if df[required_cols].isna().any().any():
        logger.warning("LME results contain NaN values")
        df = df.dropna(subset=required_cols)
    
    # Check value ranges
    if (df['p_fdr'] < 0).any() or (df['p_fdr'] > 1).any():
        raise ValueError("Invalid p_fdr values (must be 0-1)")
    
    return df
```

### Report Generation Failures

**Strategy**: Partial report generation with error logging

```python
def generate_report(self):
    """Generate report with error handling."""
    sections = []
    errors = []
    
    try:
        sections.append(self._generate_header())
    except Exception as e:
        errors.append(('Header', str(e)))
    
    try:
        sections.append(self.generate_executive_summary())
    except Exception as e:
        errors.append(('Executive Summary', str(e)))
        sections.append("## Executive Summary\n\n**Error generating summary**")
    
    # ... repeat for all sections ...
    
    # Add error report if any failures
    if errors:
        error_section = "\n## Report Generation Errors\n\n"
        for section, error in errors:
            error_section += f"- **{section}**: {error}\n"
        sections.append(error_section)
    
    # Write report
    report_text = '\n\n'.join(sections)
    self.output_path.write_text(report_text, encoding='utf-8')
    
    return self.output_path
```

## Future Enhancements

### Interactive HTML Version

**Rationale**: Enhance navigation and interactivity

**Features**:
- Collapsible sections
- Interactive plots (Plotly)
- Search functionality
- Table of contents with jump links
- Export to PDF

**Implementation**:
```python
def generate_html_report(self):
    """Generate interactive HTML version."""
    import markdown
    from jinja2 import Template
    
    # Convert markdown to HTML
    md_text = self.output_path.read_text()
    html_content = markdown.markdown(md_text, extensions=['tables', 'toc'])
    
    # Apply template
    template = Template(HTML_TEMPLATE)
    html_output = template.render(content=html_content)
    
    # Write HTML
    html_path = self.output_path.with_suffix('.html')
    html_path.write_text(html_output, encoding='utf-8')
    
    return html_path
```

### Automated Interpretation

**Rationale**: Provide AI-assisted interpretation of patterns

**Features**:
- Automated pattern detection
- Natural language summaries
- Hypothesis generation
- Literature contextualization

**Implementation**: Integration with LLM APIs for automated interpretation

### Version Comparison

**Rationale**: Track how results change across analysis iterations

**Features**:
- Diff between report versions
- Highlight changed findings
- Track effect size changes
- Version history

**Implementation**:
```python
def compare_reports(old_path, new_path):
    """Compare two report versions."""
    import difflib
    
    old_text = old_path.read_text().splitlines()
    new_text = new_path.read_text().splitlines()
    
    diff = difflib.unified_diff(old_text, new_text, lineterm='')
    
    return '\n'.join(diff)
```

## Summary

### Key Design Decisions

1. **Markdown Format**: Human-readable, version-controllable, easily convertible
2. **Automated Generation**: Triggered after analysis completion
3. **Graceful Degradation**: Handles missing data without crashing
4. **Consistent Notation**: Standardized statistical formatting throughout
5. **Hierarchical Organization**: Clear section structure for rapid navigation
6. **Actionable Insights**: Explicit identification of follow-up analyses

### Success Criteria

Report generation is successful when:
- ✅ All 15 acceptance criteria (9.1-9.15) met
- ✅ Document generated automatically after pipeline completion
- ✅ All available results synthesized and interpreted
- ✅ Statistical notation consistent throughout
- ✅ Figure references valid and accessible
- ✅ Further investigation section actionable
- ✅ Report regenerates consistently from same inputs
- ✅ Missing data handled gracefully

### Implementation Priorities

**Phase 1: Core Functionality** (Required)
1. Implement TETResultsSynthesizer class
2. Create section generation methods (9.2-9.10)
3. Implement statistical formatting utilities
4. Add figure reference management
5. Test with real data

**Phase 2: Automation** (Required)
1. Integrate with main pipeline script
2. Add command-line interface
3. Implement update detection
4. Add logging and error handling

**Phase 3: Enhancements** (Optional)
1. HTML version generation
2. Version comparison tools
3. Automated interpretation features
4. Interactive visualizations

### Current Status

**Status**: Design Complete, Ready for Implementation

**Next Steps**:
1. Create `scripts/tet/results_synthesizer.py` module
2. Implement core TETResultsSynthesizer class
3. Create section generation methods
4. Add unit tests
5. Test with real TET analysis outputs
6. Integrate with main pipeline
7. Document usage in README

---

**Document Version**: 1.0  
**Last Updated**: 2025-11-14  
**Status**: Design Complete - Ready for Implementation
