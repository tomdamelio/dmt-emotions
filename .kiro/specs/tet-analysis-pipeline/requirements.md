# Requirements Document: TET Analysis Pipeline

## Introduction

This document specifies the requirements for implementing a comprehensive analysis pipeline for Temporal Experience Tracking (TET) data in a DMT psychopharmacology study. The TET_Analysis_System processes subjective experience ratings collected at 30-second intervals across multiple dimensions (pleasantness, anxiety, imagery, etc.) during resting state (RS) and DMT sessions with varying doses (Low/High). The system performs preprocessing, statistical modeling, dimensionality reduction, and generates publication-ready outputs including tables and figures.

## Glossary

- **TET_Analysis_System**: The complete software pipeline for processing and analyzing temporal experience tracking data
- **TET_Data**: Time-series subjective experience ratings on 15 dimensions collected at 30-second intervals
- **Subject**: Individual participant identified by code (S01-S20, excluding S14)
- **Session**: One experimental recording period (4 sessions per subject: 2 RS, 2 DMT)
- **State**: Experimental condition, either RS (resting state) or DMT (drug condition)
- **Dose**: DMT dosage level, either Low or High
- **Time_Point**: Discrete time point sampled at 0.25 Hz (1 point every 4 seconds), indexed from 0
- **Dimension**: One of 15 subjective experience scales rated 0-10
- **Composite_Index**: Derived metric combining multiple dimensions (affect, imagery, self)
- **Within_Subject_Standardization**: Z-score transformation computed separately for each subject across all their sessions
- **LME_Model**: Linear Mixed Effects statistical model with random subject intercepts
- **BH_FDR**: Benjamini-Hochberg False Discovery Rate correction for multiple comparisons
- **AUC**: Area Under the Curve, integral of a time series
- **PCA_Component**: Principal component from dimensionality reduction
- **Cluster_State**: Discrete experiential state identified by clustering algorithm

## Requirements

### Requirement 1: Data Loading and Validation

**User Story:** As a researcher, I want the system to load TET data from standardized files and validate data integrity, so that I can ensure data quality before analysis.

#### Acceptance Criteria

1. WHEN THE TET_Analysis_System receives a file path, THE TET_Analysis_System SHALL load TET_Data containing columns for subject, session_id, state, dose, t_bin, t_sec, and 15 dimension ratings.

2. THE TET_Analysis_System SHALL validate that each RS session contains exactly 150 Time_Points (600s @ 0.25 Hz) and each DMT session contains exactly 300 Time_Points (1200s @ 0.25 Hz).

3. THE TET_Analysis_System SHALL validate that all dimension ratings fall within the range 0 to 10 inclusive.

4. IF any dimension rating falls outside the range 0 to 10, THEN THE TET_Analysis_System SHALL clamp the value to the nearest boundary and log the adjustment.

5. THE TET_Analysis_System SHALL validate that each Subject has exactly 4 sessions (2 RS and 2 DMT).

6. THE TET_Analysis_System SHALL generate a validation report documenting the number of subjects, sessions, time bins, and any data quality issues detected.

### Requirement 2: Data Preprocessing and Standardization

**User Story:** As a researcher, I want the system to preprocess and standardize TET data within subjects, so that individual differences in scale usage are controlled.

#### Acceptance Criteria

1. THE TET_Analysis_System SHALL trim RS sessions to the first 600 seconds (0-10 minutes) and DMT sessions to the first 1200 seconds (0-20 minutes), preserving the original 0.25 Hz sampling rate.

2. THE TET_Analysis_System SHALL create valence variables where valence_pos equals pleasantness and valence_neg equals unpleasantness.

3. THE TET_Analysis_System SHALL compute Within_Subject_Standardization by calculating z-scores for all 15 Dimensions jointly, using the global mean and standard deviation computed across all dimensions and all 4 sessions for that Subject, thereby controlling for individual differences in scale usage patterns.

4. THE TET_Analysis_System SHALL preserve original raw scale values in separate columns for descriptive reporting.

5. THE TET_Analysis_System SHALL compute affect_index_z as the mean of z-scored pleasantness and bliss minus the mean of z-scored anxiety and unpleasantness.

6. THE TET_Analysis_System SHALL compute imagery_index_z as the mean of z-scored elementary_imagery and complex_imagery.

7. THE TET_Analysis_System SHALL compute self_index_z by inverting z-scored disembodiment and combining with z-scored selfhood such that higher values indicate greater self-integration.

8. THE TET_Analysis_System SHALL document the composition and directionality of each Composite_Index in metadata.

### Requirement 3: Descriptive Statistics and Time Course Analysis

**User Story:** As a researcher, I want the system to compute descriptive statistics and generate time course summaries, so that I can characterize the temporal dynamics of subjective experiences.

#### Acceptance Criteria

1. THE TET_Analysis_System SHALL compute mean and standard error of the mean (SEM) for each Dimension at each Time_Bin, grouped by State and Dose.

2. THE TET_Analysis_System SHALL export time course data as CSV files with columns for t_bin, state, dose, dimension, mean, and sem.

3. THE TET_Analysis_System SHALL compute session-level summary metrics including peak value, time_to_peak, AUC from 0-9 minutes, slope from 0-2 minutes, and slope from 5-9 minutes for each Dimension.

4. THE TET_Analysis_System SHALL compute summary metrics separately for each Subject, Session, State, and Dose combination.

5. THE TET_Analysis_System SHALL export session-level summary metrics as CSV files with one row per session and columns for each metric.

### Requirement 4: Linear Mixed Effects Modeling

**User Story:** As a researcher, I want the system to fit LME models to test dose and state effects on each dimension, so that I can identify statistically significant experimental effects.

#### Acceptance Criteria

1. THE TET_Analysis_System SHALL fit an LME_Model for each z-scored Dimension using data from Time_Bins 0-18 (0-9 minutes).

2. THE TET_Analysis_System SHALL include fixed effects for State, Dose, Time_c (centered time), State:Dose, State:Time_c, and Dose:Time_c in each LME_Model.

3. THE TET_Analysis_System SHALL include a random intercept for Subject in each LME_Model.

4. THE TET_Analysis_System SHALL estimate LME_Model parameters using maximum likelihood estimation.

5. THE TET_Analysis_System SHALL extract coefficients, 95% confidence intervals, and p-values for each fixed effect.

6. THE TET_Analysis_System SHALL apply BH_FDR correction across the family of 15 Dimensions for each fixed effect separately.

7. THE TET_Analysis_System SHALL compute contrasts for High vs Low Dose within DMT State and within RS State.

8. THE TET_Analysis_System SHALL export LME results as CSV files with columns for dimension, effect, beta, ci_lower, ci_upper, p_value, and p_fdr.

### Requirement 4b: Time Series Visualization with Statistical Annotations

**User Story:** As a researcher, I want the system to generate time series plots showing dose effects with statistical annotations, so that I can visualize the temporal dynamics and identify significant differences between conditions.

#### Acceptance Criteria

1. THE TET_Analysis_System SHALL generate time series plots for each Dimension showing mean trajectories with standard error shading for High_Dose (40mg) and Low_Dose (20mg) conditions.

2. THE TET_Analysis_System SHALL display the RS_Baseline as the first time point, computed as the mean across the entire RS condition duration, with a grey dashed vertical line indicating DMT onset.

3. THE TET_Analysis_System SHALL annotate time points where DMT differs significantly from RS_Baseline (regardless of dose) using grey background shading.

4. THE TET_Analysis_System SHALL annotate time points with significant State:Dose interaction effects using black horizontal bars, indicating that the DMT effect is moderated by dose.

5. THE TET_Analysis_System SHALL order dimensions in the figure by the strength of the main State effect coefficient from LME models, with strongest effects displayed first.

6. THE TET_Analysis_System SHALL use consistent color coding: blue for Low_Dose (20mg), red for High_Dose (40mg), and grey for RS_Baseline.

7. THE TET_Analysis_System SHALL display time on the x-axis in minutes and z-scored intensity on the y-axis.

8. THE TET_Analysis_System SHALL export the figure as a high-resolution PNG file with dimensions suitable for publication (e.g., 300 DPI, 12×16 inches).

### Requirement 5: Peak and AUC Analysis

**User Story:** As a researcher, I want the system to compare peak values and AUC between dose conditions using non-parametric tests, so that I can assess dose effects on experience intensity and duration.

#### Acceptance Criteria

1. THE TET_Analysis_System SHALL perform Wilcoxon signed-rank tests comparing High vs Low Dose for peak values within DMT sessions for each Dimension.

2. THE TET_Analysis_System SHALL perform Wilcoxon signed-rank tests comparing High vs Low Dose for time_to_peak within DMT sessions for each Dimension.

3. THE TET_Analysis_System SHALL perform Wilcoxon signed-rank tests comparing High vs Low Dose for AUC_0_9 within DMT sessions for each Dimension.

4. THE TET_Analysis_System SHALL apply BH_FDR correction across 15 Dimensions and 3 Composite_Indices for each metric type.

5. THE TET_Analysis_System SHALL compute effect size r for each Wilcoxon test with 95% confidence intervals using bootstrap resampling with 2000 iterations.

6. THE TET_Analysis_System SHALL export peak and AUC test results as CSV files with columns for dimension, metric, statistic, p_value, p_fdr, effect_r, ci_lower, and ci_upper.

### Requirement 6: Dimensionality Reduction via PCA

**User Story:** As a researcher, I want the system to perform PCA on standardized TET data, so that I can identify the principal modes of experiential variation.

#### Acceptance Criteria

1. THE TET_Analysis_System SHALL perform PCA on z-scored Dimension values across all Time_Bins within each Subject.

2. THE TET_Analysis_System SHALL retain PCA_Components that cumulatively explain 70-80% of variance.

3. THE TET_Analysis_System SHALL fit LME_Models for PC1 and PC2 scores with fixed effects for State, Dose, Time_c, and their interactions, and random Subject intercepts.

4. THE TET_Analysis_System SHALL export PCA loadings as CSV files with columns for dimension and loading values for each retained component.

5. THE TET_Analysis_System SHALL export PCA variance explained as CSV with columns for component and variance_explained_ratio.

6. THE TET_Analysis_System SHALL export PC1 and PC2 LME results following the same format as Requirement 4.

### Requirement 7: Clustering of Experiential States

**User Story:** As a researcher, I want the system to identify discrete experiential states using clustering algorithms, so that I can characterize qualitatively distinct experience patterns.

#### Acceptance Criteria

1. THE TET_Analysis_System SHALL apply KMeans clustering to z-scored Time_Bin observations with k values of 2, 3, and 4.

2. THE TET_Analysis_System SHALL apply Gaussian Mixture Model clustering to z-scored Time_Bin observations with k values of 2, 3, and 4.

3. THE TET_Analysis_System SHALL compute silhouette scores and BIC values for each clustering solution.

4. THE TET_Analysis_System SHALL select the optimal k value based on the highest silhouette score and lowest BIC.

5. THE TET_Analysis_System SHALL assess clustering stability by performing bootstrap resampling with 1000 iterations and computing adjusted Rand index across bootstrap samples.

6. THE TET_Analysis_System SHALL compute cluster probability for each Time_Bin as the posterior probability of cluster membership.

7. THE TET_Analysis_System SHALL perform paired t-tests comparing High vs Low Dose cluster probabilities within DMT at each Time_Bin with BH_FDR correction across time bins.

8. THE TET_Analysis_System SHALL export cluster assignments and probabilities as CSV files with columns for subject, session, t_bin, cluster, and probability.

### Requirement 8: Figure Generation

**User Story:** As a researcher, I want the system to generate publication-ready figures, so that I can visualize and communicate analysis results effectively.

#### Acceptance Criteria

1. THE TET_Analysis_System SHALL generate annotated time series plots showing mean ± SEM trajectories for each Dimension with High_Dose and Low_Dose conditions, including RS_Baseline reference, grey shading for time bins where DMT differs significantly from baseline, and black bars indicating significant State:Dose interactions.

2. THE TET_Analysis_System SHALL generate coefficient plots for LME fixed effects showing beta estimates with 95% confidence intervals for each Dimension, with markers indicating p_fdr significance levels and effects ordered by strength of State main effect.

3. THE TET_Analysis_System SHALL generate boxplots comparing High vs Low Dose for peak values, time_to_peak, and AUC_0_9 within DMT sessions, with annotations showing Wilcoxon test p_fdr values and effect sizes with bootstrap confidence intervals.

4. THE TET_Analysis_System SHALL generate PCA scree plots showing variance explained by each component with cumulative variance line, indicating the threshold used for component retention.

5. THE TET_Analysis_System SHALL generate PCA loading heatmaps or bar charts showing dimension contributions to retained components, with dimensions ordered by loading magnitude to facilitate interpretation.

6. THE TET_Analysis_System SHALL generate KMeans centroid profile plots showing normalized dimension contributions for each cluster, replicating the format of preliminary analysis figures (Fig. 3.5) to enable direct comparison.

7. THE TET_Analysis_System SHALL generate KMeans cluster probability time course plots showing mean ± SEM probability trajectories for each Cluster_State over time, separated by State and Dose, to visualize temporal dynamics of experiential states and replicate preliminary analysis (Fig. 3.6).

8. WHERE GLHMM temporal state modelling is implemented, THE TET_Analysis_System SHALL generate GLHMM state probability time course plots and KMeans-GLHMM correspondence heatmaps to compare static and temporal state identification approaches.

9. THE TET_Analysis_System SHALL save all figures as PNG files with resolution of at least 300 DPI in the results/tet/figures/ directory with descriptive filenames indicating content.


### Requirement 9: Comprehensive Results Synthesis and Reporting

**User Story:** As a researcher, I want the system to generate a comprehensive results document synthesizing all TET analysis findings, so that I can understand the main outcomes, identify patterns across analyses, and determine which areas require further investigation.

#### Acceptance Criteria

1. THE TET_Analysis_System SHALL generate a comprehensive results document at docs/tet_comprehensive_results.md that synthesizes findings from all TET analysis components.

2. THE TET_Analysis_System SHALL include an Executive Summary section presenting the 3-5 most important findings across all analyses with effect sizes and statistical significance levels.

3. THE TET_Analysis_System SHALL include a Descriptive Statistics section summarizing temporal dynamics for each Dimension, including peak timing patterns, dose-dependent trajectories, and baseline comparisons.

4. THE TET_Analysis_System SHALL include an LME Results section reporting significant State, Dose, and interaction effects for each Dimension with standardized coefficients, confidence intervals, and FDR-corrected p-values organized by effect type.

5. THE TET_Analysis_System SHALL include a Peak and AUC Analysis section reporting dose comparisons for intensity and duration metrics with effect sizes, confidence intervals, and identifying dimensions showing strongest dose sensitivity.

6. THE TET_Analysis_System SHALL include a Dimensionality Reduction section interpreting retained PCA_Components by describing dimension loadings, variance explained, and temporal dynamics of component scores across conditions.

7. THE TET_Analysis_System SHALL include a Clustering Analysis section characterizing identified Cluster_States by their dimension profiles, temporal prevalence patterns, dose sensitivity, and stability metrics.

8. THE TET_Analysis_System SHALL include a Cross-Analysis Integration section identifying convergent findings across multiple analysis approaches and highlighting dimensions or patterns that show consistent effects.

9. THE TET_Analysis_System SHALL include a Methodological Notes section documenting any data quality issues, model assumptions, limitations, or analytical decisions that may affect interpretation.

10. THE TET_Analysis_System SHALL include a Further Investigation section listing specific unresolved questions, ambiguous findings, contradictory results, or patterns requiring additional analysis with concrete suggestions for follow-up analyses.

11. THE TET_Analysis_System SHALL format statistical results consistently using standardized notation (β for coefficients, r for effect sizes, p_fdr for corrected p-values) and include confidence intervals for all effect estimates.

12. THE TET_Analysis_System SHALL reference specific figures and tables from the analysis pipeline using relative paths to enable navigation between the results document and supporting materials.

13. THE TET_Analysis_System SHALL organize findings hierarchically with clear section headings, subsections for each dimension or analysis component, and bullet points for key findings to facilitate rapid comprehension.

14. THE TET_Analysis_System SHALL generate the comprehensive results document automatically after all analysis components complete, ensuring synchronization between reported findings and actual analysis outputs.

15. THE TET_Analysis_System SHALL update the comprehensive results document when any analysis component is re-run, maintaining consistency between the synthesis document and underlying results files.
