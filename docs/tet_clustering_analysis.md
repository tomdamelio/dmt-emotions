# TET Clustering and State Modelling Analysis

## Overview

This document describes the clustering and state modelling analysis pipeline for TET (Temporal Experience Tracing) data. The analysis identifies discrete experiential states using both static clustering (KMeans) and temporal state modelling (GLHMM), evaluates model quality, assesses stability, and tests for dose effects on state occupancy metrics.

## Table of Contents

1. [Methodology](#methodology)
   - [KMeans Clustering](#kmeans-clustering)
   - [GLHMM State Modelling](#glhmm-state-modelling)
   - [Model Evaluation](#model-evaluation)
   - [Bootstrap Stability Assessment](#bootstrap-stability-assessment)
   - [State Occupancy Metrics](#state-occupancy-metrics)
2. [Statistical Testing](#statistical-testing)
   - [Classical Paired t-tests](#classical-paired-t-tests)
   - [Across-Sessions-Within-Subject Permutation Tests](#across-sessions-within-subject-permutation-tests)
   - [Interaction Effect Evaluation](#interaction-effect-evaluation)
3. [Reproducing Preliminary Analyses](#reproducing-preliminary-analyses)
4. [Usage Examples](#usage-examples)
5. [Output Files](#output-files)
6. [Interpretation Guidelines](#interpretation-guidelines)

---

## Methodology

### KMeans Clustering

**Purpose**: Identify discrete experiential states using static clustering that treats each time point independently.

**Assumptions**:
- Observations are independent and identically distributed (i.i.d.)
- Clusters are spherical and have similar variance
- Number of clusters is known or can be determined empirically

**Algorithm**:
1. Extract z-scored dimension matrix (n_observations × n_dimensions)
2. Fit KMeans models for k = 2, 3, 4 clusters
3. Compute hard cluster assignments (nearest centroid)
4. Compute soft probabilistic assignments using normalized inverse distances:
   ```
   prob_k = (1/dist_k) / sum(1/dist_j for all j)
   ```

**Advantages**:
- Simple, interpretable baseline
- Fast computation
- Identifies characteristic "profiles" of experience

**Limitations**:
- Ignores temporal structure and dependencies
- Assumes spherical clusters
- Sensitive to initialization (mitigated by multiple random starts)


### GLHMM State Modelling

**Purpose**: Capture temporal structure and sequential dependencies in experiential dynamics.

**Assumptions**:
- States evolve according to a Markov process
- Observations within a state follow a Gaussian distribution
- Temporal dependencies are important for understanding experiential dynamics

**Algorithm**:
1. Construct sequences per subject-session (preserving temporal order)
2. Fit GLHMM models for S = 2, 3, 4 states
3. Decode states using Viterbi algorithm (hard assignments)
4. Compute posterior probabilities using forward-backward algorithm (soft assignments)

**Advantages**:
- Captures temporal structure and state transitions
- Models sequential dependencies
- Provides transition probabilities between states
- Can identify temporal patterns (persistence, switching)

**Limitations**:
- More complex than KMeans
- Requires more data for reliable estimation
- Computationally intensive
- Assumes Markovian dynamics

**KMeans vs GLHMM**:
- **KMeans**: Static clustering, treats time points independently, identifies characteristic profiles
- **GLHMM**: Temporal model, captures sequential structure, identifies dynamic trajectories
- **Complementary**: KMeans provides interpretable baseline; GLHMM adds temporal sophistication


### Model Evaluation

**Silhouette Score** (KMeans):
- Measures how well-separated clusters are
- Range: [-1, 1], with 1 being perfect clustering
- Formula: `s = (b - a) / max(a, b)` where:
  - `a` = mean intra-cluster distance
  - `b` = mean nearest-cluster distance

**Interpretation**:
- 0.7-1.0: Strong cluster structure
- 0.5-0.7: Reasonable cluster structure
- 0.25-0.5: Weak cluster structure
- < 0.25: No substantial cluster structure

**BIC (Bayesian Information Criterion)** (GLHMM):
- Balances model fit and complexity
- Formula: `BIC = -2 * log(L) + k * log(n)` where:
  - `L` = likelihood
  - `k` = number of parameters
  - `n` = number of observations
- Lower BIC indicates better model

**Free Energy** (GLHMM):
- Variational lower bound on model evidence
- Balances fit and complexity in variational inference
- Higher free energy indicates better model

**Model Selection**:
- KMeans: Select k with highest silhouette score
- GLHMM: Select S with lowest BIC or highest free energy


### Bootstrap Stability Assessment

**Purpose**: Evaluate robustness of clustering solutions to sampling variability.

**Algorithm**:
1. For each bootstrap iteration (default: 1000):
   - Resample observations with replacement
   - Fit clustering model on bootstrap sample
   - Compute Adjusted Rand Index (ARI) comparing bootstrap labels to original
2. Compute mean ARI and 95% confidence interval

**Adjusted Rand Index (ARI)**:
- Measures similarity between two clusterings, adjusted for chance
- Range: [-1, 1], with 1 being perfect agreement
- ARI = 0: Agreement expected by chance
- ARI < 0: Less agreement than expected by chance

**Interpretation**:
- ARI > 0.8: Highly stable clustering
- ARI 0.6-0.8: Moderately stable clustering
- ARI 0.4-0.6: Weakly stable clustering
- ARI < 0.4: Unstable clustering

**Note**: Bootstrap stability for GLHMM requires session-level resampling to preserve temporal structure. This is computationally intensive and may be skipped for initial analyses.


### State Occupancy Metrics

For each subject-session and each cluster/state, we compute:

**1. Fractional Occupancy**
- Definition: Proportion of time spent in each state
- Formula: `fractional_occupancy = n_timepoints_in_state / total_timepoints`
- Range: [0, 1]
- Interpretation: Higher values indicate more dominant states

**2. Number of Visits**
- Definition: Count of transitions into each state
- Computation: Count state changes where previous state ≠ current state
- Range: [0, ∞)
- Interpretation: Higher values indicate more state switching; lower values suggest stable, persistent states

**3. Mean Dwell Time**
- Definition: Average duration of consecutive time bins in each state
- Computation: Mean length of continuous runs in the state
- Range: [1, ∞) time bins
- Interpretation: Higher values indicate more stable, persistent states; lower values suggest rapid state switching

**Temporal Resolution**: 
- TET data sampled at 0.25 Hz (4 seconds per bin)
- Dwell times are reported in bins (multiply by 4 to get seconds)


---

## Statistical Testing

### Classical Paired t-tests

**Purpose**: Test for dose effects (High vs Low) on state occupancy metrics using parametric tests.

**Design**: Within-subject paired comparisons
- Each subject experiences both High and Low dose conditions
- Controls for individual differences

**Procedure**:
1. For each metric (fractional_occupancy, n_visits, mean_dwell_time) and cluster/state:
   - Pivot data to get paired observations (High vs Low per subject)
   - Remove subjects with missing data
   - Perform paired t-test: `t = mean(High - Low) / SE(High - Low)`
   - Compute 95% confidence interval for mean difference
2. Apply Benjamini-Hochberg FDR correction across all tests

**Assumptions**:
- Differences are normally distributed
- Paired observations are independent across subjects

**Output**: t-statistic, p-value, mean difference, 95% CI


### Across-Sessions-Within-Subject Permutation Tests

**Purpose**: Non-parametric test for dose effects that preserves within-session temporal structure.

**Why Permutation Tests?**
- No distributional assumptions (robust to non-normality)
- Appropriate for temporal data (preserves within-session order)
- Exact p-values under null hypothesis
- Robust to outliers

**Algorithm**:
1. Compute observed test statistic: mean difference (High - Low) across subjects
2. For each permutation iteration (default: 1000):
   - For each subject: randomly shuffle dose labels across sessions
   - Preserve within-session temporal order (entire sessions are reshuffled, not individual time points)
   - Recompute state occupancy metrics for permuted data
   - Compute permuted test statistic
3. Build null distribution from permuted statistics
4. Compute p-value: `p = proportion of |permuted| >= |observed|`
5. Apply Benjamini-Hochberg FDR correction

**Key Feature**: Reshuffling entire sessions (not individual time points) preserves temporal dependencies within sessions while breaking the association between dose and outcome.

**Reference**: `notebooks/Testing_across_sessions_within_subject.ipynb`

**Output**: observed statistic, permutation p-value, number of permutations


### Interaction Effect Evaluation

**Purpose**: Test whether dose effects differ between experimental states (DMT vs RS).

**Interaction Effect**:
```
Interaction = (High - Low | DMT) - (High - Low | RS)
```

**Interpretation**:
- **Positive interaction**: Dose effect is stronger in DMT than RS
  - Drug amplifies dose-dependent differences
  - Dose sensitivity is enhanced during DMT state
  
- **Negative interaction**: Dose effect is weaker in DMT than RS (or reversed)
  - Could indicate ceiling/floor effects
  - Compensatory mechanisms during DMT
  
- **Non-significant interaction**: Dose effects are similar across DMT and RS
  - Dose-dependent differences are not specific to drug state
  - General dose sensitivity independent of experimental condition

**Algorithm**:
1. Compute dose differences within each experimental state:
   - DMT: `diff_DMT = mean(High - Low | DMT)`
   - RS: `diff_RS = mean(High - Low | RS)`
2. Compute interaction: `interaction = diff_DMT - diff_RS`
3. Apply across-sessions-within-subject permutation test:
   - Shuffle dose labels within DMT sessions
   - Shuffle dose labels within RS sessions (independently)
   - Recompute interaction for each permutation
4. Compute permutation p-value
5. Apply FDR correction

**Output**: interaction effect, DMT difference, RS difference, permutation p-value


---

## Reproducing Preliminary Analyses

This section describes how the current analysis pipeline reproduces and extends preliminary findings from the thesis.

### Centroid Profile Plots (Fig. 3.5)

**Original Analysis**:
- Figure 3.5 shows normalized centroid profiles for k=2 KMeans clusters
- Each cluster has a characteristic profile across experiential dimensions
- Normalization allows comparison of relative dimension importance

**Reproduction**:
```python
from tet.state_visualization import TETStateVisualization

# Load data and assignments
viz = TETStateVisualization(
    data=data,
    kmeans_assignments=kmeans_assignments
)

# Generate centroid profile plots
viz.plot_kmeans_centroid_profiles(
    k=2,
    output_dir='results/tet/figures'
)
```

**Output**: `clustering_kmeans_centroids_k2.png`

**Interpretation**:
- Each bar shows the normalized contribution of a dimension to the cluster profile
- Positive values (blue): elevated dimensions characteristic of the state
- Negative values (red): suppressed dimensions anti-characteristic of the state
- Magnitude indicates importance for defining the cluster


### Time-Course Cluster Probability Plots (Fig. 3.6)

**Original Analysis**:
- Figure 3.6 shows how cluster probabilities evolve over time
- Separate curves for High and Low dose in DMT
- Reveals temporal dynamics of experiential states

**Reproduction**:
```python
# Generate time-course probability plots
viz.plot_kmeans_cluster_timecourses(
    k=2,
    include_rs=True,
    output_dir='results/tet/figures'
)
```

**Output**: `clustering_kmeans_prob_timecourses_with_rs.png`

**Interpretation**:
- Y-axis: Probability of cluster membership (soft assignments)
- X-axis: Time in minutes from session start
- Shaded regions: Standard error of the mean (SEM) across subjects
- Temporal patterns reveal:
  - State persistence (stable high probability)
  - State transitions (gradual changes)
  - Dose effects (divergence between High and Low curves)


### KMeans vs GLHMM Correspondence

**Purpose**: Understand how static clustering (KMeans) relates to temporal state modelling (GLHMM).

**Analysis**:
```python
# Generate correspondence heatmap
viz.plot_kmeans_glhmm_crosswalk(
    k=2,
    output_dir='results/tet/figures'
)
```

**Output**: 
- `kmeans_glhmm_crosswalk.png`: Heatmap visualization
- `kmeans_glhmm_crosswalk.csv`: Contingency table

**Interpretation**:
- **High correspondence**: KMeans and GLHMM identify similar states
  - Static and temporal models converge
  - Temporal structure is not critical for state definition
  
- **Low correspondence**: Methods capture different aspects
  - Temporal structure is important
  - GLHMM identifies temporal substates within KMeans clusters
  
- **One-to-one mapping**: Equivalent state definitions
- **Many-to-one mapping**: GLHMM identifies finer-grained substates

**Relationship to Preliminary Findings**:
- KMeans k=2 provides baseline comparable to Fig. 3.5-3.6
- GLHMM extends analysis by capturing temporal dynamics
- Correspondence analysis validates consistency between methods


---

## Usage Examples

### Complete Analysis Pipeline

```bash
# Run complete clustering analysis
python scripts/compute_clustering_analysis.py \
    --input results/tet/tet_preprocessed.csv \
    --output results/tet/clustering \
    --state-values 2 3 4 \
    --n-bootstrap 1000 \
    --n-permutations 1000
```

**Options**:
- `--input`: Path to preprocessed TET data
- `--output`: Output directory for results
- `--state-values`: Number of states to test (default: 2 3 4)
- `--n-bootstrap`: Number of bootstrap iterations (default: 1000)
- `--n-permutations`: Number of permutation iterations (default: 1000)
- `--skip-glhmm-permutation`: Skip GLHMM permutation tests (faster for debugging)

### Generate Visualizations

```bash
# Generate all clustering figures
python scripts/plot_state_results.py \
    --input results/tet/clustering \
    --output results/tet/figures
```

**Generated Figures**:
- Centroid profile plots (Fig. 3.5-like)
- Time-course cluster probability plots (Fig. 3.6-like)
- GLHMM state time-course plots
- KMeans-GLHMM correspondence heatmap


### Inspect Results

```bash
# Quick inspection of clustering results
python scripts/inspect_clustering_results.py \
    --input results/tet/clustering
```

**Output**:
- Model evaluation summary (optimal k and S)
- Bootstrap stability results (mean ARI)
- State occupancy metrics summary
- Significant dose effects
- Significant interaction effects
- Interpretation template for lab diary/manuscript

### Programmatic Usage

```python
import pandas as pd
from tet.state_model_analyzer import TETStateModelAnalyzer
from tet.state_dose_analyzer import TETStateDoseAnalyzer
from tet.state_visualization import TETStateVisualization

# Load preprocessed data
data = pd.read_csv('results/tet/tet_preprocessed.csv')

# Define z-scored dimensions
dimensions = [col for col in data.columns if col.endswith('_z')]

# Initialize analyzer
analyzer = TETStateModelAnalyzer(
    data=data,
    dimensions=dimensions,
    state_values=[2, 3, 4]
)

# Fit models
analyzer.fit_kmeans()
analyzer.fit_glhmm()

# Evaluate and select optimal models
evaluation = analyzer.evaluate_models()
optimal_k, optimal_S = analyzer.select_optimal_models()

# Assess stability
stability = analyzer.bootstrap_stability(n_bootstrap=1000)

# Compute state metrics
metrics = analyzer.compute_state_metrics()

# Export results
paths = analyzer.export_results('results/tet/clustering')

# Statistical testing
dose_analyzer = TETStateDoseAnalyzer(
    state_metrics=metrics,
    cluster_probabilities=None  # Load from exported files if needed
)

# Perform tests
classical_results = dose_analyzer.compute_pairwise_tests()
perm_results = dose_analyzer.apply_glhmm_permutation_test(n_permutations=1000)
interaction_results = dose_analyzer.evaluate_interaction_effects(n_permutations=1000)

# Apply FDR correction
dose_analyzer.apply_fdr_correction()

# Export statistical results
dose_analyzer.export_results('results/tet/clustering')
```


---

## Output Files

### Clustering Results

**clustering_kmeans_assignments.csv**
- Columns: subject, session_id, state, dose, t_bin, cluster, prob_cluster_0, prob_cluster_1, ...
- Hard cluster assignments and soft probabilities for each time point

**clustering_glhmm_viterbi.csv**
- Columns: subject, session_id, state, dose, t_bin, viterbi_state
- Hard state assignments from Viterbi decoding

**clustering_glhmm_probabilities.csv**
- Columns: subject, session_id, state, dose, t_bin, gamma_state_0, gamma_state_1, ...
- Posterior state probabilities from forward-backward algorithm

**clustering_state_metrics.csv**
- Columns: subject, session_id, state, dose, method, cluster_state, fractional_occupancy, n_visits, mean_dwell_time
- State occupancy metrics for each subject-session-state combination

**clustering_evaluation.csv**
- Columns: method, n_states, silhouette_score, bic, free_energy
- Model evaluation metrics for KMeans and GLHMM

**clustering_bootstrap_stability.csv**
- Columns: method, n_states, mean_ari, ci_lower, ci_upper
- Bootstrap stability results


### Statistical Test Results

**clustering_dose_tests_classical.csv**
- Columns: metric, method, cluster_state, n_pairs, mean_high, mean_low, mean_diff, t_statistic, p_value, ci_lower, ci_upper, p_fdr, significant
- Classical paired t-test results for dose effects

**clustering_dose_tests_permutation.csv**
- Columns: metric, method, cluster_state, observed_stat, p_value_perm, n_permutations, p_fdr, significant
- Permutation test results for dose effects

**clustering_interaction_effects.csv**
- Columns: metric, method, cluster_state, interaction_effect, dmt_diff, rs_diff, p_value_perm, n_permutations, p_fdr, significant
- Interaction effect results (State × Dose)

### Figures

**clustering_kmeans_centroids_k2.png**
- Centroid profile plots for k=2 KMeans solution (replicates Fig. 3.5)

**clustering_kmeans_prob_timecourses_with_rs.png**
- Time-course cluster probability plots (replicates Fig. 3.6)

**glhmm_state_prob_timecourses.png**
- GLHMM state probability time courses

**kmeans_glhmm_crosswalk.png**
- Correspondence heatmap between KMeans and GLHMM


---

## Interpretation Guidelines

### Centroid Profiles

**What to Look For**:
- Dimensions with large positive values: Characteristic features of the state
- Dimensions with large negative values: Anti-characteristic features
- Dimensions near zero: Neutral for that state

**Example Interpretation**:
> "Cluster 0 is characterized by elevated anxiety (0.8) and reduced pleasantness (-0.6), 
> suggesting a dysphoric experiential state. In contrast, Cluster 1 shows elevated 
> pleasantness (0.9) and reduced anxiety (-0.4), indicating a euphoric state."

### State Occupancy Metrics

**Fractional Occupancy**:
- High values (>0.5): Dominant state
- Low values (<0.2): Rare or transient state
- Compare across conditions to identify dose effects

**Number of Visits**:
- High values: Frequent state switching
- Low values: Stable, persistent states
- Reflects dynamic vs. stable experiential trajectories

**Mean Dwell Time**:
- High values (>10 bins = 40 seconds): Persistent states
- Low values (<5 bins = 20 seconds): Transient states
- Indicates state stability


### Dose Effects

**Significant Positive Effect** (High > Low):
- State is more prevalent/stable at higher dose
- Dose-dependent amplification of the state

**Significant Negative Effect** (High < Low):
- State is less prevalent/stable at higher dose
- Dose-dependent suppression of the state

**Non-Significant Effect**:
- State occupancy is dose-independent
- State is present across dose levels

**Example Interpretation**:
> "Fractional occupancy of Cluster 0 (dysphoric state) was significantly higher in 
> the High dose condition (M=0.45) compared to Low dose (M=0.32), t(15)=3.2, p=0.006, 
> suggesting dose-dependent amplification of negative affect."

### Interaction Effects

**Significant Positive Interaction**:
- Dose effect is stronger in DMT than RS
- Drug amplifies dose sensitivity

**Significant Negative Interaction**:
- Dose effect is weaker in DMT than RS
- Drug attenuates dose sensitivity

**Example Interpretation**:
> "A significant State × Dose interaction was found for fractional occupancy of 
> Cluster 1 (interaction=0.15, p=0.012), indicating that the dose effect on this 
> euphoric state was stronger during DMT (Δ=0.20) than during resting state (Δ=0.05). 
> This suggests that DMT amplifies dose-dependent modulation of positive affect."


### Time-Course Patterns

**Stable High Probability**:
- Persistent, dominant state throughout session
- Low variability across time

**Gradual Increase/Decrease**:
- Smooth state transitions
- Progressive changes in experiential state

**Rapid Changes**:
- Abrupt state switching
- Discrete transitions between states

**Oscillations**:
- Alternating between states
- Cyclical experiential dynamics

**Dose Divergence**:
- Curves separate at specific time points
- Indicates when dose effects emerge temporally

**Example Interpretation**:
> "Cluster 0 probability showed a gradual increase over the first 10 minutes in the 
> High dose condition, reaching a plateau at ~0.7, while remaining stable at ~0.3 
> in the Low dose condition. This temporal divergence suggests that dose effects on 
> this dysphoric state emerge progressively during the early phase of the experience."

---

## References

- **Preliminary Analysis**: Thesis Chapter 3, Figures 3.5-3.6
- **Permutation Testing**: `notebooks/Testing_across_sessions_within_subject.ipynb`
- **GLHMM**: Vidaurre et al. (2018). "Discovering dynamic brain networks from big data in rest and task." NeuroImage.
- **Bootstrap Stability**: Hennig (2007). "Cluster-wise assessment of cluster stability." Computational Statistics & Data Analysis.

---

## Notes

- All analyses use z-scored dimensions to ensure comparable scales
- Random seed (default: 22) ensures reproducibility
- FDR correction controls false discovery rate at α=0.05
- Permutation tests provide exact p-values under null hypothesis
- Bootstrap stability assesses robustness to sampling variability

