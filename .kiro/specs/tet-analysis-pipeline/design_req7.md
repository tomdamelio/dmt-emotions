Aquí tenés el **documento de diseño completo**, totalmente reescrito, integrado y coherente, siguiendo:

* **La estructura y estilo** del documento original que me compartiste.
* **Redacción contractual** (“TET_Analysis_System SHALL …”).
* **Inclusión completa** del framework GLHMM + contraste *across-sessions-within-subject* + interacción High vs Low × RS vs DMT.
* **Formato markdown (.md)**.
* **Sin guiones largos (—)** como pedís en tus preferencias.

Este documento está listo para usar dentro de un repositorio de especificaciones técnicas.

---

# Design Document: Requirement 7 - State Modelling of Experiential Dynamics

## Overview

This document specifies the complete design for identifying discrete experiential states in TET data using both static clustering and temporal state modelling. The system shall apply KMeans clustering as a static baseline and Gaussian Linear Hidden Markov Models (GLHMM) to capture temporal structure in experiential dynamics. Analyses shall be performed on z scored experiential dimensions across all time points.

The system shall evaluate model quality using silhouette scores (for static clustering) and information criteria or variational free energy (for temporal HMMs), assess stability with bootstrap resampling, compute Viterbi paths and posterior state probabilities, quantify occupancy metrics, and evaluate dose related differences using both classical statistics and permutation based GLHMM statistical tests.

Critically, the system shall implement the across sessions within subject permutation test, which enables robust inference of session dependent changes in latent experiential states while preserving within session temporal structure. This test shall be used to evaluate interaction effects across conditions, specifically assessing whether High vs Low dose differences appear under DMT but not during Resting State.

---

## Requirements Reference

**Requirement 7: Clustering of Experiential States**

User Story: As a researcher, I want the system to identify discrete experiential states using both static and temporal models, so that I can characterize qualitatively distinct experience patterns and how these patterns differ across time, dose, and session.

### Acceptance Criteria

* **7.1:** TET_Analysis_System SHALL apply KMeans clustering with k = 2, 3, 4 to z scored experiential dimensions.
* **7.2:** TET_Analysis_System SHALL fit Gaussian Linear Hidden Markov Models (GLHMM) with S = 2, 3, 4 latent states for each subject and session, preserving the temporal ordering of time bins.
* **7.3:**

  * For KMeans: TET_Analysis_System SHALL compute silhouette scores for each k.
  * For GLHMM: TET_Analysis_System SHALL compute BIC or variational free energy for each S.
* **7.4:** TET_Analysis_System SHALL select optimal solutions based on highest silhouette score (static) and lowest BIC or highest free energy (temporal).
* **7.5:** TET_Analysis_System SHALL evaluate the stability of clustering and state solutions via 1000 bootstrap resampling iterations using adjusted Rand index (ARI) calculated on KMeans labels or GLHMM Viterbi paths.
* **7.6:**

  * For KMeans: TET_Analysis_System SHALL compute distance based soft assignments for cluster probabilities.
  * For GLHMM: TET_Analysis_System SHALL compute posterior state probabilities (gamma) and Viterbi hard state assignments.
* **7.7:** TET_Analysis_System SHALL compute state occupancy metrics including fractional occupancy, number of visits, and dwell times for all KMeans and GLHMM solutions.
* **7.8:** TET_Analysis_System SHALL implement the across sessions within subject permutation test for GLHMM derived state time courses and aggregated state metrics.
* **7.9:** TET_Analysis_System SHALL evaluate interaction effects such as (High minus Low in DMT) minus (High minus Low in Resting State) using the across sessions within subject test.
* **7.10:** TET_Analysis_System SHALL apply classical paired t tests with BH FDR correction for secondary-confirmatory comparisons where appropriate.
* **7.11:** TET_Analysis_System SHALL export state assignments, posterior probabilities, occupancy metrics, permutation test results, and classical statistical outputs as CSV files.

---

# Architecture

## Component Overview

```
TETStateModelAnalyzer
├── fit_kmeans()                
├── fit_glhmm()                 
├── evaluate_models()           
├── select_optimal_models()     
├── bootstrap_stability()       
├── decode_states()             
├── compute_state_metrics()     
└── export_results()            

TETStateDoseAnalyzer
├── prepare_state_data()        
├── compute_pairwise_tests()    
├── apply_glhmm_permutation_test()  
├── evaluate_interaction_effects()  
├── apply_fdr_correction()      
└── export_results()            
```

## Data Flow

```
Preprocessed TET Data (tet_preprocessed.csv)
    ↓
Extract z-scored experiential dimensions
    ↓
Static and temporal models:
    Fit KMeans (k = 2, 3, 4)
    Fit GLHMM (S = 2, 3, 4)
    ↓
Model evaluation:
    Silhouette (KMeans)
    BIC or Free Energy (GLHMM)
    ↓
Select optimal k and S
    ↓
Stability assessment (1000 bootstraps)
    ↓
Decoding:
    KMeans soft assignments
    GLHMM Viterbi paths
    GLHMM posterior probabilities (gamma)
    ↓
State metrics:
    Fractional occupancy
    Visits and dwell times
    Time resolved state probabilities
    ↓
Statistical testing:
    Across sessions within subject permutation test
    Interaction effect (High vs Low in DMT minus RS)
    Paired t tests with FDR
    ↓
Exports:
    Assignments, probabilities, state metrics, permutation outputs
```

---

# Detailed Design

## 1. TETStateModelAnalyzer Class

### 1.1 Initialization

```python
class TETStateModelAnalyzer:
    def __init__(
        self,
        data: pd.DataFrame,
        dimensions: List[str],
        subject_id_col: str,
        session_id_col: str,
        time_col: str,
        state_values: List[int] = [2, 3, 4],
        random_seed: int = 22,
    ):
        self.data = data.copy()
        self.dimensions = dimensions
        self.subject_id_col = subject_id_col
        self.session_id_col = session_id_col
        self.time_col = time_col
        self.state_values = state_values
        self.random_seed = random_seed

        self.kmeans_models = {}
        self.glhmm_models = {}

        self.evaluation_results = None
        self.optimal_kmeans_states = None
        self.optimal_glhmm_states = None

        self.bootstrap_results = None

        self.kmeans_assignments = None
        self.glhmm_viterbi = None
        self.glhmm_probabilities = None
        self.occupancy_measures = None
```

### 1.2 KMeans Fitting

**Purpose:** Provide a static baseline clustering method.

**Design Requirements:**

* TET_Analysis_System SHALL fit KMeans for k = 2, 3, 4.
* TET_Analysis_System SHALL store models and cluster labels.
* TET_Analysis_System SHALL compute soft assignment probabilities using normalised inverse distances.

### 1.3 GLHMM Fitting

**Purpose:** Infer latent experiential states with temporal structure.

**Design Requirements:**

* TET_Analysis_System SHALL construct sequences per subject and session, preserving time order.
* TET_Analysis_System SHALL fit GLHMM models for S = 2, 3, 4.
* TET_Analysis_System SHALL compute BIC or Free Energy for each S.
* TET_Analysis_System SHALL store:

  * Viterbi hard paths
  * Posterior state probabilities (gamma)
  * Model parameters

### 1.4 Model Evaluation

* For KMeans: silhouette score
* For GLHMM: BIC or variational free energy

The system SHALL select optimal k and S.

### 1.5 Bootstrap Stability

* TET_Analysis_System SHALL perform 1000 bootstrap iterations.
* TET_Analysis_System SHALL compute ARI for each iteration.
* Stability SHALL be reported for both KMeans and GLHMM Viterbi assignments.

### 1.6 Decoding and Probability Computation

* GLHMM decoding via Viterbi algorithm.
* Posterior probabilities (gamma) from GLHMM inference.
* KMeans probabilistic assignment via normalised inverse distance.

### 1.7 State Occupancy Metrics

The system SHALL compute:

* Fractional occupancy per state
* Number of state visits
* Mean dwell times
* State sequences per subject and session

Metrics SHALL be computed for both KMeans (hard labels only) and GLHMM (Viterbi and gamma).

---

# 2. Statistical Testing Framework

## 2.1 Classical Analyses

* Paired t tests High vs Low dose
* BH FDR correction
* Applied to:

  * fractional occupancy
  * visits
  * dwell times
  * time resolved gamma

## 2.2 Across Sessions Within Subject Permutation Test

### Purpose

Evaluate whether state time courses and aggregated metrics differ across sessions within the same subject, while preserving temporal structure inside each session.

### Specification

* TET_Analysis_System SHALL reshuffle entire sessions across permutation iterations.
* TET_Analysis_System SHALL maintain trial order within each session.
* TET_Analysis_System SHALL compute state time courses and occupancy metrics for each permutation.
* The null distribution SHALL be constructed from permuted results.
* P values SHALL be computed as rank based comparisons of the empirical (unpermuted) metrics.
* TET_Analysis_System SHALL apply this test to:

  * Viterbi based state time courses
  * Gamma based state probabilities
  * Fractional occupancy
  * Visits
  * Dwell times

### Interaction Effect

The experimental design requires:

* No High vs Low difference during Resting State.
* Significant High vs Low difference during DMT.

Thus:

* TET_Analysis_System SHALL compute the interaction effect:

  (High minus Low in DMT) minus (High minus Low in Resting State)

* TET_Analysis_System SHALL apply the across sessions within subject test to this interaction.

### Implementation Guidance

The system SHALL follow the methods in the GLHMM tutorial:

```
Testing_across_sessions_within_subject.ipynb
https://github.com/vidaurre/glhmm/blob/main/docs/notebooks/Testing_across_sessions_within_subject.ipynb
```


---

# 3. Outputs and Export

The system SHALL export as CSV:

* KMeans hard labels and probabilities
* GLHMM Viterbi paths
* GLHMM posterior state probabilities (gamma)
* Fractional occupancy per subject and session
* Visits and dwell times
* Permutation derived p values
* Interaction effect p values
* Classical test statistics and FDR corrected results

---

# 4. Summary

This design provides a comprehensive framework for modelling experiential states using both static and temporal methods, with rigorous statistical testing including a session level permutation framework specifically suited for longitudinal session based designs such as RS vs DMT at High vs Low doses.

