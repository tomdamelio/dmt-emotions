# -*- coding: utf-8 -*-
"""
TET State Model Analyzer Module

This module provides functionality for identifying discrete experiential states
using both static clustering (KMeans) and temporal state modelling (GLHMM).
It evaluates model quality, assesses stability, and computes state occupancy metrics.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.cluster import adjusted_rand_score

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TETStateModelAnalyzer:
    """
    Analyzes experiential states using static clustering and temporal HMMs.
    
    This class applies KMeans clustering as a static baseline and Gaussian Linear
    Hidden Markov Models (GLHMM) to capture temporal structure in experiential
    dynamics. It evaluates model quality, assesses stability via bootstrap resampling,
    and computes state occupancy metrics.
    
    KMeans vs GLHMM Differences:
        - KMeans: Static clustering that treats each time point independently.
          Assumes observations are independent and identically distributed (i.i.d.).
          Provides a baseline for identifying experiential states without considering
          temporal dynamics. Useful for identifying characteristic "profiles" of
          experience across dimensions.
        
        - GLHMM: Temporal state model that captures sequential structure and state
          transitions. Accounts for temporal dependencies and allows states to evolve
          over time. Better suited for modeling dynamic experiential trajectories.
          Provides transition probabilities between states and can identify temporal
          patterns like state persistence and switching.
    
    Interpretation Guidelines:
        - Centroid Profiles: Each KMeans cluster has a centroid representing the
          average z-scored dimension values for observations in that cluster.
          Normalized centroids show the relative importance of each dimension
          within a cluster, independent of absolute intensity. Dimensions with
          large positive values are characteristic of that experiential state,
          while dimensions with large negative values are anti-characteristic.
        
        - State Occupancy Metrics:
          * Fractional Occupancy: Proportion of time spent in each state. Higher
            values indicate more dominant states. Useful for comparing state
            prevalence across conditions (e.g., High vs Low dose).
          * Number of Visits: Frequency of transitions into each state. Higher
            values indicate more state switching. Low values suggest stable,
            persistent states.
          * Mean Dwell Time: Average duration of consecutive time bins in each
            state. Higher values indicate more stable, persistent states. Lower
            values suggest rapid state switching.
    
    Attributes:
        data (pd.DataFrame): Preprocessed TET data with z-scored dimensions
        dimensions (List[str]): List of z-scored dimension column names
        subject_id_col (str): Column name for subject identifier
        session_id_col (str): Column name for session identifier
        time_col (str): Column name for time bin
        state_values (List[int]): Number of states to test (default: [2, 3, 4])
        random_seed (int): Random seed for reproducibility (default: 22)
        kmeans_models (Dict): Fitted KMeans models for each k
        glhmm_models (Dict): Fitted GLHMM models for each S
        evaluation_results (pd.DataFrame): Model evaluation metrics
        optimal_kmeans_states (int): Optimal number of KMeans clusters
        optimal_glhmm_states (int): Optimal number of GLHMM states
        bootstrap_results (pd.DataFrame): Bootstrap stability results
        kmeans_assignments (pd.DataFrame): KMeans cluster assignments
        glhmm_viterbi (pd.DataFrame): GLHMM Viterbi paths
        glhmm_probabilities (pd.DataFrame): GLHMM posterior probabilities
        occupancy_measures (pd.DataFrame): State occupancy metrics
    
    Example:
        >>> import pandas as pd
        >>> from tet.state_model_analyzer import TETStateModelAnalyzer
        >>> 
        >>> # Load preprocessed data
        >>> data = pd.read_csv('results/tet/tet_preprocessed.csv')
        >>> 
        >>> # Define z-scored dimensions
        >>> dimensions = [col for col in data.columns if col.endswith('_z')]
        >>> 
        >>> # Initialize analyzer
        >>> analyzer = TETStateModelAnalyzer(
        ...     data=data,
        ...     dimensions=dimensions,
        ...     subject_id_col='subject',
        ...     session_id_col='session_id',
        ...     time_col='t_bin'
        ... )
        >>> 
        >>> # Fit models
        >>> kmeans_results = analyzer.fit_kmeans()
        >>> glhmm_results = analyzer.fit_glhmm()
        >>> 
        >>> # Evaluate and select optimal models
        >>> evaluation = analyzer.evaluate_models()
        >>> optimal_k, optimal_S = analyzer.select_optimal_models()
        >>> 
        >>> # Assess stability
        >>> stability = analyzer.bootstrap_stability()
        >>> 
        >>> # Compute state metrics
        >>> metrics = analyzer.compute_state_metrics()
        >>> 
        >>> # Export results
        >>> paths = analyzer.export_results('results/tet/clustering')
    """
    
    def __init__(
        self,
        data: pd.DataFrame,
        dimensions: List[str],
        subject_id_col: str = 'subject',
        session_id_col: str = 'session_id',
        time_col: str = 't_bin',
        state_values: List[int] = None,
        random_seed: int = 22,
    ):
        """
        Initialize state model analyzer with preprocessed TET data.
        
        Args:
            data (pd.DataFrame): Preprocessed TET data with columns for subject,
                session_id, state, dose, t_bin, t_sec, and z-scored dimensions
            dimensions (List[str]): List of z-scored dimension column names
                (e.g., ['pleasantness_z', 'anxiety_z', ...])
            subject_id_col (str): Column name for subject identifier (default: 'subject')
            session_id_col (str): Column name for session identifier (default: 'session_id')
            time_col (str): Column name for time bin (default: 't_bin')
            state_values (List[int]): Number of states to test (default: [2, 3, 4])
            random_seed (int): Random seed for reproducibility (default: 22)
        """
        self.data = data.copy()
        self.dimensions = dimensions
        self.subject_id_col = subject_id_col
        self.session_id_col = session_id_col
        self.time_col = time_col
        self.state_values = state_values if state_values is not None else [2, 3, 4]
        self.random_seed = random_seed
        
        # Set random seed for reproducibility
        np.random.seed(random_seed)
        
        # Initialize storage for models and results
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
        
        logger.info(f"Initialized TETStateModelAnalyzer with {len(data)} rows, "
                   f"{data[subject_id_col].nunique()} subjects, "
                   f"{len(dimensions)} dimensions")
        logger.info(f"Testing state values: {self.state_values}")
        logger.info(f"Random seed: {random_seed}")

    def fit_kmeans(self) -> Dict[int, Tuple]:
        """
        Fit KMeans clustering models for k=2, 3, 4 states.
        
        This method provides a static baseline clustering approach that does not
        consider temporal structure. It fits KMeans models for different numbers
        of clusters and computes both hard cluster assignments and soft probabilistic
        assignments based on normalized inverse distances to cluster centers.
        
        Soft Assignment Computation:
            For each observation, compute distance to each cluster center.
            Convert distances to probabilities using normalized inverse distances:
            prob_k = (1/dist_k) / sum(1/dist_j for all j)
            
            This provides a probabilistic interpretation where observations closer
            to a cluster center have higher probability of belonging to that cluster.
        
        Returns:
            Dict[int, Tuple]: Dictionary mapping k to (model, hard_labels, soft_probs)
                - model: Fitted KMeans model
                - hard_labels: Hard cluster assignments (n_observations,)
                - soft_probs: Soft probabilistic assignments (n_observations, k)
        """
        logger.info("Fitting KMeans clustering models...")
        
        # Extract z-scored dimension matrix (n_observations × n_dimensions)
        X = self.data[self.dimensions].values
        n_observations, n_dimensions = X.shape
        
        logger.info(f"Data matrix shape: {X.shape} "
                   f"({n_observations} observations × {n_dimensions} dimensions)")
        
        # Fit KMeans for each k value
        for k in self.state_values:
            logger.info(f"Fitting KMeans with k={k} clusters...")
            
            # Fit KMeans model
            kmeans = KMeans(
                n_clusters=k,
                random_state=self.random_seed,
                n_init=10,  # Number of initializations
                max_iter=300
            )
            
            # Fit and predict
            hard_labels = kmeans.fit_predict(X)
            
            # Compute soft probabilistic assignments using normalized inverse distances
            # Get distances to all cluster centers
            distances = kmeans.transform(X)  # Shape: (n_observations, k)
            
            # Avoid division by zero by adding small epsilon
            epsilon = 1e-10
            inverse_distances = 1.0 / (distances + epsilon)
            
            # Normalize to get probabilities
            soft_probs = inverse_distances / inverse_distances.sum(axis=1, keepdims=True)
            
            # Store results
            self.kmeans_models[k] = (kmeans, hard_labels, soft_probs)
            
            # Log cluster sizes
            unique, counts = np.unique(hard_labels, return_counts=True)
            cluster_sizes = dict(zip(unique, counts))
            logger.info(f"  k={k}: Cluster sizes = {cluster_sizes}")
            logger.info(f"  k={k}: Inertia = {kmeans.inertia_:.2f}")
        
        logger.info(f"Fitted KMeans models for k={self.state_values}")
        
        return self.kmeans_models

    def fit_glhmm(self) -> Dict[int, Tuple]:
        """
        Fit Gaussian Linear Hidden Markov Models for S=2, 3, 4 states.
        
        This method fits temporal state models that capture the sequential structure
        of experiential dynamics. GLHMM models are fitted separately for each
        subject-session combination to preserve within-session temporal order.
        
        For each subject-session:
        - Extract time-ordered z-scored dimensions
        - Create sequence array (n_timepoints × n_dimensions)
        - Fit GLHMM model
        
        The method computes:
        - Viterbi paths: Hard state assignments (most likely state sequence)
        - Posterior probabilities (gamma): Soft probabilistic state assignments
        
        Note: This implementation requires the glhmm library. If not available,
        the method will log a warning and return empty results.
        
        Reference:
            notebooks/Testing_across_sessions_within_subject.ipynb
            https://github.com/vidaurre/glhmm
        
        Returns:
            Dict[int, Tuple]: Dictionary mapping S to (model, viterbi_paths, gamma_probs)
                - model: Fitted GLHMM model
                - viterbi_paths: Hard state assignments per timepoint
                - gamma_probs: Posterior state probabilities per timepoint
        """
        logger.info("Fitting GLHMM models...")
        
        # Check if glhmm is available
        try:
            import glhmm
        except ImportError:
            logger.error("glhmm library not available. Install with: "
                        "pip install git+https://github.com/vidaurre/glhmm")
            logger.error("Skipping GLHMM fitting.")
            return {}
        
        # Construct sequences per subject and session
        logger.info("Constructing sequences per subject-session...")
        
        sequences = []
        sequence_metadata = []
        
        # Group by subject and session
        session_groups = self.data.groupby([self.subject_id_col, self.session_id_col])
        
        for (subject, session_id), session_data in session_groups:
            # Sort by time to ensure temporal order
            session_data = session_data.sort_values(self.time_col)
            
            # Extract z-scored dimensions
            sequence = session_data[self.dimensions].values
            
            # Store sequence and metadata
            sequences.append(sequence)
            sequence_metadata.append({
                'subject': subject,
                'session_id': session_id,
                'n_timepoints': len(sequence)
            })
        
        n_sequences = len(sequences)
        logger.info(f"Constructed {n_sequences} sequences")
        
        # Fit GLHMM for each S value
        for S in self.state_values:
            logger.info(f"Fitting GLHMM with S={S} states...")
            
            try:
                # Import the glhmm class from glhmm.glhmm module
                from glhmm.glhmm import glhmm as GLHMM
                
                # Initialize GLHMM model with K states
                model = GLHMM(K=S, covtype='shareddiag')
                
                # Fit model using dual_estimate (EM algorithm)
                # This is the main fitting method in glhmm
                model.dual_estimate(sequences)
                
                # Decode states for all sequences
                viterbi_paths = []
                gamma_probs = []
                
                for seq in sequences:
                    # Viterbi decoding (hard state assignments)
                    # The decode method returns the most likely state sequence
                    viterbi = model.decode(seq)
                    viterbi_paths.append(viterbi)
                    
                    # Posterior probabilities (soft state assignments)
                    # Use sample_Gamma to get posterior state probabilities
                    gamma = model.sample_Gamma(seq)
                    gamma_probs.append(gamma)
                
                # Store results
                self.glhmm_models[S] = (model, viterbi_paths, gamma_probs)
                
                # Log model quality metrics
                # Get free energy (variational lower bound)
                fe = model.get_fe()
                logger.info(f"  S={S}: Free Energy = {fe:.2f}")
                
            except Exception as e:
                logger.error(f"GLHMM fitting failed for S={S}: {e}")
                logger.error(f"Error details: {type(e).__name__}: {str(e)}")
                logger.error("This is expected if glhmm library is not properly installed.")
                import traceback
                logger.error(f"Traceback: {traceback.format_exc()}")
                continue
        
        if len(self.glhmm_models) > 0:
            logger.info(f"Fitted GLHMM models for S={list(self.glhmm_models.keys())}")
        else:
            logger.warning("No GLHMM models were successfully fitted. "
                          "Check glhmm library installation.")
        
        return self.glhmm_models

    def evaluate_models(self) -> pd.DataFrame:
        """
        Evaluate clustering and HMM models using quality metrics.
        
        For KMeans solutions:
        - Compute silhouette score for each k
        - Higher silhouette score indicates better-defined clusters
        - Range: [-1, 1], with 1 being perfect clustering
        
        For GLHMM solutions:
        - Compute BIC (Bayesian Information Criterion) or variational free energy
        - Lower BIC indicates better model fit with appropriate complexity
        - Free energy balances model fit and complexity
        
        Silhouette Score Interpretation:
            - 0.7-1.0: Strong structure
            - 0.5-0.7: Reasonable structure
            - 0.25-0.5: Weak structure
            - < 0.25: No substantial structure
        
        Returns:
            pd.DataFrame: Evaluation results with columns:
                - method: 'KMeans' or 'GLHMM'
                - n_states: Number of states/clusters
                - silhouette_score: Silhouette score (KMeans only)
                - bic: Bayesian Information Criterion (GLHMM only)
                - free_energy: Variational free energy (GLHMM only)
        """
        if not self.kmeans_models and not self.glhmm_models:
            raise ValueError("Must fit models before evaluation. "
                           "Call fit_kmeans() and/or fit_glhmm() first.")
        
        logger.info("Evaluating models...")
        
        # Extract data matrix for silhouette computation
        X = self.data[self.dimensions].values
        
        evaluation_list = []
        
        # Evaluate KMeans models
        for k, (model, hard_labels, soft_probs) in self.kmeans_models.items():
            logger.info(f"Evaluating KMeans k={k}...")
            
            # Compute silhouette score
            silhouette = silhouette_score(X, hard_labels)
            
            evaluation_list.append({
                'method': 'KMeans',
                'n_states': k,
                'silhouette_score': silhouette,
                'bic': np.nan,
                'free_energy': np.nan
            })
            
            logger.info(f"  k={k}: Silhouette = {silhouette:.3f}")
        
        # Evaluate GLHMM models
        for S, (model, viterbi_paths, gamma_probs) in self.glhmm_models.items():
            logger.info(f"Evaluating GLHMM S={S}...")
            
            # Extract BIC and free energy if available
            bic = model.bic if hasattr(model, 'bic') else np.nan
            free_energy = model.free_energy if hasattr(model, 'free_energy') else np.nan
            
            evaluation_list.append({
                'method': 'GLHMM',
                'n_states': S,
                'silhouette_score': np.nan,
                'bic': bic,
                'free_energy': free_energy
            })
            
            if not np.isnan(bic):
                logger.info(f"  S={S}: BIC = {bic:.2f}")
            if not np.isnan(free_energy):
                logger.info(f"  S={S}: Free Energy = {free_energy:.2f}")
        
        # Create evaluation DataFrame
        self.evaluation_results = pd.DataFrame(evaluation_list)
        
        logger.info("Model evaluation complete")
        
        return self.evaluation_results

    def select_optimal_models(self) -> Tuple[int, int]:
        """
        Select optimal number of states for KMeans and GLHMM.
        
        Selection criteria:
        - KMeans: Select k with highest silhouette score
        - GLHMM: Select S with lowest BIC or highest free energy
        
        The optimal models are stored in self.optimal_kmeans_states and
        self.optimal_glhmm_states for use in subsequent analyses.
        
        Returns:
            Tuple[int, int]: (optimal_k, optimal_S)
                - optimal_k: Optimal number of KMeans clusters
                - optimal_S: Optimal number of GLHMM states
        """
        if self.evaluation_results is None:
            raise ValueError("Must evaluate models before selection. "
                           "Call evaluate_models() first.")
        
        logger.info("Selecting optimal models...")
        
        # Select optimal KMeans model (highest silhouette score)
        kmeans_results = self.evaluation_results[
            self.evaluation_results['method'] == 'KMeans'
        ]
        
        if len(kmeans_results) > 0:
            optimal_kmeans_idx = kmeans_results['silhouette_score'].idxmax()
            self.optimal_kmeans_states = int(
                kmeans_results.loc[optimal_kmeans_idx, 'n_states']
            )
            optimal_silhouette = kmeans_results.loc[optimal_kmeans_idx, 'silhouette_score']
            
            logger.info(f"Optimal KMeans: k={self.optimal_kmeans_states} "
                       f"(silhouette={optimal_silhouette:.3f})")
        else:
            logger.warning("No KMeans results available for selection")
            self.optimal_kmeans_states = None
        
        # Select optimal GLHMM model (lowest BIC or highest free energy)
        glhmm_results = self.evaluation_results[
            self.evaluation_results['method'] == 'GLHMM'
        ]
        
        if len(glhmm_results) > 0:
            # Prefer BIC if available, otherwise use free energy
            if glhmm_results['bic'].notna().any():
                # Lower BIC is better
                optimal_glhmm_idx = glhmm_results['bic'].idxmin()
                self.optimal_glhmm_states = int(
                    glhmm_results.loc[optimal_glhmm_idx, 'n_states']
                )
                optimal_bic = glhmm_results.loc[optimal_glhmm_idx, 'bic']
                
                logger.info(f"Optimal GLHMM: S={self.optimal_glhmm_states} "
                           f"(BIC={optimal_bic:.2f})")
            elif glhmm_results['free_energy'].notna().any():
                # Higher free energy is better
                optimal_glhmm_idx = glhmm_results['free_energy'].idxmax()
                self.optimal_glhmm_states = int(
                    glhmm_results.loc[optimal_glhmm_idx, 'n_states']
                )
                optimal_fe = glhmm_results.loc[optimal_glhmm_idx, 'free_energy']
                
                logger.info(f"Optimal GLHMM: S={self.optimal_glhmm_states} "
                           f"(Free Energy={optimal_fe:.2f})")
            else:
                logger.warning("No valid GLHMM metrics for selection")
                self.optimal_glhmm_states = None
        else:
            logger.warning("No GLHMM results available for selection")
            self.optimal_glhmm_states = None
        
        return self.optimal_kmeans_states, self.optimal_glhmm_states

    def bootstrap_stability(self, n_bootstrap: int = 1000) -> pd.DataFrame:
        """
        Assess clustering stability using bootstrap resampling.
        
        This method performs bootstrap resampling to evaluate the stability of
        clustering solutions. For each bootstrap iteration:
        1. Resample observations with replacement
        2. Fit KMeans and GLHMM with optimal k and S
        3. Compute adjusted Rand index (ARI) comparing bootstrap labels to original
        
        ARI measures the similarity between two clusterings, adjusted for chance:
        - ARI = 1: Perfect agreement
        - ARI = 0: Agreement expected by chance
        - ARI < 0: Less agreement than expected by chance
        
        Stability Interpretation:
            - ARI > 0.8: Highly stable clustering
            - ARI 0.6-0.8: Moderately stable
            - ARI 0.4-0.6: Weakly stable
            - ARI < 0.4: Unstable clustering
        
        Args:
            n_bootstrap (int): Number of bootstrap iterations (default: 1000)
        
        Returns:
            pd.DataFrame: Bootstrap stability results with columns:
                - method: 'KMeans' or 'GLHMM'
                - n_states: Number of states
                - mean_ari: Mean ARI across bootstrap samples
                - ci_lower: 2.5th percentile of ARI distribution
                - ci_upper: 97.5th percentile of ARI distribution
        """
        if self.optimal_kmeans_states is None and self.optimal_glhmm_states is None:
            raise ValueError("Must select optimal models before stability analysis. "
                           "Call select_optimal_models() first.")
        
        logger.info(f"Performing bootstrap stability analysis ({n_bootstrap} iterations)...")
        
        # Extract data matrix
        X = self.data[self.dimensions].values
        n_observations = len(X)
        
        stability_results = []
        
        # Bootstrap stability for KMeans
        if self.optimal_kmeans_states is not None:
            k = self.optimal_kmeans_states
            logger.info(f"Bootstrap stability for KMeans k={k}...")
            
            # Get original labels
            _, original_labels, _ = self.kmeans_models[k]
            
            # Bootstrap iterations
            ari_scores = []
            
            for i in range(n_bootstrap):
                if (i + 1) % 100 == 0:
                    logger.info(f"  KMeans bootstrap iteration {i+1}/{n_bootstrap}")
                
                # Resample with replacement
                indices = np.random.choice(n_observations, size=n_observations, replace=True)
                X_boot = X[indices]
                original_labels_boot = original_labels[indices]
                
                try:
                    # Fit KMeans on bootstrap sample
                    kmeans_boot = KMeans(
                        n_clusters=k,
                        random_state=self.random_seed + i,  # Different seed per iteration
                        n_init=10,
                        max_iter=300
                    )
                    boot_labels = kmeans_boot.fit_predict(X_boot)
                    
                    # Compute ARI
                    ari = adjusted_rand_score(original_labels_boot, boot_labels)
                    ari_scores.append(ari)
                    
                except Exception as e:
                    logger.warning(f"Bootstrap iteration {i} failed: {e}")
                    continue
            
            # Compute statistics
            if len(ari_scores) > 0:
                mean_ari = np.mean(ari_scores)
                ci_lower = np.percentile(ari_scores, 2.5)
                ci_upper = np.percentile(ari_scores, 97.5)
                
                stability_results.append({
                    'method': 'KMeans',
                    'n_states': k,
                    'mean_ari': mean_ari,
                    'ci_lower': ci_lower,
                    'ci_upper': ci_upper
                })
                
                logger.info(f"  KMeans k={k}: Mean ARI = {mean_ari:.3f} "
                           f"[{ci_lower:.3f}, {ci_upper:.3f}]")
            else:
                logger.warning("No valid bootstrap samples for KMeans")
        
        # Bootstrap stability for GLHMM
        if self.optimal_glhmm_states is not None:
            S = self.optimal_glhmm_states
            logger.info(f"Bootstrap stability for GLHMM S={S}...")
            
            # Get original Viterbi paths
            _, original_viterbi, _ = self.glhmm_models[S]
            
            # Concatenate all Viterbi paths into single array
            original_labels = np.concatenate(original_viterbi)
            
            # Note: GLHMM bootstrap is more complex due to temporal structure
            # This is a simplified implementation that treats timepoints as independent
            # A more sophisticated approach would resample entire sessions
            
            logger.info("  GLHMM bootstrap stability analysis requires session-level "
                       "resampling and is computationally intensive.")
            logger.info("  Skipping GLHMM bootstrap for now. Consider implementing "
                       "session-level bootstrap if needed.")
            
            # Placeholder: could implement session-level bootstrap here
            # For now, we skip GLHMM bootstrap
        
        # Create results DataFrame
        self.bootstrap_results = pd.DataFrame(stability_results)
        
        logger.info("Bootstrap stability analysis complete")
        
        return self.bootstrap_results

    def compute_state_metrics(self) -> pd.DataFrame:
        """
        Compute state occupancy metrics for each subject-session combination.
        
        For each subject-session and each cluster/state:
        - Fractional occupancy: Proportion of time spent in each state
        - Number of visits: Count of transitions into each state
        - Mean dwell time: Average consecutive time bins in each state
        
        Metrics are computed for both:
        - KMeans hard labels (optimal k)
        - GLHMM Viterbi paths (optimal S)
        
        Metric Interpretations:
            - Fractional occupancy: Indicates how dominant a state is in the session
            - Number of visits: Indicates state switching frequency
            - Mean dwell time: Indicates state stability/persistence
        
        Returns:
            pd.DataFrame: State metrics with columns:
                - subject: Subject identifier
                - session_id: Session identifier
                - state: Experimental state (RS or DMT)
                - dose: Dose level (Baja or Alta)
                - method: 'KMeans' or 'GLHMM'
                - cluster_state: Cluster/state identifier (0, 1, 2, ...)
                - fractional_occupancy: Proportion of time in state
                - n_visits: Number of visits to state
                - mean_dwell_time: Mean consecutive time bins in state
        """
        if self.optimal_kmeans_states is None and self.optimal_glhmm_states is None:
            raise ValueError("Must select optimal models before computing metrics. "
                           "Call select_optimal_models() first.")
        
        logger.info("Computing state occupancy metrics...")
        
        metrics_list = []
        
        # Compute metrics for KMeans
        if self.optimal_kmeans_states is not None:
            k = self.optimal_kmeans_states
            logger.info(f"Computing metrics for KMeans k={k}...")
            
            # Get KMeans labels
            _, kmeans_labels, _ = self.kmeans_models[k]
            
            # Add labels to data
            data_with_labels = self.data.copy()
            data_with_labels['cluster_state'] = kmeans_labels
            
            # Group by subject and session
            session_groups = data_with_labels.groupby([
                self.subject_id_col, 
                self.session_id_col,
                'state',
                'dose'
            ])
            
            for (subject, session_id, exp_state, dose), session_data in session_groups:
                # Sort by time
                session_data = session_data.sort_values(self.time_col)
                labels = session_data['cluster_state'].values
                
                # Compute metrics for each cluster state
                for cluster_state in range(k):
                    # Fractional occupancy
                    fractional_occupancy = np.mean(labels == cluster_state)
                    
                    # Number of visits (state transitions)
                    # A visit is when we enter a state from a different state
                    state_changes = np.diff(labels == cluster_state)
                    n_visits = np.sum(state_changes == 1)
                    
                    # If session starts in this state, count it as a visit
                    if labels[0] == cluster_state:
                        n_visits += 1
                    
                    # Mean dwell time
                    # Find consecutive runs of this state
                    is_state = (labels == cluster_state).astype(int)
                    
                    # Find run lengths
                    dwell_times = []
                    current_run = 0
                    
                    for i in range(len(is_state)):
                        if is_state[i] == 1:
                            current_run += 1
                        else:
                            if current_run > 0:
                                dwell_times.append(current_run)
                                current_run = 0
                    
                    # Don't forget last run
                    if current_run > 0:
                        dwell_times.append(current_run)
                    
                    mean_dwell_time = np.mean(dwell_times) if len(dwell_times) > 0 else 0
                    
                    # Store metrics
                    metrics_list.append({
                        'subject': subject,
                        'session_id': session_id,
                        'state': exp_state,
                        'dose': dose,
                        'method': 'KMeans',
                        'cluster_state': cluster_state,
                        'fractional_occupancy': fractional_occupancy,
                        'n_visits': n_visits,
                        'mean_dwell_time': mean_dwell_time
                    })
            
            logger.info(f"  Computed KMeans metrics for {len(session_groups)} sessions")
        
        # Compute metrics for GLHMM
        if self.optimal_glhmm_states is not None:
            S = self.optimal_glhmm_states
            logger.info(f"Computing metrics for GLHMM S={S}...")
            
            # Get GLHMM Viterbi paths
            _, viterbi_paths, _ = self.glhmm_models[S]
            
            # Match Viterbi paths to sessions
            session_groups = self.data.groupby([
                self.subject_id_col, 
                self.session_id_col,
                'state',
                'dose'
            ])
            
            for idx, ((subject, session_id, exp_state, dose), session_data) in enumerate(session_groups):
                if idx >= len(viterbi_paths):
                    logger.warning(f"Missing Viterbi path for session {idx}")
                    continue
                
                # Get Viterbi path for this session
                labels = viterbi_paths[idx]
                
                # Compute metrics for each GLHMM state
                for glhmm_state in range(S):
                    # Fractional occupancy
                    fractional_occupancy = np.mean(labels == glhmm_state)
                    
                    # Number of visits
                    state_changes = np.diff(labels == glhmm_state)
                    n_visits = np.sum(state_changes == 1)
                    
                    if labels[0] == glhmm_state:
                        n_visits += 1
                    
                    # Mean dwell time
                    is_state = (labels == glhmm_state).astype(int)
                    dwell_times = []
                    current_run = 0
                    
                    for i in range(len(is_state)):
                        if is_state[i] == 1:
                            current_run += 1
                        else:
                            if current_run > 0:
                                dwell_times.append(current_run)
                                current_run = 0
                    
                    if current_run > 0:
                        dwell_times.append(current_run)
                    
                    mean_dwell_time = np.mean(dwell_times) if len(dwell_times) > 0 else 0
                    
                    # Store metrics
                    metrics_list.append({
                        'subject': subject,
                        'session_id': session_id,
                        'state': exp_state,
                        'dose': dose,
                        'method': 'GLHMM',
                        'cluster_state': glhmm_state,
                        'fractional_occupancy': fractional_occupancy,
                        'n_visits': n_visits,
                        'mean_dwell_time': mean_dwell_time
                    })
            
            logger.info(f"  Computed GLHMM metrics for {len(session_groups)} sessions")
        
        # Create metrics DataFrame
        self.occupancy_measures = pd.DataFrame(metrics_list)
        
        # Log summary statistics
        if len(self.occupancy_measures) > 0:
            logger.info(f"Computed {len(self.occupancy_measures)} state metric rows")
            logger.info(f"  Fractional occupancy range: "
                       f"[{self.occupancy_measures['fractional_occupancy'].min():.3f}, "
                       f"{self.occupancy_measures['fractional_occupancy'].max():.3f}]")
            logger.info(f"  Mean dwell time range: "
                       f"[{self.occupancy_measures['mean_dwell_time'].min():.1f}, "
                       f"{self.occupancy_measures['mean_dwell_time'].max():.1f}]")
        
        return self.occupancy_measures

    def export_results(self, output_dir: str) -> Dict[str, str]:
        """
        Export clustering and state modelling results to CSV files.
        
        Creates output directory if it doesn't exist and exports:
        - clustering_kmeans_assignments.csv: KMeans cluster assignments and probabilities
        - clustering_glhmm_viterbi.csv: GLHMM Viterbi paths (hard state assignments)
        - clustering_glhmm_probabilities.csv: GLHMM posterior probabilities (gamma)
        - clustering_state_metrics.csv: State occupancy metrics
        - clustering_evaluation.csv: Model evaluation results
        - clustering_bootstrap_stability.csv: Bootstrap stability results
        
        Args:
            output_dir (str): Directory to save output files
        
        Returns:
            Dict[str, str]: Dictionary mapping file types to file paths
        """
        import os
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        output_paths = {}
        
        # Export KMeans assignments and probabilities
        if self.optimal_kmeans_states is not None:
            k = self.optimal_kmeans_states
            _, hard_labels, soft_probs = self.kmeans_models[k]
            
            # Create DataFrame with assignments
            kmeans_df = self.data[[
                self.subject_id_col, 
                self.session_id_col, 
                'state', 
                'dose', 
                self.time_col
            ]].copy()
            kmeans_df['cluster'] = hard_labels
            
            # Add soft probabilities
            for i in range(k):
                kmeans_df[f'prob_cluster_{i}'] = soft_probs[:, i]
            
            # Export
            kmeans_path = os.path.join(output_dir, 'clustering_kmeans_assignments.csv')
            kmeans_df.to_csv(kmeans_path, index=False)
            output_paths['kmeans_assignments'] = kmeans_path
            logger.info(f"Exported KMeans assignments to: {kmeans_path}")
        
        # Export GLHMM Viterbi paths
        if self.optimal_glhmm_states is not None:
            S = self.optimal_glhmm_states
            _, viterbi_paths, gamma_probs = self.glhmm_models[S]
            
            # Concatenate Viterbi paths with metadata
            viterbi_list = []
            gamma_list = []
            
            session_groups = self.data.groupby([
                self.subject_id_col, 
                self.session_id_col
            ])
            
            for idx, ((subject, session_id), session_data) in enumerate(session_groups):
                if idx >= len(viterbi_paths):
                    continue
                
                # Get session metadata
                session_data = session_data.sort_values(self.time_col)
                
                # Viterbi paths
                viterbi_df = session_data[[
                    self.subject_id_col, 
                    self.session_id_col, 
                    'state', 
                    'dose', 
                    self.time_col
                ]].copy()
                viterbi_df['viterbi_state'] = viterbi_paths[idx]
                viterbi_list.append(viterbi_df)
                
                # Gamma probabilities
                gamma_df = session_data[[
                    self.subject_id_col, 
                    self.session_id_col, 
                    'state', 
                    'dose', 
                    self.time_col
                ]].copy()
                
                for i in range(S):
                    gamma_df[f'gamma_state_{i}'] = gamma_probs[idx][:, i]
                
                gamma_list.append(gamma_df)
            
            # Export Viterbi paths
            if len(viterbi_list) > 0:
                viterbi_full = pd.concat(viterbi_list, ignore_index=True)
                viterbi_path = os.path.join(output_dir, 'clustering_glhmm_viterbi.csv')
                viterbi_full.to_csv(viterbi_path, index=False)
                output_paths['glhmm_viterbi'] = viterbi_path
                logger.info(f"Exported GLHMM Viterbi paths to: {viterbi_path}")
            
            # Export gamma probabilities
            if len(gamma_list) > 0:
                gamma_full = pd.concat(gamma_list, ignore_index=True)
                gamma_path = os.path.join(output_dir, 'clustering_glhmm_probabilities.csv')
                gamma_full.to_csv(gamma_path, index=False)
                output_paths['glhmm_probabilities'] = gamma_path
                logger.info(f"Exported GLHMM probabilities to: {gamma_path}")
        
        # Export state occupancy metrics
        if self.occupancy_measures is not None:
            metrics_path = os.path.join(output_dir, 'clustering_state_metrics.csv')
            self.occupancy_measures.to_csv(metrics_path, index=False)
            output_paths['state_metrics'] = metrics_path
            logger.info(f"Exported state metrics to: {metrics_path}")
        
        # Export evaluation results
        if self.evaluation_results is not None:
            eval_path = os.path.join(output_dir, 'clustering_evaluation.csv')
            self.evaluation_results.to_csv(eval_path, index=False)
            output_paths['evaluation'] = eval_path
            logger.info(f"Exported evaluation results to: {eval_path}")
        
        # Export bootstrap stability results
        if self.bootstrap_results is not None:
            bootstrap_path = os.path.join(output_dir, 'clustering_bootstrap_stability.csv')
            self.bootstrap_results.to_csv(bootstrap_path, index=False)
            output_paths['bootstrap_stability'] = bootstrap_path
            logger.info(f"Exported bootstrap stability to: {bootstrap_path}")
        
        return output_paths
