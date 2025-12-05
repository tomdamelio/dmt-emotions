# -*- coding: utf-8 -*-
"""
TET State Visualization Module

This module provides visualization functionality for clustering and state modelling results,
including centroid profiles, time-course probability plots, and correspondence analyses
between KMeans and GLHMM solutions.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, List, Dict, Tuple
import logging
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TETStateVisualization:
    """
    Visualizes clustering and state modelling results for TET data.
    
    This class generates publication-ready figures for:
    - KMeans centroid profiles (replicating Fig. 3.5)
    - Time-course cluster probability plots (replicating Fig. 3.6)
    - GLHMM state time-course plots
    - KMeans-GLHMM correspondence heatmaps
    
    Interpretation of Centroid Profiles:
        Centroid profiles show the characteristic experiential "signature" of each
        cluster. Each bar represents the normalized contribution of a dimension to
        that cluster's profile:
        
        - Positive values (blue): Dimensions that are elevated in this state relative
          to the cluster's overall intensity. These are the defining features of the
          experiential state.
        
        - Negative values (red): Dimensions that are suppressed in this state. These
          represent experiences that are anti-characteristic of the state.
        
        - Magnitude: Larger absolute values indicate dimensions that are more
          important for defining the cluster. Small values indicate dimensions that
          are relatively neutral for that state.
        
        Normalization by maximum value allows comparison of relative dimension
        importance within each cluster, independent of absolute intensity differences
        between clusters.
    
    Interpretation of State Probability Time Courses:
        Time-course plots show how the probability of being in each state evolves
        over time, averaged across subjects:
        
        - Y-axis: Probability of cluster/state membership at each time point.
          For KMeans soft probabilities, this is based on normalized inverse
          distances to cluster centers. For GLHMM, this is the posterior probability
          (gamma) from the forward-backward algorithm.
        
        - Shaded regions: Standard error of the mean (SEM) across subjects. Wider
          bands indicate greater inter-subject variability.
        
        - Temporal patterns:
          * Stable high probability: Persistent, dominant state
          * Gradual increase/decrease: Smooth state transitions
          * Rapid changes: Abrupt state switching
          * Oscillations: Alternating between states
        
        - Dose comparisons: Differences between High and Low dose curves indicate
          dose-dependent modulation of state dynamics. Divergence at specific time
          points suggests dose effects emerge at those times.
    
    KMeans vs GLHMM Correspondence:
        The correspondence heatmap shows how KMeans clusters map onto GLHMM states:
        
        - High correspondence (warm colors): KMeans cluster and GLHMM state identify
          similar experiential patterns. This suggests the static and temporal models
          converge on similar state definitions.
        
        - Low correspondence (cool colors): KMeans cluster and GLHMM state capture
          different aspects of the data. This could indicate that temporal structure
          is important for defining states, or that the two methods identify states
          at different levels of granularity.
        
        - One-to-one mapping: Each KMeans cluster corresponds primarily to one GLHMM
          state. This suggests the two methods identify equivalent states.
        
        - Many-to-one mapping: Multiple GLHMM states map to one KMeans cluster. This
          suggests GLHMM identifies finer-grained temporal substates within a broader
          KMeans cluster.
    
    Attributes:
        data (pd.DataFrame): Preprocessed TET data with z-scored dimensions
        kmeans_assignments (pd.DataFrame): KMeans cluster assignments and soft probabilities
        glhmm_viterbi (pd.DataFrame): GLHMM Viterbi paths (hard state assignments)
        glhmm_probabilities (pd.DataFrame): GLHMM posterior probabilities (gamma)
        dimensions (List[str]): List of z-scored dimension column names
    
    Example:
        >>> import pandas as pd
        >>> from tet.state_visualization import TETStateVisualization
        >>> 
        >>> # Load data
        >>> data = pd.read_csv('results/tet/tet_preprocessed.csv')
        >>> kmeans_assignments = pd.read_csv('results/tet/clustering/clustering_kmeans_assignments.csv')
        >>> glhmm_viterbi = pd.read_csv('results/tet/clustering/clustering_glhmm_viterbi.csv')
        >>> glhmm_probs = pd.read_csv('results/tet/clustering/clustering_glhmm_probabilities.csv')
        >>> 
        >>> # Initialize visualizer
        >>> viz = TETStateVisualization(
        ...     data=data,
        ...     kmeans_assignments=kmeans_assignments,
        ...     glhmm_viterbi=glhmm_viterbi,
        ...     glhmm_probabilities=glhmm_probs
        ... )
        >>> 
        >>> # Generate plots
        >>> viz.plot_kmeans_centroid_profiles(output_dir='results/tet/figures')
        >>> viz.plot_kmeans_cluster_timecourses(output_dir='results/tet/figures')
        >>> viz.plot_glhmm_state_timecourses(output_dir='results/tet/figures')
        >>> viz.plot_kmeans_glhmm_crosswalk(output_dir='results/tet/figures')
    """
    
    def __init__(
        self,
        data: pd.DataFrame,
        kmeans_assignments: Optional[pd.DataFrame] = None,
        glhmm_viterbi: Optional[pd.DataFrame] = None,
        glhmm_probabilities: Optional[pd.DataFrame] = None
    ):
        """
        Initialize state visualization with clustering results.
        
        Args:
            data (pd.DataFrame): Preprocessed TET data with z-scored dimensions
            kmeans_assignments (pd.DataFrame, optional): KMeans cluster assignments
                with columns: subject, session_id, state, dose, t_bin, cluster, prob_cluster_*
            glhmm_viterbi (pd.DataFrame, optional): GLHMM Viterbi paths
                with columns: subject, session_id, state, dose, t_bin, viterbi_state
            glhmm_probabilities (pd.DataFrame, optional): GLHMM posterior probabilities
                with columns: subject, session_id, state, dose, t_bin, gamma_state_*
        """
        self.data = data.copy()
        self.kmeans_assignments = kmeans_assignments.copy() if kmeans_assignments is not None else None
        self.glhmm_viterbi = glhmm_viterbi.copy() if glhmm_viterbi is not None else None
        self.glhmm_probabilities = glhmm_probabilities.copy() if glhmm_probabilities is not None else None
        
        # Identify z-scored dimensions
        self.dimensions = [col for col in data.columns if col.endswith('_z') and 
                          col not in ['valence_index_z']]
        
        logger.info(f"Initialized TETStateVisualization with {len(data)} rows")
        logger.info(f"  Z-scored dimensions: {len(self.dimensions)}")
        
        if kmeans_assignments is not None:
            logger.info(f"  KMeans assignments: {len(kmeans_assignments)} rows")
        if glhmm_viterbi is not None:
            logger.info(f"  GLHMM Viterbi paths: {len(glhmm_viterbi)} rows")
        if glhmm_probabilities is not None:
            logger.info(f"  GLHMM probabilities: {len(glhmm_probabilities)} rows")

    def plot_kmeans_centroid_profiles(
        self,
        k: int = 2,
        output_dir: Optional[str] = None,
        figsize: Tuple[float, float] = (12, 6),
        dpi: int = 300
    ) -> str:
        """
        Plot KMeans centroid profiles (replicating Fig. 3.5).
        
        This method visualizes the characteristic profiles of each cluster by:
        1. Computing centroid coordinates in original z-scored dimensions
        2. Normalizing each centroid by its maximum dimension value to [0, 1]
        3. Plotting normalized contributions per dimension for each cluster
        
        The normalization allows comparison of relative dimension importance
        within each cluster, independent of absolute intensity.
        
        Args:
            k (int): Number of clusters (default: 2)
            output_dir (str, optional): Directory to save figure
            figsize (Tuple[float, float]): Figure size in inches (default: (12, 6))
            dpi (int): Resolution for saved figure (default: 300)
        
        Returns:
            str: Path to saved figure (if output_dir provided)
        
        Raises:
            ValueError: If KMeans assignments not available or k not found
        """
        if self.kmeans_assignments is None:
            raise ValueError("KMeans assignments not available. "
                           "Provide kmeans_assignments in __init__.")
        
        logger.info(f"Plotting KMeans centroid profiles for k={k}...")
        
        # Check if cluster column exists
        if 'cluster' not in self.kmeans_assignments.columns:
            raise ValueError("KMeans assignments missing 'cluster' column")
        
        # Verify k matches available clusters
        n_clusters = self.kmeans_assignments['cluster'].nunique()
        if k != n_clusters:
            logger.warning(f"Requested k={k} but assignments have {n_clusters} clusters. "
                          f"Using k={n_clusters}")
            k = n_clusters
        
        # Merge assignments with data to get z-scored dimensions
        merged = self.kmeans_assignments.merge(
            self.data[['subject', 'session_id', 't_bin'] + self.dimensions],
            on=['subject', 'session_id', 't_bin'],
            how='left'
        )
        
        # Compute centroid coordinates (mean z-score per dimension per cluster)
        centroids = []
        
        for cluster_id in range(k):
            cluster_data = merged[merged['cluster'] == cluster_id]
            centroid = cluster_data[self.dimensions].mean().values
            centroids.append(centroid)
        
        centroids = np.array(centroids)  # Shape: (k, n_dimensions)
        
        logger.info(f"  Computed centroids shape: {centroids.shape}")
        
        # Normalize each centroid by its maximum absolute value
        centroids_normalized = np.zeros_like(centroids)
        
        for i in range(k):
            max_val = np.max(np.abs(centroids[i]))
            if max_val > 0:
                centroids_normalized[i] = centroids[i] / max_val
            else:
                centroids_normalized[i] = centroids[i]
        
        # Create dimension labels (remove '_z' suffix for cleaner display)
        dimension_labels = [dim.replace('_z', '').replace('_', ' ').title() 
                           for dim in self.dimensions]
        
        # Create figure
        fig, axes = plt.subplots(1, k, figsize=figsize, sharey=True)
        
        if k == 1:
            axes = [axes]
        
        # Plot each cluster's profile
        for cluster_id in range(k):
            ax = axes[cluster_id]
            
            # Get normalized centroid values
            values = centroids_normalized[cluster_id]
            
            # Create bar plot
            colors = ['red' if v < 0 else 'blue' for v in values]
            bars = ax.barh(range(len(values)), values, color=colors, alpha=0.7)
            
            # Customize axis
            ax.set_yticks(range(len(dimension_labels)))
            ax.set_yticklabels(dimension_labels, fontsize=9)
            ax.set_xlabel('Normalized Contribution', fontsize=11)
            ax.set_title(f'Cluster {cluster_id}', fontsize=12, fontweight='bold')
            ax.axvline(0, color='black', linewidth=0.8, linestyle='-')
            ax.set_xlim(-1.1, 1.1)
            ax.grid(axis='x', alpha=0.3)
            
            # Show y-labels on all plots for better readability
            # (removed the condition that hid labels on right plots)
        
        plt.tight_layout()
        
        # Save figure if output directory provided
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f'clustering_kmeans_centroids_k{k}.png')
            plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
            logger.info(f"  Saved centroid profile plot to: {output_path}")
            plt.close()
            return output_path
        else:
            plt.show()
            return None

    def plot_kmeans_cluster_timecourses(
        self,
        k: int = 2,
        include_rs: bool = True,
        output_dir: Optional[str] = None,
        figsize: Tuple[float, float] = (14, 8),
        dpi: int = 300
    ) -> str:
        """
        Plot time-course cluster probability plots (replicating Fig. 3.6).
        
        This method visualizes how cluster probabilities evolve over time by:
        1. Computing mean cluster probability at each time point
        2. Averaging across subjects separately for High and Low dose in DMT
        3. Optionally including RS as comparison
        4. Plotting probability curves with shaded SEM error bands
        
        Args:
            k (int): Number of clusters (default: 2)
            include_rs (bool): Include RS condition as comparison (default: True)
            output_dir (str, optional): Directory to save figure
            figsize (Tuple[float, float]): Figure size in inches (default: (14, 8))
            dpi (int): Resolution for saved figure (default: 300)
        
        Returns:
            str: Path to saved figure (if output_dir provided)
        
        Raises:
            ValueError: If KMeans assignments not available or k not found
        """
        if self.kmeans_assignments is None:
            raise ValueError("KMeans assignments not available. "
                           "Provide kmeans_assignments in __init__.")
        
        logger.info(f"Plotting KMeans cluster time courses for k={k}...")
        
        # Verify k matches available clusters
        n_clusters = self.kmeans_assignments['cluster'].nunique()
        if k != n_clusters:
            logger.warning(f"Requested k={k} but assignments have {n_clusters} clusters. "
                          f"Using k={n_clusters}")
            k = n_clusters
        
        # Get probability columns
        prob_cols = [f'prob_cluster_{i}' for i in range(k)]
        
        # Check if probability columns exist
        missing_cols = [col for col in prob_cols if col not in self.kmeans_assignments.columns]
        if missing_cols:
            raise ValueError(f"Missing probability columns: {missing_cols}")
        
        # Add time in seconds if not present
        if 't_sec' not in self.kmeans_assignments.columns:
            # Assume 0.25 Hz sampling (4 seconds per bin)
            self.kmeans_assignments['t_sec'] = self.kmeans_assignments['t_bin'] * 4
        
        # Filter data
        if include_rs:
            states_to_plot = ['RS', 'DMT']
        else:
            states_to_plot = ['DMT']
        
        data_filtered = self.kmeans_assignments[
            self.kmeans_assignments['state'].isin(states_to_plot)
        ].copy()
        
        # Convert time to minutes
        data_filtered['t_min'] = data_filtered['t_sec'] / 60.0
        
        # Determine number of subplots
        n_conditions = len(states_to_plot) * 2  # High and Low for each state
        n_cols = 2
        n_rows = (n_conditions + n_cols - 1) // n_cols
        
        # Create figure
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, sharex=True, sharey=True)
        axes = axes.flatten() if n_conditions > 1 else [axes]
        
        # Define colors for clusters
        cluster_colors = plt.cm.Set2(np.linspace(0, 1, k))
        
        # Plot each condition
        plot_idx = 0
        
        for state in states_to_plot:
            for dose in ['Baja', 'Alta']:  # Low, High
                if plot_idx >= len(axes):
                    break
                
                ax = axes[plot_idx]
                
                # Filter data for this condition
                condition_data = data_filtered[
                    (data_filtered['state'] == state) & 
                    (data_filtered['dose'] == dose)
                ]
                
                if len(condition_data) == 0:
                    logger.warning(f"No data for {state} {dose}")
                    plot_idx += 1
                    continue
                
                # Compute mean and SEM for each cluster at each time point
                for cluster_id in range(k):
                    prob_col = f'prob_cluster_{cluster_id}'
                    
                    # Group by time and compute statistics
                    time_stats = condition_data.groupby('t_min')[prob_col].agg(['mean', 'sem']).reset_index()
                    
                    # Plot mean with shaded SEM
                    ax.plot(
                        time_stats['t_min'],
                        time_stats['mean'],
                        color=cluster_colors[cluster_id],
                        linewidth=2,
                        label=f'Cluster {cluster_id}'
                    )
                    
                    ax.fill_between(
                        time_stats['t_min'],
                        time_stats['mean'] - time_stats['sem'],
                        time_stats['mean'] + time_stats['sem'],
                        color=cluster_colors[cluster_id],
                        alpha=0.2
                    )
                
                # Customize axis
                dose_label = 'High (40mg)' if dose == 'Alta' else 'Low (20mg)'
                ax.set_title(f'{state} - {dose_label}', fontsize=12, fontweight='bold')
                ax.set_xlabel('Time (minutes)', fontsize=11)
                ax.set_ylabel('Cluster Probability', fontsize=11)
                ax.set_ylim(0, 1)
                ax.grid(alpha=0.3)
                ax.legend(loc='best', fontsize=9)
                
                plot_idx += 1
        
        # Hide unused subplots
        for idx in range(plot_idx, len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        
        # Save figure if output directory provided
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
            suffix = '_with_rs' if include_rs else '_dmt_only'
            output_path = os.path.join(
                output_dir, 
                f'clustering_kmeans_prob_timecourses{suffix}.png'
            )
            plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
            logger.info(f"  Saved cluster time course plot to: {output_path}")
            plt.close()
            return output_path
        else:
            plt.show()
            return None

    def plot_glhmm_state_timecourses(
        self,
        states_to_plot: Optional[List[str]] = None,
        subset_states: Optional[List[int]] = None,
        output_dir: Optional[str] = None,
        figsize: Tuple[float, float] = (14, 8),
        dpi: int = 300
    ) -> str:
        """
        Plot GLHMM state time-course probability plots.
        
        This method visualizes how GLHMM state probabilities (gamma) evolve over time by:
        1. Computing mean state probability at each time point
        2. Averaging across subjects separately for High and Low dose
        3. Plotting state probability curves with shaded SEM error bands
        4. Allowing filtering by experimental state (RS vs DMT) and subset of GLHMM states
        
        Args:
            states_to_plot (List[str], optional): Experimental states to include 
                (default: ['RS', 'DMT'])
            subset_states (List[int], optional): Subset of GLHMM states to plot
                (e.g., [0, 1] for S=2 solution). If None, plots all states.
            output_dir (str, optional): Directory to save figure
            figsize (Tuple[float, float]): Figure size in inches (default: (14, 8))
            dpi (int): Resolution for saved figure (default: 300)
        
        Returns:
            str: Path to saved figure (if output_dir provided)
        
        Raises:
            ValueError: If GLHMM probabilities not available
        """
        if self.glhmm_probabilities is None:
            raise ValueError("GLHMM probabilities not available. "
                           "Provide glhmm_probabilities in __init__.")
        
        logger.info("Plotting GLHMM state time courses...")
        
        # Default to all experimental states
        if states_to_plot is None:
            states_to_plot = ['RS', 'DMT']
        
        # Get gamma columns
        gamma_cols = [col for col in self.glhmm_probabilities.columns 
                     if col.startswith('gamma_state_')]
        
        if len(gamma_cols) == 0:
            raise ValueError("No gamma_state_* columns found in GLHMM probabilities")
        
        # Determine number of GLHMM states
        n_states = len(gamma_cols)
        logger.info(f"  Found {n_states} GLHMM states")
        
        # Filter to subset of states if requested
        if subset_states is not None:
            gamma_cols = [f'gamma_state_{i}' for i in subset_states if i < n_states]
            logger.info(f"  Plotting subset of states: {subset_states}")
        
        # Add time in seconds if not present
        if 't_sec' not in self.glhmm_probabilities.columns:
            # Assume 0.25 Hz sampling (4 seconds per bin)
            self.glhmm_probabilities['t_sec'] = self.glhmm_probabilities['t_bin'] * 4
        
        # Filter data
        data_filtered = self.glhmm_probabilities[
            self.glhmm_probabilities['state'].isin(states_to_plot)
        ].copy()
        
        # Convert time to minutes
        data_filtered['t_min'] = data_filtered['t_sec'] / 60.0
        
        # Determine number of subplots
        n_conditions = len(states_to_plot) * 2  # High and Low for each state
        n_cols = 2
        n_rows = (n_conditions + n_cols - 1) // n_cols
        
        # Create figure
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, sharex=True, sharey=True)
        axes = axes.flatten() if n_conditions > 1 else [axes]
        
        # Define colors for states
        state_colors = plt.cm.Set1(np.linspace(0, 1, len(gamma_cols)))
        
        # Plot each condition
        plot_idx = 0
        
        for exp_state in states_to_plot:
            for dose in ['Baja', 'Alta']:  # Low, High
                if plot_idx >= len(axes):
                    break
                
                ax = axes[plot_idx]
                
                # Filter data for this condition
                condition_data = data_filtered[
                    (data_filtered['state'] == exp_state) & 
                    (data_filtered['dose'] == dose)
                ]
                
                if len(condition_data) == 0:
                    logger.warning(f"No data for {exp_state} {dose}")
                    plot_idx += 1
                    continue
                
                # Compute mean and SEM for each GLHMM state at each time point
                for idx, gamma_col in enumerate(gamma_cols):
                    state_id = int(gamma_col.split('_')[-1])
                    
                    # Group by time and compute statistics
                    time_stats = condition_data.groupby('t_min')[gamma_col].agg(['mean', 'sem']).reset_index()
                    
                    # Plot mean with shaded SEM
                    ax.plot(
                        time_stats['t_min'],
                        time_stats['mean'],
                        color=state_colors[idx],
                        linewidth=2,
                        label=f'State {state_id}'
                    )
                    
                    ax.fill_between(
                        time_stats['t_min'],
                        time_stats['mean'] - time_stats['sem'],
                        time_stats['mean'] + time_stats['sem'],
                        color=state_colors[idx],
                        alpha=0.2
                    )
                
                # Customize axis
                dose_label = 'High (40mg)' if dose == 'Alta' else 'Low (20mg)'
                ax.set_title(f'{exp_state} - {dose_label}', fontsize=12, fontweight='bold')
                ax.set_xlabel('Time (minutes)', fontsize=11)
                ax.set_ylabel('State Probability (γ)', fontsize=11)
                ax.set_ylim(0, 1)
                ax.grid(alpha=0.3)
                ax.legend(loc='best', fontsize=9)
                
                plot_idx += 1
        
        # Hide unused subplots
        for idx in range(plot_idx, len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        
        # Save figure if output directory provided
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, 'glhmm_state_prob_timecourses.png')
            plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
            logger.info(f"  Saved GLHMM state time course plot to: {output_path}")
            plt.close()
            return output_path
        else:
            plt.show()
            return None

    def plot_kmeans_glhmm_crosswalk(
        self,
        k: int = 2,
        output_dir: Optional[str] = None,
        figsize: Tuple[float, float] = (8, 6),
        dpi: int = 300
    ) -> Tuple[str, str]:
        """
        Plot KMeans-GLHMM correspondence heatmap and export contingency table.
        
        This method analyzes the correspondence between KMeans clusters and GLHMM states by:
        1. Computing contingency table between cluster assignments and state assignments
        2. Normalizing contingency to probabilities
        3. Visualizing correspondence using a heatmap
        4. Exporting underlying contingency table as CSV
        
        The correspondence analysis helps understand whether the two methods
        identify similar experiential states or capture different aspects of the data.
        
        Args:
            k (int): Number of KMeans clusters (default: 2)
            output_dir (str, optional): Directory to save figure and CSV
            figsize (Tuple[float, float]): Figure size in inches (default: (8, 6))
            dpi (int): Resolution for saved figure (default: 300)
        
        Returns:
            Tuple[str, str]: Paths to (figure, CSV) if output_dir provided, else (None, None)
        
        Raises:
            ValueError: If KMeans assignments or GLHMM Viterbi paths not available
        """
        if self.kmeans_assignments is None:
            raise ValueError("KMeans assignments not available. "
                           "Provide kmeans_assignments in __init__.")
        
        if self.glhmm_viterbi is None:
            raise ValueError("GLHMM Viterbi paths not available. "
                           "Provide glhmm_viterbi in __init__.")
        
        logger.info(f"Computing KMeans-GLHMM correspondence for k={k}...")
        
        # Merge KMeans and GLHMM assignments on common keys
        merged = self.kmeans_assignments.merge(
            self.glhmm_viterbi,
            on=['subject', 'session_id', 'state', 'dose', 't_bin'],
            how='inner'
        )
        
        if len(merged) == 0:
            raise ValueError("No overlapping time points between KMeans and GLHMM assignments")
        
        logger.info(f"  Found {len(merged)} overlapping time points")
        
        # Get cluster and state assignments
        kmeans_clusters = merged['cluster'].values
        glhmm_states = merged['viterbi_state'].values
        
        # Determine number of GLHMM states
        n_glhmm_states = len(np.unique(glhmm_states))
        
        logger.info(f"  KMeans clusters: {k}")
        logger.info(f"  GLHMM states: {n_glhmm_states}")
        
        # Compute contingency table
        contingency = np.zeros((k, n_glhmm_states))
        
        for i in range(k):
            for j in range(n_glhmm_states):
                contingency[i, j] = np.sum((kmeans_clusters == i) & (glhmm_states == j))
        
        # Normalize to probabilities (row-wise: P(GLHMM state | KMeans cluster))
        contingency_prob = contingency / contingency.sum(axis=1, keepdims=True)
        
        # Create contingency DataFrame for export
        contingency_df = pd.DataFrame(
            contingency,
            index=[f'KMeans_Cluster_{i}' for i in range(k)],
            columns=[f'GLHMM_State_{j}' for j in range(n_glhmm_states)]
        )
        
        contingency_prob_df = pd.DataFrame(
            contingency_prob,
            index=[f'KMeans_Cluster_{i}' for i in range(k)],
            columns=[f'GLHMM_State_{j}' for j in range(n_glhmm_states)]
        )
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=figsize)
        
        sns.heatmap(
            contingency_prob_df,
            annot=True,
            fmt='.3f',
            cmap='YlOrRd',
            cbar_kws={'label': 'P(GLHMM State | KMeans Cluster)'},
            ax=ax,
            vmin=0,
            vmax=1
        )
        
        ax.set_xlabel('GLHMM State', fontsize=12, fontweight='bold')
        ax.set_ylabel('KMeans Cluster', fontsize=12, fontweight='bold')
        ax.set_title('KMeans-GLHMM Correspondence', fontsize=13, fontweight='bold')
        
        plt.tight_layout()
        
        # Save outputs if directory provided
        fig_path = None
        csv_path = None
        
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
            
            # Save figure
            fig_path = os.path.join(output_dir, 'kmeans_glhmm_crosswalk.png')
            plt.savefig(fig_path, dpi=dpi, bbox_inches='tight')
            logger.info(f"  Saved correspondence heatmap to: {fig_path}")
            
            # Save contingency tables
            csv_path = os.path.join(output_dir, 'kmeans_glhmm_crosswalk.csv')
            
            # Combine counts and probabilities in single CSV
            combined_df = pd.DataFrame({
                'KMeans_Cluster': list(range(k)) * n_glhmm_states,
                'GLHMM_State': [j for j in range(n_glhmm_states) for _ in range(k)],
                'Count': contingency.T.flatten(),
                'Probability': contingency_prob.T.flatten()
            })
            
            combined_df.to_csv(csv_path, index=False)
            logger.info(f"  Saved contingency table to: {csv_path}")
            
            plt.close()
        else:
            plt.show()
        
        return fig_path, csv_path
