"""
TET Results Synthesizer

This module provides functionality to synthesize TET analysis results into a
comprehensive markdown report.
"""

import logging
from pathlib import Path
from typing import Dict, Optional, List
import pandas as pd
import numpy as np
from datetime import datetime

from .results_formatter import StatisticalFormatter
from .results_analyzer import (
    rank_findings,
    extract_descriptive_stats,
    extract_lme_effects,
    interpret_pca_components,
    extract_clustering_results,
    compute_cross_method_rankings,
    extract_methodological_metadata,
    detect_ambiguous_findings
)

logger = logging.getLogger(__name__)


class TETResultsSynthesizer:
    """
    Synthesize TET analysis results into comprehensive report.
    
    This class loads results from all TET analysis components (descriptive stats,
    LME models, peak/AUC analysis, PCA, clustering) and generates a structured
    markdown document at docs/tet_comprehensive_results.md.
    
    Parameters
    ----------
    results_dir : str, optional
        Path to TET results directory, by default 'results/tet'
    output_path : str, optional
        Path to output markdown file, by default 'docs/tet_comprehensive_results.md'
    
    Attributes
    ----------
    results_dir : Path
        Path to results directory
    output_path : Path
        Path to output document
    descriptive_data : dict or None
        Loaded descriptive statistics
    lme_results : pd.DataFrame or None
        Loaded LME coefficients
    pca_results : dict or None
        Loaded PCA results (loadings, variance, LME)
    clustering_results : dict or None
        Loaded clustering results (assignments, metrics, tests)
    
    Examples
    --------
    >>> synthesizer = TETResultsSynthesizer()
    >>> synthesizer.load_all_results()
    >>> report_path = synthesizer.generate_report()
    >>> print(f"Report generated: {report_path}")
    """
    
    def __init__(
        self,
        results_dir: str = 'results/tet',
        output_path: str = 'docs/tet_comprehensive_results.md'
    ):
        self.results_dir = Path(results_dir)
        self.output_path = Path(output_path)
        
        # Data containers
        self.descriptive_data = None
        self.lme_results = None
        self.pca_results = None
        self.clustering_results = None
        
        logger.info(f"Initialized TETResultsSynthesizer")
        logger.info(f"  Results directory: {self.results_dir}")
        logger.info(f"  Output path: {self.output_path}")

    
    def _load_descriptive(self) -> Optional[Dict]:
        """
        Load descriptive statistics (time courses and session summaries).
        
        Returns
        -------
        dict or None
            Dictionary with 'timecourses' and 'summaries' keys, or None if not found
        """
        desc_dir = self.results_dir / 'descriptive'
        
        if not desc_dir.exists():
            logger.warning(f"Descriptive directory not found: {desc_dir}")
            return None
        
        try:
            # Load session summaries
            summaries_path = desc_dir / 'session_summaries.csv'
            if not summaries_path.exists():
                logger.warning(f"Session summaries not found: {summaries_path}")
                return None
            
            summaries = pd.read_csv(summaries_path)
            
            # Load time courses (one file per dimension)
            timecourses = {}
            for tc_file in desc_dir.glob('timecourse_*.csv'):
                dimension = tc_file.stem.replace('timecourse_', '')
                timecourses[dimension] = pd.read_csv(tc_file)
            
            logger.info(f"Loaded descriptive data: {len(timecourses)} time courses, {len(summaries)} session summaries")
            
            return {
                'timecourses': timecourses,
                'summaries': summaries
            }
            
        except Exception as e:
            logger.error(f"Error loading descriptive data: {e}")
            return None
    
    def _load_lme(self) -> Optional[pd.DataFrame]:
        """
        Load LME coefficients.
        
        Returns
        -------
        pd.DataFrame or None
            LME coefficients DataFrame, or None if not found
        """
        lme_dir = self.results_dir / 'lme'
        
        if not lme_dir.exists():
            logger.warning(f"LME directory not found: {lme_dir}")
            return None
        
        try:
            # Try to load combined coefficients file
            lme_path = lme_dir / 'lme_coefficients_all_dimensions.csv'
            if not lme_path.exists():
                logger.warning(f"LME coefficients not found: {lme_path}")
                return None
            
            lme_results = pd.read_csv(lme_path)
            
            # Validate required columns
            required_cols = ['dimension', 'effect', 'beta', 'ci_lower', 'ci_upper', 'p_fdr']
            missing_cols = set(required_cols) - set(lme_results.columns)
            if missing_cols:
                logger.error(f"LME results missing columns: {missing_cols}")
                return None
            
            logger.info(f"Loaded LME results: {len(lme_results)} coefficients")
            
            return lme_results
            
        except Exception as e:
            logger.error(f"Error loading LME results: {e}")
            return None
    
    def _load_pca(self) -> Optional[Dict]:
        """
        Load PCA results (loadings, variance explained, LME results).
        
        Returns
        -------
        dict or None
            Dictionary with 'loadings', 'variance', 'lme' keys, or None if not found
        """
        pca_dir = self.results_dir / 'pca'
        
        if not pca_dir.exists():
            logger.warning(f"PCA directory not found: {pca_dir}")
            return None
        
        try:
            results = {}
            
            # Load loadings
            loadings_path = pca_dir / 'pca_loadings.csv'
            if loadings_path.exists():
                results['loadings'] = pd.read_csv(loadings_path)
            else:
                logger.warning(f"PCA loadings not found: {loadings_path}")
            
            # Load variance explained
            variance_path = pca_dir / 'pca_variance_explained.csv'
            if variance_path.exists():
                results['variance'] = pd.read_csv(variance_path)
            else:
                logger.warning(f"PCA variance not found: {variance_path}")
            
            # Load LME results for PC scores (optional)
            lme_path = pca_dir / 'pca_lme_results.csv'
            if lme_path.exists():
                results['lme'] = pd.read_csv(lme_path)
            else:
                logger.info(f"PCA LME results not found (optional): {lme_path}")
            
            if not results:
                return None
            
            logger.info(f"Loaded PCA results: {list(results.keys())}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error loading PCA results: {e}")
            return None
    
    def _load_clustering(self) -> Optional[Dict]:
        """
        Load clustering results (assignments, metrics, statistical tests).
        
        Returns
        -------
        dict or None
            Dictionary with clustering data, or None if not found
        """
        clustering_dir = self.results_dir / 'clustering'
        
        if not clustering_dir.exists():
            logger.warning(f"Clustering directory not found: {clustering_dir}")
            return None
        
        try:
            results = {}
            
            # Load KMeans assignments
            kmeans_path = clustering_dir / 'clustering_kmeans_assignments.csv'
            if kmeans_path.exists():
                results['kmeans_assignments'] = pd.read_csv(kmeans_path)
            else:
                logger.warning(f"KMeans assignments not found: {kmeans_path}")
            
            # Load clustering metrics
            metrics_path = clustering_dir / 'clustering_metrics.csv'
            if metrics_path.exists():
                results['metrics'] = pd.read_csv(metrics_path)
            else:
                logger.warning(f"Clustering metrics not found: {metrics_path}")
            
            # Load state occupancy
            occupancy_path = clustering_dir / 'state_occupancy.csv'
            if occupancy_path.exists():
                results['occupancy'] = pd.read_csv(occupancy_path)
            else:
                logger.warning(f"State occupancy not found: {occupancy_path}")
            
            # Load statistical tests (optional)
            tests_path = clustering_dir / 'clustering_dose_tests_classical.csv'
            if tests_path.exists():
                results['dose_tests'] = pd.read_csv(tests_path)
            
            if not results:
                return None
            
            logger.info(f"Loaded clustering results: {list(results.keys())}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error loading clustering results: {e}")
            return None
    
    def load_all_results(self):
        """
        Load all analysis results.
        
        This method loads data from all TET analysis components. Missing files
        are handled gracefully by logging warnings and setting the corresponding
        attribute to None.
        """
        logger.info("Loading all TET analysis results...")
        
        self.descriptive_data = self._load_descriptive()
        self.lme_results = self._load_lme()
        self.pca_results = self._load_pca()
        self.clustering_results = self._load_clustering()
        
        # Count loaded components
        loaded = sum([
            self.descriptive_data is not None,
            self.lme_results is not None,
            self.pca_results is not None,
            self.clustering_results is not None
        ])
        
        logger.info(f"Loaded {loaded}/4 analysis components")

    
    def _generate_header(self) -> str:
        """
        Generate document header with title and metadata.
        
        Returns
        -------
        str
            Formatted markdown header
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        header = f"""# TET Analysis: Comprehensive Results

**Generated:** {timestamp}

**Analysis Pipeline:** TET (Temporal Experience Tracking) Analysis

**Data:** 19 subjects, 76 sessions (38 RS, 38 DMT: 19 Low dose, 19 High dose)

---
"""
        return header
    
    def generate_executive_summary(self) -> str:
        """
        Generate executive summary with top 3-5 findings.
        
        Returns
        -------
        str
            Formatted markdown section
        """
        logger.info("Generating executive summary...")
        
        # Rank findings across all methods
        top_findings = rank_findings(
            self.lme_results,
            None,  # peak_auc_results removed
            self.clustering_results,
            top_n=5
        )
        
        if not top_findings:
            return """## Executive Summary

**Note:** Insufficient data available to generate executive summary. Please ensure
all analysis components have been completed.
"""
        
        # Format summary
        summary = """## Executive Summary

This analysis identified the following key findings from TET data across 19 subjects
experiencing DMT at two doses (20mg Low, 40mg High) compared to Resting State:

"""
        
        for i, finding in enumerate(top_findings, 1):
            finding_type = finding['type']
            
            if 'dimension' in finding:
                dimension_name = StatisticalFormatter.format_dimension_name(finding['dimension'])
                
                if finding_type == 'LME_State':
                    effect_str = StatisticalFormatter.format_lme_result(
                        finding['effect_size'],
                        finding['ci_lower'],
                        finding['ci_upper'],
                        finding['p_fdr']
                    )
                    summary += f"{i}. **Strong DMT State Effect on {dimension_name}** ({effect_str}): "
                    summary += f"{dimension_name} showed a strong increase during DMT compared to RS.\n\n"
                
                elif 'Dose' in finding_type:
                    effect_str = StatisticalFormatter.format_effect_size(
                        finding['effect_size'],
                        finding['ci_lower'],
                        finding['ci_upper'],
                        finding['p_fdr']
                    )
                    direction = "increased" if finding['effect_size'] > 0 else "decreased"
                    summary += f"{i}. **Dose-Dependent Effect on {dimension_name}** ({effect_str}): "
                    summary += f"{dimension_name} {direction} at High dose compared to Low dose.\n\n"
            
            elif 'cluster' in finding:
                metric_name = finding.get('metric', 'occupancy')
                cluster_id = finding['cluster']
                p_str = StatisticalFormatter.format_p_value(finding['p_fdr'])
                summary += f"{i}. **Cluster {cluster_id} Dose Effect on {metric_name}** ({p_str}): "
                summary += f"Significant dose-dependent changes in cluster {cluster_id} {metric_name}.\n\n"
        
        logger.info(f"Generated executive summary with {len(top_findings)} findings")
        
        return summary
    
    def generate_descriptive_section(self) -> str:
        """
        Generate descriptive statistics section.
        
        Returns
        -------
        str
            Formatted markdown section
        """
        logger.info("Generating descriptive statistics section...")
        
        if self.descriptive_data is None:
            return """## 1. Descriptive Statistics

**Note:** Descriptive statistics data not available. Please run descriptive analysis first.
"""
        
        # Extract descriptive stats
        stats_dict = extract_descriptive_stats(self.descriptive_data)
        
        if not stats_dict:
            return """## 1. Descriptive Statistics

**Note:** Unable to extract descriptive statistics from available data.
"""
        
        section = """## 1. Descriptive Statistics

### 1.1 Temporal Dynamics Overview

Across all dimensions, DMT sessions showed characteristic temporal profiles with:
- Rapid onset (mean time to peak varies by dimension)
- Sustained plateau during peak effects
- Gradual offset returning toward baseline

### 1.2 Dimension-Specific Patterns

"""
        
        # Sort dimensions by peak timing for organized presentation
        sorted_dims = sorted(stats_dict.items(), key=lambda x: x[1]['peak_time_mean'])
        
        for dimension, stats in sorted_dims:
            dim_name = StatisticalFormatter.format_dimension_name(dimension)
            
            section += f"#### {dim_name}\n\n"
            
            # Peak timing
            peak_time_str = StatisticalFormatter.format_mean_sem(
                stats['peak_time_mean'],
                stats['peak_time_std']
            )
            peak_range = stats['peak_time_range']
            section += f"- **Peak Timing**: {peak_time_str} minutes (range: {peak_range[0]:.1f}-{peak_range[1]:.1f} min)\n"
            
            # Peak intensity by dose
            if not np.isnan(stats['peak_high']) and not np.isnan(stats['peak_low']):
                section += f"- **Peak Intensity**: High dose = {stats['peak_high']:.1f}, Low dose = {stats['peak_low']:.1f} (raw scale)\n"
            
            # Baseline
            if not np.isnan(stats['baseline_rs']):
                section += f"- **Baseline**: RS = {stats['baseline_rs']:.1f}\n"
            
            # Variability
            if not np.isnan(stats['cv']):
                variability = "high" if stats['cv'] > 0.5 else "moderate" if stats['cv'] > 0.3 else "low"
                section += f"- **Variability**: CV = {stats['cv']:.2f} ({variability} inter-subject consistency)\n"
            
            section += "\n"
        
        logger.info(f"Generated descriptive section for {len(stats_dict)} dimensions")
        
        return section
    
    def generate_lme_section(self) -> str:
        """
        Generate LME results section.
        
        Returns
        -------
        str
            Formatted markdown section
        """
        logger.info("Generating LME results section...")
        
        if self.lme_results is None:
            return """## 2. Linear Mixed Effects Results

**Note:** LME results not available. Please run LME analysis first.
"""
        
        # Extract LME effects
        effects_dict = extract_lme_effects(self.lme_results)
        
        if not effects_dict:
            return """## 2. Linear Mixed Effects Results

**Note:** Unable to extract LME effects from available data.
"""
        
        section = """## 2. Linear Mixed Effects Results

"""
        
        # Process each effect type
        effect_titles = {
            'State': '### 2.1 State Effects (DMT vs RS)',
            'Dose': '### 2.2 Dose Effects (High vs Low)',
            'Interaction': '### 2.3 State:Dose Interaction Effects',
            'Time': '### 2.4 Temporal Effects'
        }
        
        for effect_name, title in effect_titles.items():
            if effect_name not in effects_dict:
                continue
            
            effect_data = effects_dict[effect_name]
            
            # Count significant effects
            n_sig = len(effect_data['strong']) + len(effect_data['moderate']) + len(effect_data['weak'])
            
            section += f"{title}\n\n"
            
            if n_sig == 0:
                section += f"No significant {effect_name} effects observed (all p_fdr ≥ 0.05).\n\n"
                continue
            
            section += f"Significant {effect_name} effects (p_fdr < 0.05) were observed for {n_sig} dimensions:\n\n"
            
            # Strong effects
            if not effect_data['strong'].empty:
                section += "**Strong Effects (|β| > 1.5)**:\n\n"
                for _, row in effect_data['strong'].iterrows():
                    dim_name = StatisticalFormatter.format_dimension_name(row['dimension'])
                    effect_str = StatisticalFormatter.format_lme_result(
                        row['beta'],
                        row['ci_lower'],
                        row['ci_upper'],
                        row['p_fdr']
                    )
                    section += f"- **{dim_name}**: {effect_str}\n"
                section += "\n"
            
            # Moderate effects
            if not effect_data['moderate'].empty:
                section += "**Moderate Effects (0.8 < |β| < 1.5)**:\n\n"
                for _, row in effect_data['moderate'].iterrows():
                    dim_name = StatisticalFormatter.format_dimension_name(row['dimension'])
                    effect_str = StatisticalFormatter.format_lme_result(
                        row['beta'],
                        row['ci_lower'],
                        row['ci_upper'],
                        row['p_fdr']
                    )
                    section += f"- **{dim_name}**: {effect_str}\n"
                section += "\n"
            
            # Weak effects
            if not effect_data['weak'].empty:
                section += "**Weak Effects (|β| < 0.8)**:\n\n"
                for _, row in effect_data['weak'].iterrows():
                    dim_name = StatisticalFormatter.format_dimension_name(row['dimension'])
                    effect_str = StatisticalFormatter.format_lme_result(
                        row['beta'],
                        row['ci_lower'],
                        row['ci_upper'],
                        row['p_fdr']
                    )
                    section += f"- **{dim_name}**: {effect_str}\n"
                section += "\n"
            
            # Non-significant (show first 3)
            if not effect_data['non_sig'].empty:
                section += "**Non-Significant** (showing top 3 by p-value):\n\n"
                for _, row in effect_data['non_sig'].head(3).iterrows():
                    dim_name = StatisticalFormatter.format_dimension_name(row['dimension'])
                    p_str = StatisticalFormatter.format_p_value(row['p_fdr'])
                    section += f"- {dim_name}: β = {row['beta']:.2f}, {p_str}\n"
                section += "\n"
        
        # Add figure reference
        section += "[See Figure: ../results/tet/figures/lme_coefficients_forest.png]\n\n"
        
        logger.info(f"Generated LME section for {len(effects_dict)} effect types")
        
        return section
    

    def generate_pca_section(self) -> str:
        """Generate PCA section."""
        logger.info("Generating PCA section...")
        
        if self.pca_results is None:
            return """## 4. Dimensionality Reduction

**Note:** PCA results not available. Please run PCA analysis first.
"""
        
        interpretations = interpret_pca_components(self.pca_results)
        
        if not interpretations:
            return """## 4. Dimensionality Reduction

**Note:** Unable to interpret PCA components.
"""
        
        section = """## 4. Dimensionality Reduction

### 4.1 Principal Components Interpretation

"""
        
        # Report components retained
        n_components = len(interpretations)
        total_var = sum(interp['variance_explained'] for interp in interpretations.values() if not np.isnan(interp['variance_explained']))
        section += f"PCA identified {n_components} principal components explaining {total_var:.1%} of total variance.\n\n"
        
        # Describe each component
        for comp_name, interp in interpretations.items():
            section += f"#### {comp_name}: {interp['interpretation']}\n\n"
            
            if interp['top_positive']:
                section += "**Top Positive Loadings**:\n\n"
                for dim, loading in interp['top_positive'][:5]:
                    dim_name = StatisticalFormatter.format_dimension_name(dim)
                    section += f"- {dim_name}: {loading:.2f}\n"
                section += "\n"
            
            if interp['top_negative']:
                section += "**Top Negative Loadings**:\n\n"
                for dim, loading in interp['top_negative'][:5]:
                    dim_name = StatisticalFormatter.format_dimension_name(dim)
                    section += f"- {dim_name}: {loading:.2f}\n"
                section += "\n"
            
            # LME effects for PC scores
            if interp['lme_effects'] is not None and not interp['lme_effects'].empty:
                section += "**Temporal Dynamics**:\n\n"
                for _, row in interp['lme_effects'].iterrows():
                    if 'p_fdr' in row and row['p_fdr'] < 0.05:
                        effect_str = StatisticalFormatter.format_lme_result(
                            row['beta'], row['ci_lower'], row['ci_upper'], row['p_fdr']
                        )
                        section += f"- {row['effect']}: {effect_str}\n"
                    elif 'p_value' in row and row['p_value'] < 0.05:
                        effect_str = StatisticalFormatter.format_lme_result(
                            row['beta'], row['ci_lower'], row['ci_upper'], row['p_value']
                        )
                        section += f"- {row['effect']}: {effect_str}\n"
                section += "\n"
        
        section += "[See Figures: ../results/tet/figures/pca_scree_plot.png, ../results/tet/figures/pca_loadings_heatmap.png]\n\n"
        
        logger.info(f"Generated PCA section for {n_components} components")
        return section
    
    def generate_clustering_section(self) -> str:
        """Generate clustering section."""
        logger.info("Generating clustering section...")
        
        if self.clustering_results is None:
            return """## 5. Clustering Analysis

**Note:** Clustering results not available. Please run clustering analysis first.
"""
        
        cluster_info = extract_clustering_results(self.clustering_results)
        
        if not cluster_info['clusters']:
            return """## 5. Clustering Analysis

**Note:** Unable to extract clustering information.
"""
        
        section = """## 5. Clustering Analysis

### 5.1 Optimal Cluster Solution

"""
        
        if cluster_info['optimal_k']:
            section += f"**KMeans**: k = {cluster_info['optimal_k']} selected based on silhouette score ({cluster_info['silhouette_score']:.2f})\n\n"
        
        section += "### 5.2 Cluster Characterization\n\n"
        
        for cluster_id, cluster_data in cluster_info['clusters'].items():
            section += f"#### Cluster {cluster_id}\n\n"
            
            if cluster_data['elevated']:
                section += "**Elevated Dimensions** (z > 0.5):\n\n"
                for dim, z_score in cluster_data['elevated'][:5]:
                    dim_name = StatisticalFormatter.format_dimension_name(dim)
                    section += f"- {dim_name}: z = {z_score:.2f}\n"
                section += "\n"
            
            if cluster_data['suppressed']:
                section += "**Suppressed Dimensions** (z < -0.5):\n\n"
                for dim, z_score in cluster_data['suppressed'][:5]:
                    dim_name = StatisticalFormatter.format_dimension_name(dim)
                    section += f"- {dim_name}: z = {z_score:.2f}\n"
                section += "\n"
        
        if cluster_info['dose_effects']:
            section += "### 5.3 Dose Effects on Cluster Occupancy\n\n"
            for effect in cluster_info['dose_effects'][:5]:
                p_str = StatisticalFormatter.format_p_value(effect['p_fdr'])
                section += f"- Cluster {effect.get('cluster_state', 'X')} {effect.get('metric', 'occupancy')}: {p_str}\n"
            section += "\n"
        
        section += "[See Figures: ../results/tet/figures/clustering_kmeans_centroids_k2.png, ../results/tet/figures/clustering_kmeans_prob_timecourses_dmt_only.png]\n\n"
        
        logger.info(f"Generated clustering section for {len(cluster_info['clusters'])} clusters")
        return section
    
    def generate_integration_section(self) -> str:
        """Generate cross-analysis integration section."""
        logger.info("Generating integration section...")
        
        cross_results = compute_cross_method_rankings(
            self.lme_results,
            None,  # peak_auc_results removed
            self.pca_results,
            self.clustering_results
        )
        
        if not cross_results['rankings']:
            return """## 6. Cross-Analysis Integration

**Note:** Insufficient data for cross-method comparison.
"""
        
        section = """## 6. Cross-Analysis Integration

### 6.1 Convergent Findings

"""
        
        if cross_results['convergent_dimensions']:
            section += "The following dimensions showed consistent effects across multiple methods:\n\n"
            for dim in cross_results['convergent_dimensions']:
                dim_name = StatisticalFormatter.format_dimension_name(dim)
                section += f"- **{dim_name}**: Significant across {len([r for r in cross_results['rankings'].values() if dim in r[:5]])} methods\n"
            section += "\n"
        
        section += "### 6.2 Method Correlations\n\n"
        
        if cross_results['correlations']:
            for comparison, stats in cross_results['correlations'].items():
                section += f"- **{comparison}**: ρ = {stats['correlation']:.2f}, p = {stats['p_value']:.3f} (n = {stats['n_dimensions']} dimensions)\n"
            section += "\n"
        
        logger.info("Generated integration section")
        return section
    
    def generate_methodological_notes(self) -> str:
        """Generate methodological notes section."""
        logger.info("Generating methodological notes...")
        
        metadata = extract_methodological_metadata(
            self.descriptive_data,
            self.lme_results,
            self.pca_results,
            self.clustering_results
        )
        
        section = """## 7. Methodological Notes

### 7.1 Data Quality

"""
        
        for key, value in metadata['data_quality'].items():
            section += f"- **{key.replace('_', ' ').title()}**: {value}\n"
        section += "\n"
        
        section += "### 7.2 Preprocessing\n\n"
        for key, value in metadata['preprocessing'].items():
            section += f"- **{key.replace('_', ' ').title()}**: {value}\n"
        section += "\n"
        
        section += "### 7.3 Model Specifications\n\n"
        for model_name, spec in metadata['models'].items():
            section += f"- **{model_name}**: {spec}\n"
        section += "\n"
        
        section += "### 7.4 Analytical Decisions\n\n"
        for key, value in metadata['decisions'].items():
            section += f"- **{key.replace('_', ' ').title()}**: {value}\n"
        section += "\n"
        
        logger.info("Generated methodological notes")
        return section
    
    def generate_further_investigation(self) -> str:
        """Generate further investigation section."""
        logger.info("Generating further investigation section...")
        
        ambiguities = detect_ambiguous_findings(
            self.lme_results,
            None,  # peak_auc_results removed
            self.pca_results,
            self.clustering_results
        )
        
        section = """## 8. Further Investigation

### 8.1 Unresolved Questions and Ambiguous Findings

"""
        
        if not ambiguities:
            section += "No major ambiguities detected. All significant findings are consistent across methods.\n\n"
        else:
            for i, ambiguity in enumerate(ambiguities, 1):
                section += f"{i}. **{ambiguity['type']}**: {ambiguity['description']}\n"
                section += f"   - *Suggested analysis*: {ambiguity['suggestion']}\n\n"
        
        section += """### 8.2 Suggested Follow-up Analyses

**High Priority**:

1. **Generalized Additive Models (GAMs)**: Capture non-linear temporal dynamics
2. **Individual Difference Analysis**: Characterize dose response variability
3. **GLHMM State Modeling**: Model temporal state transitions

**Medium Priority**:

4. **Multivariate Time Series Analysis**: Understand dimension co-variation
5. **Sensitivity Analyses**: Test robustness to analytical decisions

"""
        
        logger.info(f"Generated further investigation section with {len(ambiguities)} ambiguities")
        return section
    
    def generate_report(self) -> Path:
        """
        Generate comprehensive results report.
        
        Assembles all sections and writes the complete markdown document.
        
        Returns
        -------
        Path
            Path to generated report file
        """
        logger.info("Generating comprehensive results report...")
        
        sections = []
        errors = []
        
        # Header
        try:
            sections.append(self._generate_header())
        except Exception as e:
            logger.error(f"Error generating header: {e}")
            errors.append(('Header', str(e)))
        
        # Executive Summary
        try:
            sections.append(self.generate_executive_summary())
        except Exception as e:
            logger.error(f"Error generating executive summary: {e}")
            errors.append(('Executive Summary', str(e)))
            sections.append("## Executive Summary\n\n**Error generating summary**\n")
        
        # Descriptive Statistics
        try:
            sections.append(self.generate_descriptive_section())
        except Exception as e:
            logger.error(f"Error generating descriptive section: {e}")
            errors.append(('Descriptive Statistics', str(e)))
            sections.append("## 1. Descriptive Statistics\n\n**Error generating section**\n")
        
        # LME Results
        try:
            sections.append(self.generate_lme_section())
        except Exception as e:
            logger.error(f"Error generating LME section: {e}")
            errors.append(('LME Results', str(e)))
            sections.append("## 2. Linear Mixed Effects Results\n\n**Error generating section**\n")
        
        # PCA / Dimensionality Reduction
        try:
            sections.append(self.generate_pca_section())
        except Exception as e:
            logger.error(f"Error generating PCA section: {e}")
            errors.append(('Dimensionality Reduction', str(e)))
            sections.append("## 3. Dimensionality Reduction\n\n**Error generating section**\n")
        
        # Clustering Analysis
        try:
            sections.append(self.generate_clustering_section())
        except Exception as e:
            logger.error(f"Error generating clustering section: {e}")
            errors.append(('Clustering Analysis', str(e)))
            sections.append("## 4. Clustering Analysis\n\n**Error generating section**\n")
        
        # Cross-Analysis Integration
        try:
            sections.append(self.generate_integration_section())
        except Exception as e:
            logger.error(f"Error generating integration section: {e}")
            errors.append(('Cross-Analysis Integration', str(e)))
            sections.append("## 5. Cross-Analysis Integration\n\n**Error generating section**\n")
        
        # Methodological Notes
        try:
            sections.append(self.generate_methodological_notes())
        except Exception as e:
            logger.error(f"Error generating methodological notes: {e}")
            errors.append(('Methodological Notes', str(e)))
            sections.append("## 7. Methodological Notes\n\n**Error generating section**\n")
        
        # Further Investigation
        try:
            sections.append(self.generate_further_investigation())
        except Exception as e:
            logger.error(f"Error generating further investigation section: {e}")
            errors.append(('Further Investigation', str(e)))
            sections.append("## 8. Further Investigation\n\n**Error generating section**\n")
        
        # Add error report if any failures
        if errors:
            error_section = "\n## Report Generation Errors\n\n"
            for section, error in errors:
                error_section += f"- **{section}**: {error}\n"
            sections.append(error_section)
        
        # Assemble report
        report_text = '\n\n'.join(sections)
        
        # Write to file
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.write_text(report_text, encoding='utf-8')
        
        file_size = self.output_path.stat().st_size
        logger.info(f"Report generated: {self.output_path} ({file_size} bytes)")
        logger.info(f"Sections generated: {len(sections) - len(errors)}/{len(sections)}")
        
        if errors:
            logger.warning(f"Errors encountered: {len(errors)}")
        
        return self.output_path
