"""
TET Results Analyzer

This module provides utilities for analyzing and ranking TET results.
"""

import logging
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np
from scipy.stats import spearmanr

logger = logging.getLogger(__name__)


def rank_findings(
    lme_results: Optional[pd.DataFrame],
    peak_auc_results: Optional[Dict],
    clustering_results: Optional[Dict],
    top_n: int = 5
) -> List[Dict]:
    """
    Rank findings by importance across all analysis methods.
    
    Computes importance scores for significant effects from LME, peak/AUC, and
    clustering analyses, then returns the top N findings.
    
    Parameters
    ----------
    lme_results : pd.DataFrame or None
        LME coefficients with columns: dimension, effect, beta, p_fdr
    peak_auc_results : dict or None
        Peak/AUC results with 'peak' and 'auc' DataFrames
    clustering_results : dict or None
        Clustering results with 'dose_tests' DataFrame
    top_n : int, optional
        Number of top findings to return, by default 5
    
    Returns
    -------
    list of dict
        Top N findings, each with keys: type, dimension/cluster, effect_size, p_fdr, score
    
    Notes
    -----
    Importance score = abs(effect_size) * (1 - p_fdr) * weight
    Weights: LME State effects (2.0), Peak/AUC (1.5), Clustering (1.2)
    """
    findings = []
    
    # Extract LME State effects (highest weight)
    if lme_results is not None:
        lme_state = lme_results[
            (lme_results['effect'].str.contains('state', case=False)) &
            (lme_results['p_fdr'] < 0.01)
        ].copy()
        
        for _, row in lme_state.iterrows():
            score = abs(row['beta']) * (1 - row['p_fdr']) * 2.0
            findings.append({
                'type': 'LME_State',
                'dimension': row['dimension'],
                'effect_size': row['beta'],
                'p_fdr': row['p_fdr'],
                'ci_lower': row.get('ci_lower', np.nan),
                'ci_upper': row.get('ci_upper', np.nan),
                'score': score,
                'description': f"State effect on {row['dimension']}"
            })
    
    # Extract Peak/AUC dose effects
    if peak_auc_results is not None:
        for metric_type, df in peak_auc_results.items():
            if df is None:
                continue
            
            sig_effects = df[df['p_fdr'] < 0.05].copy()
            
            for _, row in sig_effects.iterrows():
                score = abs(row['effect_r']) * (1 - row['p_fdr']) * 1.5
                findings.append({
                    'type': f'{metric_type.capitalize()}_Dose',
                    'dimension': row['dimension'],
                    'effect_size': row['effect_r'],
                    'p_fdr': row['p_fdr'],
                    'ci_lower': row.get('ci_lower', np.nan),
                    'ci_upper': row.get('ci_upper', np.nan),
                    'score': score,
                    'description': f"Dose effect on {metric_type} for {row['dimension']}"
                })
    
    # Extract Clustering dose effects
    if clustering_results is not None and 'dose_tests' in clustering_results:
        dose_tests = clustering_results['dose_tests']
        sig_effects = dose_tests[dose_tests['p_fdr'] < 0.05].copy()
        
        for _, row in sig_effects.iterrows():
            score = abs(row.get('t_statistic', 0)) * (1 - row['p_fdr']) * 1.2
            findings.append({
                'type': 'Cluster_Dose',
                'cluster': row.get('cluster_state', 'unknown'),
                'metric': row.get('metric', 'unknown'),
                'effect_size': row.get('mean_diff', np.nan),
                'p_fdr': row['p_fdr'],
                'ci_lower': row.get('ci_lower', np.nan),
                'ci_upper': row.get('ci_upper', np.nan),
                'score': score,
                'description': f"Dose effect on {row.get('metric', 'metric')} for cluster {row.get('cluster_state', 'X')}"
            })
    
    # Sort by score and return top N
    findings.sort(key=lambda x: x['score'], reverse=True)
    
    logger.info(f"Ranked {len(findings)} findings, returning top {top_n}")
    
    return findings[:top_n]



def extract_descriptive_stats(
    descriptive_data: Optional[Dict],
    dimensions: Optional[List[str]] = None
) -> Dict[str, Dict]:
    """
    Extract descriptive statistics for each dimension.
    
    Parameters
    ----------
    descriptive_data : dict or None
        Descriptive data with 'timecourses' and 'summaries' keys
    dimensions : list of str, optional
        List of dimensions to extract stats for. If None, extracts for all dimensions.
    
    Returns
    -------
    dict
        Dictionary mapping dimension names to their statistics
    """
    if descriptive_data is None or 'summaries' not in descriptive_data:
        logger.warning("No descriptive data available")
        return {}
    
    summaries = descriptive_data['summaries']
    
    # Get unique dimensions
    if dimensions is None:
        dimensions = summaries['dimension'].unique()
    
    stats_by_dimension = {}
    
    for dimension in dimensions:
        dim_data = summaries[summaries['dimension'] == dimension].copy()
        
        if len(dim_data) == 0:
            continue
        
        # Extract statistics
        stats = {
            'peak_time_mean': dim_data['time_to_peak_min'].mean(),
            'peak_time_std': dim_data['time_to_peak_min'].std(),
            'peak_time_range': (
                dim_data['time_to_peak_min'].min(),
                dim_data['time_to_peak_min'].max()
            ),
            'peak_high': dim_data[dim_data['dose'] == 'High']['peak_value'].mean(),
            'peak_low': dim_data[dim_data['dose'] == 'Low']['peak_value'].mean(),
            'cv': dim_data['peak_value'].std() / dim_data['peak_value'].mean() if dim_data['peak_value'].mean() != 0 else np.nan
        }
        
        # Get baseline RS values if available
        rs_data = dim_data[dim_data['state'] == 'RS']
        if len(rs_data) > 0:
            stats['baseline_rs'] = rs_data['peak_value'].mean()
        else:
            stats['baseline_rs'] = np.nan
        
        stats_by_dimension[dimension] = stats
    
    logger.info(f"Extracted descriptive stats for {len(stats_by_dimension)} dimensions")
    
    return stats_by_dimension


def extract_lme_effects(
    lme_results: Optional[pd.DataFrame],
    significance_threshold: float = 0.05
) -> Dict[str, Dict]:
    """
    Extract and organize LME effects by type and magnitude.
    
    Parameters
    ----------
    lme_results : pd.DataFrame or None
        LME coefficients with columns: dimension, effect, beta, p_fdr
    significance_threshold : float, optional
        P-value threshold for significance, by default 0.05
    
    Returns
    -------
    dict
        Dictionary with keys for each effect type (State, Dose, Interaction, Time)
    """
    if lme_results is None:
        logger.warning("No LME results available")
        return {}
    
    effects_by_type = {
        'State': {'strong': [], 'moderate': [], 'weak': [], 'non_significant': []},
        'Dose': {'strong': [], 'moderate': [], 'weak': [], 'non_significant': []},
        'Interaction': {'strong': [], 'moderate': [], 'weak': [], 'non_significant': []},
        'Time': {'strong': [], 'moderate': [], 'weak': [], 'non_significant': []}
    }
    
    # Categorize effects
    for _, row in lme_results.iterrows():
        effect_name = row['effect']
        dimension = row['dimension']
        beta = row['beta']
        p_fdr = row['p_fdr']
        
        # Determine effect type
        effect_type = None
        if 'state' in effect_name.lower() and ':' not in effect_name.lower():
            effect_type = 'State'
        elif 'dose' in effect_name.lower() and ':' not in effect_name.lower():
            effect_type = 'Dose'
        elif ':' in effect_name.lower() or 'interaction' in effect_name.lower():
            effect_type = 'Interaction'
        elif 'time' in effect_name.lower():
            effect_type = 'Time'
        
        if effect_type is None:
            continue
        
        # Create effect entry
        effect_entry = {
            'dimension': dimension,
            'beta': beta,
            'ci_lower': row.get('ci_lower', np.nan),
            'ci_upper': row.get('ci_upper', np.nan),
            'p_fdr': p_fdr,
            'effect_name': effect_name
        }
        
        # Categorize by significance and magnitude
        if p_fdr >= significance_threshold:
            effects_by_type[effect_type]['non_significant'].append(effect_entry)
        else:
            abs_beta = abs(beta)
            if abs_beta > 1.5:
                effects_by_type[effect_type]['strong'].append(effect_entry)
            elif abs_beta > 0.8:
                effects_by_type[effect_type]['moderate'].append(effect_entry)
            else:
                effects_by_type[effect_type]['weak'].append(effect_entry)
    
    # Sort each category by |beta|
    for effect_type in effects_by_type:
        for magnitude in ['strong', 'moderate', 'weak']:
            effects_by_type[effect_type][magnitude].sort(
                key=lambda x: abs(x['beta']),
                reverse=True
            )
    
    logger.info(f"Extracted LME effects for {len(lme_results)} coefficients")
    
    return effects_by_type


def extract_peak_auc_effects(
    peak_auc_results: Optional[Dict],
    significance_threshold: float = 0.05
) -> Dict[str, List]:
    """
    Extract and organize peak/AUC effects by metric type.
    
    Parameters
    ----------
    peak_auc_results : dict or None
        Peak/AUC results with 'peak' and 'auc' DataFrames
    significance_threshold : float, optional
        P-value threshold for significance, by default 0.05
    
    Returns
    -------
    dict
        Dictionary with keys for each metric type
    """
    if peak_auc_results is None:
        logger.warning("No peak/AUC results available")
        return {}
    
    effects_by_metric = {}
    
    for metric_type, df in peak_auc_results.items():
        if df is None:
            continue
        
        # Filter significant effects
        sig_effects = df[df['p_fdr'] < significance_threshold].copy()
        
        # Separate increased vs decreased
        increased = []
        decreased = []
        
        for _, row in sig_effects.iterrows():
            effect_entry = {
                'dimension': row['dimension'],
                'effect_r': row['effect_r'],
                'ci_lower': row.get('ci_lower', np.nan),
                'ci_upper': row.get('ci_upper', np.nan),
                'p_fdr': row['p_fdr']
            }
            
            if row['effect_r'] > 0:
                increased.append(effect_entry)
            else:
                decreased.append(effect_entry)
        
        # Sort by |effect_r|
        increased.sort(key=lambda x: abs(x['effect_r']), reverse=True)
        decreased.sort(key=lambda x: abs(x['effect_r']), reverse=True)
        
        effects_by_metric[metric_type] = {
            'increased': increased,
            'decreased': decreased,
            'all': sig_effects.sort_values('effect_r', key=abs, ascending=False).to_dict('records')
        }
    
    logger.info(f"Extracted peak/AUC effects for {len(effects_by_metric)} metric types")
    
    return effects_by_metric



def extract_descriptive_stats(
    descriptive_data: Optional[Dict],
    dimensions: Optional[List[str]] = None
) -> Dict[str, Dict]:
    """
    Extract descriptive statistics for each dimension.
    
    Parameters
    ----------
    descriptive_data : dict or None
        Dictionary with 'timecourses' and 'summaries' keys
    dimensions : list of str, optional
        List of dimensions to extract stats for. If None, extracts for all dimensions.
    
    Returns
    -------
    dict
        Dictionary mapping dimension names to their statistics:
        - peak_time_mean: Mean time to peak (minutes)
        - peak_time_std: Std of time to peak
        - peak_time_range: (min, max) time to peak
        - peak_high: Mean peak value at high dose
        - peak_low: Mean peak value at low dose
        - baseline_rs: Mean RS baseline value
        - cv: Coefficient of variation
    """
    if descriptive_data is None:
        logger.warning("No descriptive data available")
        return {}
    
    summaries = descriptive_data.get('summaries')
    timecourses = descriptive_data.get('timecourses', {})
    
    if summaries is None or summaries.empty:
        logger.warning("No session summaries available")
        return {}
    
    # Get dimensions to process
    if dimensions is None:
        dimensions = summaries['dimension'].unique()
    
    stats_dict = {}
    
    for dimension in dimensions:
        try:
            # Filter summaries for this dimension
            dim_summaries = summaries[summaries['dimension'] == dimension].copy()
            
            if dim_summaries.empty:
                logger.warning(f"No summaries found for dimension: {dimension}")
                continue
            
            # Extract peak timing statistics
            peak_times = dim_summaries['time_to_peak_min']
            peak_time_mean = peak_times.mean()
            peak_time_std = peak_times.std()
            peak_time_range = (peak_times.min(), peak_times.max())
            
            # Extract peak intensity by dose (DMT sessions only)
            dmt_summaries = dim_summaries[dim_summaries['state'] == 'DMT']
            
            if not dmt_summaries.empty:
                high_dose = dmt_summaries[dmt_summaries['dose'] == 'High']
                low_dose = dmt_summaries[dmt_summaries['dose'] == 'Low']
                
                peak_high = high_dose['peak_value'].mean() if not high_dose.empty else np.nan
                peak_low = low_dose['peak_value'].mean() if not low_dose.empty else np.nan
            else:
                peak_high = np.nan
                peak_low = np.nan
            
            # Extract baseline RS values from time courses
            baseline_rs = np.nan
            if dimension in timecourses:
                tc = timecourses[dimension]
                rs_data = tc[tc['state'] == 'RS']
                if not rs_data.empty and dimension in rs_data.columns:
                    baseline_rs = rs_data[dimension].mean()
            
            # Compute coefficient of variation
            all_peaks = dim_summaries['peak_value']
            cv = all_peaks.std() / all_peaks.mean() if all_peaks.mean() != 0 else np.nan
            
            stats_dict[dimension] = {
                'peak_time_mean': peak_time_mean,
                'peak_time_std': peak_time_std,
                'peak_time_range': peak_time_range,
                'peak_high': peak_high,
                'peak_low': peak_low,
                'baseline_rs': baseline_rs,
                'cv': cv
            }
            
        except Exception as e:
            logger.error(f"Error extracting stats for {dimension}: {e}")
            continue
    
    logger.info(f"Extracted descriptive stats for {len(stats_dict)} dimensions")
    
    return stats_dict



def extract_lme_effects(
    lme_results: Optional[pd.DataFrame]
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Extract and organize LME effects by type and magnitude.
    
    Parameters
    ----------
    lme_results : pd.DataFrame or None
        LME coefficients with columns: dimension, effect, beta, ci_lower, ci_upper, p_fdr
    
    Returns
    -------
    dict
        Dictionary organized by effect type (State, Dose, Interaction, Time),
        each containing DataFrames categorized by magnitude (strong, moderate, weak, non_sig)
    """
    if lme_results is None or lme_results.empty:
        logger.warning("No LME results available")
        return {}
    
    effects_dict = {}
    
    # Define effect types to extract
    effect_types = {
        'State': ['state', 'C(state'],
        'Dose': ['dose', 'C(dose'],
        'Interaction': ['state.*dose', 'dose.*state', ':'],
        'Time': ['time_c', 'Time']
    }
    
    for effect_name, patterns in effect_types.items():
        # Filter for this effect type
        mask = lme_results['effect'].str.contains('|'.join(patterns), case=False, regex=True)
        effect_data = lme_results[mask].copy()
        
        if effect_data.empty:
            continue
        
        # Separate significant and non-significant
        sig_data = effect_data[effect_data['p_fdr'] < 0.05].copy()
        non_sig_data = effect_data[effect_data['p_fdr'] >= 0.05].copy()
        
        # Categorize significant effects by magnitude
        sig_data['abs_beta'] = sig_data['beta'].abs()
        
        strong = sig_data[sig_data['abs_beta'] > 1.5].sort_values('abs_beta', ascending=False)
        moderate = sig_data[(sig_data['abs_beta'] >= 0.8) & (sig_data['abs_beta'] <= 1.5)].sort_values('abs_beta', ascending=False)
        weak = sig_data[sig_data['abs_beta'] < 0.8].sort_values('abs_beta', ascending=False)
        
        effects_dict[effect_name] = {
            'strong': strong,
            'moderate': moderate,
            'weak': weak,
            'non_sig': non_sig_data.sort_values('p_fdr')
        }
    
    logger.info(f"Extracted LME effects for {len(effects_dict)} effect types")
    
    return effects_dict


def extract_peak_auc_effects(
    peak_auc_results: Optional[Dict]
) -> Dict[str, Dict]:
    """
    Extract and organize peak/AUC effects by metric type.
    
    Parameters
    ----------
    peak_auc_results : dict or None
        Dictionary with 'peak' and 'auc' DataFrames
    
    Returns
    -------
    dict
        Dictionary organized by metric type (peak, time_to_peak, auc),
        each containing 'increased' and 'decreased' DataFrames
    """
    if peak_auc_results is None:
        logger.warning("No peak/AUC results available")
        return {}
    
    effects_dict = {}
    
    # Process peak results
    if 'peak' in peak_auc_results:
        peak_df = peak_auc_results['peak']
        sig_peak = peak_df[peak_df['p_fdr'] < 0.05].copy()
        
        if not sig_peak.empty:
            effects_dict['peak'] = {
                'increased': sig_peak[sig_peak['effect_r'] > 0].sort_values('effect_r', ascending=False, key=abs),
                'decreased': sig_peak[sig_peak['effect_r'] < 0].sort_values('effect_r', ascending=True, key=abs)
            }
    
    # Process AUC results
    if 'auc' in peak_auc_results:
        auc_df = peak_auc_results['auc']
        sig_auc = auc_df[auc_df['p_fdr'] < 0.05].copy()
        
        if not sig_auc.empty:
            effects_dict['auc'] = {
                'increased': sig_auc[sig_auc['effect_r'] > 0].sort_values('effect_r', ascending=False, key=abs),
                'decreased': sig_auc[sig_auc['effect_r'] < 0].sort_values('effect_r', ascending=True, key=abs)
            }
    
    # Check for time_to_peak in peak results
    if 'peak' in peak_auc_results:
        peak_df = peak_auc_results['peak']
        if 'metric' in peak_df.columns:
            ttp_df = peak_df[peak_df['metric'] == 'time_to_peak']
            sig_ttp = ttp_df[ttp_df['p_fdr'] < 0.05].copy()
            
            if not sig_ttp.empty:
                effects_dict['time_to_peak'] = {
                    'increased': sig_ttp[sig_ttp['effect_r'] > 0].sort_values('effect_r', ascending=False, key=abs),
                    'decreased': sig_ttp[sig_ttp['effect_r'] < 0].sort_values('effect_r', ascending=True, key=abs)
                }
    
    logger.info(f"Extracted peak/AUC effects for {len(effects_dict)} metric types")
    
    return effects_dict



def interpret_pca_components(
    pca_results: Optional[Dict]
) -> Dict[str, Dict]:
    """
    Interpret PCA components based on loadings.
    
    Parameters
    ----------
    pca_results : dict or None
        Dictionary with 'loadings', 'variance', 'lme' keys
    
    Returns
    -------
    dict
        Dictionary mapping component names to interpretations:
        - top_positive: List of (dimension, loading) tuples
        - top_negative: List of (dimension, loading) tuples
        - variance_explained: Variance explained by this component
        - interpretation: Text interpretation
        - lme_effects: LME results for PC scores (if available)
    """
    if pca_results is None:
        logger.warning("No PCA results available")
        return {}
    
    loadings_df = pca_results.get('loadings')
    variance_df = pca_results.get('variance')
    lme_df = pca_results.get('lme')
    
    if loadings_df is None or loadings_df.empty:
        logger.warning("No PCA loadings available")
        return {}
    
    interpretations = {}
    
    # Get unique components
    components = loadings_df['component'].unique()
    
    for component in components:
        comp_loadings = loadings_df[loadings_df['component'] == component].copy()
        comp_loadings = comp_loadings.sort_values('loading', key=abs, ascending=False)
        
        # Get top positive and negative loadings
        positive = comp_loadings[comp_loadings['loading'] > 0].head(5)
        negative = comp_loadings[comp_loadings['loading'] < 0].head(5)
        
        top_positive = [(row['dimension'], row['loading']) for _, row in positive.iterrows()]
        top_negative = [(row['dimension'], row['loading']) for _, row in negative.iterrows()]
        
        # Get variance explained
        variance_explained = np.nan
        if variance_df is not None and not variance_df.empty:
            var_row = variance_df[variance_df['component'] == component]
            if not var_row.empty:
                variance_explained = var_row['variance_explained'].values[0]
        
        # Get LME effects for this component
        lme_effects = None
        if lme_df is not None and not lme_df.empty:
            lme_effects = lme_df[lme_df['component'] == component]
        
        # Generate interpretation based on loadings
        interpretation = _interpret_component_loadings(top_positive, top_negative)
        
        interpretations[component] = {
            'top_positive': top_positive,
            'top_negative': top_negative,
            'variance_explained': variance_explained,
            'interpretation': interpretation,
            'lme_effects': lme_effects
        }
    
    logger.info(f"Interpreted {len(interpretations)} PCA components")
    
    return interpretations


def _interpret_component_loadings(
    top_positive: List[Tuple[str, float]],
    top_negative: List[Tuple[str, float]]
) -> str:
    """
    Generate interpretation text based on component loadings.
    
    Parameters
    ----------
    top_positive : list of tuples
        Top positive loadings (dimension, loading)
    top_negative : list of tuples
        Top negative loadings (dimension, loading)
    
    Returns
    -------
    str
        Interpretation text
    """
    # Simple heuristic-based interpretation
    pos_dims = [dim.lower() for dim, _ in top_positive]
    neg_dims = [dim.lower() for dim, _ in top_negative]
    
    # Check for imagery cluster
    imagery_terms = ['complex_imagery', 'elementary_imagery', 'visual']
    if any(term in ' '.join(pos_dims) for term in imagery_terms):
        if 'anxiety' in ' '.join(neg_dims) or 'unpleasantness' in ' '.join(neg_dims):
            return "Psychedelic intensity factor: high imagery with reduced negative affect"
        return "Imagery intensity factor"
    
    # Check for valence
    positive_affect = ['pleasantness', 'bliss', 'insight']
    negative_affect = ['anxiety', 'unpleasantness', 'confusion']
    
    if any(term in ' '.join(pos_dims) for term in positive_affect):
        if any(term in ' '.join(neg_dims) for term in negative_affect):
            return "Affective valence factor: positive vs negative emotional experience"
    
    # Check for ego dissolution
    if 'disembodiment' in ' '.join(pos_dims) or 'selfhood' in ' '.join(neg_dims):
        return "Ego dissolution factor"
    
    return "Mixed experiential factor"


def extract_clustering_results(
    clustering_results: Optional[Dict]
) -> Dict:
    """
    Extract and characterize clustering results.
    
    Parameters
    ----------
    clustering_results : dict or None
        Dictionary with clustering data
    
    Returns
    -------
    dict
        Dictionary with:
        - optimal_k: Optimal number of clusters
        - silhouette_score: Quality metric
        - stability: Bootstrap ARI statistics
        - clusters: Dict mapping cluster IDs to characterizations
        - dose_effects: Significant dose effects on occupancy
    """
    if clustering_results is None:
        logger.warning("No clustering results available")
        return {}
    
    result = {
        'optimal_k': None,
        'silhouette_score': None,
        'stability': None,
        'clusters': {},
        'dose_effects': []
    }
    
    # Extract optimal k and quality metrics
    metrics_df = clustering_results.get('metrics')
    if metrics_df is not None and not metrics_df.empty:
        # Assume k=2 is optimal (or find max silhouette)
        kmeans_metrics = metrics_df[metrics_df['method'] == 'kmeans']
        if not kmeans_metrics.empty:
            best = kmeans_metrics.loc[kmeans_metrics['silhouette_score'].idxmax()]
            result['optimal_k'] = int(best['n_states'])
            result['silhouette_score'] = best['silhouette_score']
    
    # Extract cluster characterizations from assignments
    assignments_df = clustering_results.get('kmeans_assignments')
    if assignments_df is not None and not assignments_df.empty:
        # Get z-scored dimension columns
        dim_cols = [col for col in assignments_df.columns if col.endswith('_z')]
        
        for cluster_id in assignments_df['cluster'].unique():
            cluster_data = assignments_df[assignments_df['cluster'] == cluster_id]
            
            # Compute mean z-scores for this cluster
            cluster_means = cluster_data[dim_cols].mean()
            
            # Identify elevated and suppressed dimensions
            elevated = cluster_means[cluster_means > 0.5].sort_values(ascending=False)
            suppressed = cluster_means[cluster_means < -0.5].sort_values()
            
            result['clusters'][cluster_id] = {
                'elevated': [(dim.replace('_z', ''), val) for dim, val in elevated.items()],
                'suppressed': [(dim.replace('_z', ''), val) for dim, val in suppressed.items()],
                'n_observations': len(cluster_data)
            }
    
    # Extract dose effects
    dose_tests = clustering_results.get('dose_tests')
    if dose_tests is not None and not dose_tests.empty:
        sig_effects = dose_tests[dose_tests['p_fdr'] < 0.05]
        result['dose_effects'] = sig_effects.to_dict('records')
    
    logger.info(f"Extracted clustering results: k={result['optimal_k']}, {len(result['clusters'])} clusters")
    
    return result


def compute_cross_method_rankings(
    lme_results: Optional[pd.DataFrame],
    peak_auc_results: Optional[Dict],
    pca_results: Optional[Dict],
    clustering_results: Optional[Dict]
) -> Dict:
    """
    Compute dimension rankings across methods and correlations.
    
    Parameters
    ----------
    lme_results : pd.DataFrame or None
        LME coefficients
    peak_auc_results : dict or None
        Peak/AUC results
    pca_results : dict or None
        PCA results
    clustering_results : dict or None
        Clustering results
    
    Returns
    -------
    dict
        Dictionary with:
        - rankings: Dict mapping method names to ranked dimension lists
        - correlations: Dict of pairwise Spearman correlations
        - convergent_dimensions: List of dimensions consistent across methods
    """
    rankings = {}
    
    # LME State effect rankings
    if lme_results is not None and not lme_results.empty:
        state_effects = lme_results[lme_results['effect'].str.contains('state', case=False)]
        state_effects = state_effects.sort_values('beta', key=abs, ascending=False)
        rankings['LME'] = state_effects['dimension'].tolist()
    
    # Peak dose effect rankings
    if peak_auc_results is not None and 'peak' in peak_auc_results:
        peak_df = peak_auc_results['peak']
        peak_df = peak_df.sort_values('effect_r', key=abs, ascending=False)
        rankings['Peak'] = peak_df['dimension'].tolist()
    
    # PCA PC1 loading rankings
    if pca_results is not None and 'loadings' in pca_results:
        loadings_df = pca_results['loadings']
        pc1 = loadings_df[loadings_df['component'] == 'PC1']
        pc1 = pc1.sort_values('loading', key=abs, ascending=False)
        rankings['PCA'] = pc1['dimension'].tolist()
    
    # Clustering discriminability rankings
    if clustering_results is not None and 'clusters' in clustering_results:
        cluster_data = clustering_results['clusters']
        if len(cluster_data) >= 2:
            # Compute discriminability as difference between cluster means
            dims_scores = {}
            for cluster_id, cluster_info in cluster_data.items():
                for dim, val in cluster_info['elevated'] + cluster_info['suppressed']:
                    if dim not in dims_scores:
                        dims_scores[dim] = []
                    dims_scores[dim].append(abs(val))
            
            # Average absolute z-scores
            dim_discriminability = {dim: np.mean(scores) for dim, scores in dims_scores.items()}
            sorted_dims = sorted(dim_discriminability.items(), key=lambda x: x[1], reverse=True)
            rankings['Clustering'] = [dim for dim, _ in sorted_dims]
    
    # Compute pairwise correlations
    correlations = {}
    method_names = list(rankings.keys())
    
    for i, method1 in enumerate(method_names):
        for method2 in method_names[i+1:]:
            # Find common dimensions
            common_dims = set(rankings[method1]) & set(rankings[method2])
            if len(common_dims) < 3:
                continue
            
            # Get ranks for common dimensions
            ranks1 = [rankings[method1].index(dim) for dim in common_dims]
            ranks2 = [rankings[method2].index(dim) for dim in common_dims]
            
            # Compute Spearman correlation
            if len(ranks1) > 2:
                corr, p_value = spearmanr(ranks1, ranks2)
                correlations[f"{method1}_vs_{method2}"] = {
                    'correlation': corr,
                    'p_value': p_value,
                    'n_dimensions': len(common_dims)
                }
    
    # Identify convergent dimensions (appear in top 5 of at least 3 methods)
    convergent = []
    all_dims = set()
    for ranking in rankings.values():
        all_dims.update(ranking[:5])
    
    for dim in all_dims:
        count = sum(1 for ranking in rankings.values() if dim in ranking[:5])
        if count >= min(3, len(rankings)):
            convergent.append(dim)
    
    logger.info(f"Computed cross-method rankings: {len(rankings)} methods, {len(convergent)} convergent dimensions")
    
    return {
        'rankings': rankings,
        'correlations': correlations,
        'convergent_dimensions': convergent
    }


def extract_methodological_metadata(
    descriptive_data: Optional[Dict],
    lme_results: Optional[pd.DataFrame],
    pca_results: Optional[Dict],
    clustering_results: Optional[Dict]
) -> Dict:
    """
    Extract methodological metadata from analysis results.
    
    Parameters
    ----------
    descriptive_data : dict or None
        Descriptive statistics
    lme_results : pd.DataFrame or None
        LME results
    pca_results : dict or None
        PCA results
    clustering_results : dict or None
        Clustering results
    
    Returns
    -------
    dict
        Dictionary with methodological metadata
    """
    metadata = {
        'data_quality': {},
        'preprocessing': {},
        'models': {},
        'decisions': {}
    }
    
    # Data quality
    if descriptive_data is not None and 'summaries' in descriptive_data:
        summaries = descriptive_data['summaries']
        metadata['data_quality'] = {
            'n_subjects': summaries['subject'].nunique() if 'subject' in summaries.columns else 'unknown',
            'n_sessions': len(summaries) if not summaries.empty else 0,
            'temporal_resolution': '0.25 Hz (4-second sampling)',
            'session_lengths': 'RS: 150 points (10 min), DMT: 300 points (20 min)'
        }
    
    # Preprocessing
    metadata['preprocessing'] = {
        'standardization': 'Global within-subject z-scoring',
        'time_windows': 'LME: 0-9 min, AUC: 0-9 min',
        'trimming': 'RS: 0-10 min, DMT: 0-20 min'
    }
    
    # Models
    if lme_results is not None:
        metadata['models']['LME'] = 'Random intercepts, fixed effects: State, Dose, Time, interactions'
    
    if pca_results is not None and 'variance' in pca_results:
        variance_df = pca_results['variance']
        if not variance_df.empty:
            n_components = len(variance_df)
            cum_var = variance_df['cumulative_variance'].max() if 'cumulative_variance' in variance_df.columns else 'unknown'
            metadata['models']['PCA'] = f'{n_components} components retained, {cum_var:.1%} variance explained'
    
    if clustering_results is not None:
        cluster_info = extract_clustering_results(clustering_results)
        if cluster_info['optimal_k']:
            metadata['models']['Clustering'] = f'KMeans k={cluster_info["optimal_k"]}, silhouette={cluster_info["silhouette_score"]:.2f}'
    
    # Analytical decisions
    metadata['decisions'] = {
        'fdr_correction': 'Benjamini-Hochberg, applied separately per effect type',
        'significance_threshold': 'p_fdr < 0.05',
        'pca_threshold': '70-80% cumulative variance',
        'clustering_selection': 'Highest silhouette score'
    }
    
    logger.info("Extracted methodological metadata")
    
    return metadata


def detect_ambiguous_findings(
    lme_results: Optional[pd.DataFrame],
    peak_auc_results: Optional[Dict],
    pca_results: Optional[Dict],
    clustering_results: Optional[Dict]
) -> List[Dict]:
    """
    Detect ambiguous or contradictory findings.
    
    Parameters
    ----------
    lme_results : pd.DataFrame or None
        LME results
    peak_auc_results : dict or None
        Peak/AUC results
    pca_results : dict or None
        PCA results
    clustering_results : dict or None
        Clustering results
    
    Returns
    -------
    list of dict
        List of ambiguous findings, each with:
        - type: Type of ambiguity
        - description: Description of the issue
        - suggestion: Suggested follow-up analysis
    """
    ambiguities = []
    
    # Check for non-significant trends (0.05 < p < 0.10)
    if lme_results is not None and not lme_results.empty:
        trends = lme_results[(lme_results['p_fdr'] >= 0.05) & (lme_results['p_fdr'] < 0.10)]
        if not trends.empty:
            for _, row in trends.head(3).iterrows():
                ambiguities.append({
                    'type': 'Non-significant trend',
                    'description': f"{row['dimension']} shows trend for {row['effect']} (p_fdr = {row['p_fdr']:.3f})",
                    'suggestion': 'Increase sample size or examine individual differences'
                })
    
    # Check for contradictory results between methods
    if lme_results is not None and peak_auc_results is not None:
        # Find dimensions with significant LME dose effect but non-significant peak effect
        lme_dose = lme_results[
            (lme_results['effect'].str.contains('dose', case=False)) &
            (lme_results['p_fdr'] < 0.05)
        ]
        
        if 'peak' in peak_auc_results:
            peak_df = peak_auc_results['peak']
            for _, lme_row in lme_dose.iterrows():
                dim = lme_row['dimension']
                peak_row = peak_df[peak_df['dimension'] == dim]
                if not peak_row.empty and peak_row['p_fdr'].values[0] >= 0.05:
                    ambiguities.append({
                        'type': 'Method disagreement',
                        'description': f"{dim} shows LME dose effect but not peak dose effect",
                        'suggestion': 'Examine temporal profiles - effect may be in timing rather than magnitude'
                    })
    
    # Check for heterogeneous clusters
    if clustering_results is not None:
        cluster_info = extract_clustering_results(clustering_results)
        for cluster_id, cluster_data in cluster_info['clusters'].items():
            if len(cluster_data['elevated']) < 2 and len(cluster_data['suppressed']) < 2:
                ambiguities.append({
                    'type': 'Heterogeneous cluster',
                    'description': f"Cluster {cluster_id} has few defining dimensions",
                    'suggestion': 'Consider k=3 or k=4 solutions to separate heterogeneous states'
                })
    
    logger.info(f"Detected {len(ambiguities)} ambiguous findings")
    
    return ambiguities
