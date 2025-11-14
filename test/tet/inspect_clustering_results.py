# -*- coding: utf-8 -*-
"""
Inspect Clustering Results Script

This script loads and displays clustering analysis results, providing quick
summaries and interpretations for inclusion in lab diary or manuscript.

Usage:
    python scripts/inspect_clustering_results.py --input results/tet/clustering
"""

import pandas as pd
import numpy as np
import argparse
import os
import sys
from pathlib import Path


def load_results(input_dir: str) -> dict:
    """
    Load all clustering result files.
    
    Args:
        input_dir: Directory containing clustering results
    
    Returns:
        Dictionary of DataFrames with clustering results
    """
    results = {}
    
    # Define expected files
    files = {
        'evaluation': 'clustering_evaluation.csv',
        'bootstrap': 'clustering_bootstrap_stability.csv',
        'metrics': 'clustering_state_metrics.csv',
        'permutation': 'clustering_dose_tests_permutation.csv',
        'interaction': 'clustering_interaction_effects.csv',
        'classical': 'clustering_dose_tests_classical.csv'
    }
    
    # Load each file if it exists
    for key, filename in files.items():
        filepath = os.path.join(input_dir, filename)
        if os.path.exists(filepath):
            results[key] = pd.read_csv(filepath)
            print(f"✓ Loaded {filename}")
        else:
            print(f"✗ Missing {filename}")
            results[key] = None
    
    return results


def display_model_evaluation(evaluation_df: pd.DataFrame) -> None:
    """Display model evaluation results."""
    print("\n" + "="*80)
    print("MODEL EVALUATION")
    print("="*80)
    
    if evaluation_df is None or len(evaluation_df) == 0:
        print("No evaluation results available.")
        return
    
    # KMeans results
    kmeans_results = evaluation_df[evaluation_df['method'] == 'KMeans']
    if len(kmeans_results) > 0:
        print("\nKMeans Clustering:")
        print("-" * 40)
        for _, row in kmeans_results.iterrows():
            print(f"  k={int(row['n_states'])}: Silhouette = {row['silhouette_score']:.3f}")
        
        # Identify optimal k
        optimal_idx = kmeans_results['silhouette_score'].idxmax()
        optimal_k = int(kmeans_results.loc[optimal_idx, 'n_states'])
        optimal_sil = kmeans_results.loc[optimal_idx, 'silhouette_score']
        print(f"\n  → Optimal: k={optimal_k} (silhouette={optimal_sil:.3f})")
        
        # Interpretation
        if optimal_sil >= 0.7:
            interpretation = "Strong cluster structure"
        elif optimal_sil >= 0.5:
            interpretation = "Reasonable cluster structure"
        elif optimal_sil >= 0.25:
            interpretation = "Weak cluster structure"
        else:
            interpretation = "No substantial cluster structure"
        print(f"  → Interpretation: {interpretation}")
    
    # GLHMM results
    glhmm_results = evaluation_df[evaluation_df['method'] == 'GLHMM']
    if len(glhmm_results) > 0:
        print("\nGLHMM State Models:")
        print("-" * 40)
        
        # Check which metric is available
        has_bic = glhmm_results['bic'].notna().any()
        has_fe = glhmm_results['free_energy'].notna().any()
        
        if has_bic:
            for _, row in glhmm_results.iterrows():
                print(f"  S={int(row['n_states'])}: BIC = {row['bic']:.2f}")
            
            # Identify optimal S (lowest BIC)
            optimal_idx = glhmm_results['bic'].idxmin()
            optimal_s = int(glhmm_results.loc[optimal_idx, 'n_states'])
            optimal_bic = glhmm_results.loc[optimal_idx, 'bic']
            print(f"\n  → Optimal: S={optimal_s} (BIC={optimal_bic:.2f})")
            print(f"  → Lower BIC indicates better model fit with appropriate complexity")
        
        elif has_fe:
            for _, row in glhmm_results.iterrows():
                print(f"  S={int(row['n_states'])}: Free Energy = {row['free_energy']:.2f}")
            
            # Identify optimal S (highest free energy)
            optimal_idx = glhmm_results['free_energy'].idxmax()
            optimal_s = int(glhmm_results.loc[optimal_idx, 'n_states'])
            optimal_fe = glhmm_results.loc[optimal_idx, 'free_energy']
            print(f"\n  → Optimal: S={optimal_s} (Free Energy={optimal_fe:.2f})")
            print(f"  → Higher free energy indicates better model fit")
        else:
            print("  No BIC or Free Energy metrics available")


def display_bootstrap_stability(bootstrap_df: pd.DataFrame) -> None:
    """Display bootstrap stability results."""
    print("\n" + "="*80)
    print("BOOTSTRAP STABILITY ANALYSIS")
    print("="*80)
    
    if bootstrap_df is None or len(bootstrap_df) == 0:
        print("No bootstrap stability results available.")
        return
    
    for _, row in bootstrap_df.iterrows():
        method = row['method']
        n_states = int(row['n_states'])
        mean_ari = row['mean_ari']
        ci_lower = row['ci_lower']
        ci_upper = row['ci_upper']
        
        print(f"\n{method} (n_states={n_states}):")
        print(f"  Mean ARI: {mean_ari:.3f} [95% CI: {ci_lower:.3f}, {ci_upper:.3f}]")
        
        # Interpretation
        if mean_ari >= 0.8:
            interpretation = "Highly stable clustering"
        elif mean_ari >= 0.6:
            interpretation = "Moderately stable clustering"
        elif mean_ari >= 0.4:
            interpretation = "Weakly stable clustering"
        else:
            interpretation = "Unstable clustering"
        print(f"  → {interpretation}")


def display_state_metrics_summary(metrics_df: pd.DataFrame) -> None:
    """Display summary statistics for state occupancy metrics."""
    print("\n" + "="*80)
    print("STATE OCCUPANCY METRICS SUMMARY")
    print("="*80)
    
    if metrics_df is None or len(metrics_df) == 0:
        print("No state metrics available.")
        return
    
    # Group by method and cluster state
    for method in metrics_df['method'].unique():
        method_data = metrics_df[metrics_df['method'] == method]
        
        print(f"\n{method}:")
        print("-" * 40)
        
        for cluster_state in sorted(method_data['cluster_state'].unique()):
            cluster_data = method_data[method_data['cluster_state'] == cluster_state]
            
            print(f"\n  Cluster/State {cluster_state}:")
            
            # Fractional occupancy
            frac_occ = cluster_data['fractional_occupancy']
            print(f"    Fractional Occupancy: {frac_occ.mean():.3f} ± {frac_occ.std():.3f}")
            
            # Number of visits
            n_visits = cluster_data['n_visits']
            print(f"    Number of Visits: {n_visits.mean():.1f} ± {n_visits.std():.1f}")
            
            # Mean dwell time
            dwell_time = cluster_data['mean_dwell_time']
            print(f"    Mean Dwell Time: {dwell_time.mean():.1f} ± {dwell_time.std():.1f} bins")


def display_dose_effects(permutation_df: pd.DataFrame, classical_df: pd.DataFrame = None) -> None:
    """Display significant dose effects from permutation and classical tests."""
    print("\n" + "="*80)
    print("SIGNIFICANT DOSE EFFECTS")
    print("="*80)
    
    # Permutation test results
    if permutation_df is not None and len(permutation_df) > 0:
        print("\nPermutation Test Results:")
        print("-" * 40)
        
        # Check if FDR correction was applied
        has_fdr = 'p_fdr' in permutation_df.columns
        p_col = 'p_fdr' if has_fdr else 'p_value_perm'
        sig_col = 'significant' if 'significant' in permutation_df.columns else None
        
        if sig_col is not None:
            sig_results = permutation_df[permutation_df[sig_col]]
        else:
            sig_results = permutation_df[permutation_df[p_col] < 0.05]
        
        if len(sig_results) > 0:
            print(f"\n{len(sig_results)} significant effects (p < 0.05):")
            
            # Group by metric
            for metric in sig_results['metric'].unique():
                metric_results = sig_results[sig_results['metric'] == metric]
                print(f"\n  {metric}:")
                
                for _, row in metric_results.iterrows():
                    method = row['method']
                    cluster_state = int(row['cluster_state'])
                    observed = row['observed_stat']
                    p_value = row[p_col]
                    
                    direction = "↑" if observed > 0 else "↓"
                    print(f"    {method} State {cluster_state}: {direction} "
                          f"Δ={observed:.4f}, p={p_value:.4f}")
        else:
            print("\n  No significant dose effects detected.")
    
    # Classical test results (if available)
    if classical_df is not None and len(classical_df) > 0:
        print("\n\nClassical t-test Results:")
        print("-" * 40)
        
        has_fdr = 'p_fdr' in classical_df.columns
        p_col = 'p_fdr' if has_fdr else 'p_value'
        sig_col = 'significant' if 'significant' in classical_df.columns else None
        
        if sig_col is not None:
            sig_results = classical_df[classical_df[sig_col]]
        else:
            sig_results = classical_df[classical_df[p_col] < 0.05]
        
        if len(sig_results) > 0:
            print(f"\n{len(sig_results)} significant effects (p < 0.05):")
            
            # Group by metric
            for metric in sig_results['metric'].unique():
                metric_results = sig_results[sig_results['metric'] == metric]
                print(f"\n  {metric}:")
                
                for _, row in metric_results.iterrows():
                    method = row['method']
                    cluster_state = int(row['cluster_state'])
                    mean_diff = row['mean_diff']
                    t_stat = row['t_statistic']
                    p_value = row[p_col]
                    
                    direction = "↑" if mean_diff > 0 else "↓"
                    print(f"    {method} State {cluster_state}: {direction} "
                          f"Δ={mean_diff:.4f}, t={t_stat:.2f}, p={p_value:.4f}")
        else:
            print("\n  No significant dose effects detected.")


def display_interaction_effects(interaction_df: pd.DataFrame) -> None:
    """Display significant interaction effects."""
    print("\n" + "="*80)
    print("SIGNIFICANT INTERACTION EFFECTS (State × Dose)")
    print("="*80)
    
    if interaction_df is None or len(interaction_df) == 0:
        print("No interaction effect results available.")
        return
    
    # Check if FDR correction was applied
    has_fdr = 'p_fdr' in interaction_df.columns
    p_col = 'p_fdr' if has_fdr else 'p_value_perm'
    sig_col = 'significant' if 'significant' in interaction_df.columns else None
    
    if sig_col is not None:
        sig_results = interaction_df[interaction_df[sig_col]]
    else:
        sig_results = interaction_df[interaction_df[p_col] < 0.05]
    
    if len(sig_results) > 0:
        print(f"\n{len(sig_results)} significant interactions (p < 0.05):")
        
        # Group by metric
        for metric in sig_results['metric'].unique():
            metric_results = sig_results[sig_results['metric'] == metric]
            print(f"\n  {metric}:")
            
            for _, row in metric_results.iterrows():
                method = row['method']
                cluster_state = int(row['cluster_state'])
                interaction = row['interaction_effect']
                dmt_diff = row['dmt_diff']
                rs_diff = row['rs_diff']
                p_value = row[p_col]
                
                print(f"    {method} State {cluster_state}:")
                print(f"      Interaction: {interaction:.4f}, p={p_value:.4f}")
                print(f"      DMT (High-Low): {dmt_diff:.4f}")
                print(f"      RS (High-Low): {rs_diff:.4f}")
                
                # Interpretation
                if interaction > 0:
                    print(f"      → Dose effect stronger in DMT than RS")
                else:
                    print(f"      → Dose effect weaker in DMT than RS")
    else:
        print("\n  No significant interaction effects detected.")


def generate_interpretation_template(results: dict) -> str:
    """Generate textual interpretation template for lab diary or manuscript."""
    template = []
    
    template.append("\n" + "="*80)
    template.append("INTERPRETATION TEMPLATE FOR LAB DIARY / MANUSCRIPT")
    template.append("="*80)
    
    # Model selection
    if results['evaluation'] is not None:
        eval_df = results['evaluation']
        
        kmeans_results = eval_df[eval_df['method'] == 'KMeans']
        if len(kmeans_results) > 0:
            optimal_idx = kmeans_results['silhouette_score'].idxmax()
            optimal_k = int(kmeans_results.loc[optimal_idx, 'n_states'])
            optimal_sil = kmeans_results.loc[optimal_idx, 'silhouette_score']
            
            template.append(f"\nKMeans clustering identified {optimal_k} distinct experiential states ")
            template.append(f"(silhouette score = {optimal_sil:.3f}), suggesting ")
            
            if optimal_sil >= 0.5:
                template.append("well-separated clusters with clear boundaries.")
            else:
                template.append("moderate cluster separation with some overlap between states.")
        
        glhmm_results = eval_df[eval_df['method'] == 'GLHMM']
        if len(glhmm_results) > 0:
            has_bic = glhmm_results['bic'].notna().any()
            if has_bic:
                optimal_idx = glhmm_results['bic'].idxmin()
                optimal_s = int(glhmm_results.loc[optimal_idx, 'n_states'])
                template.append(f"\n\nGLHMM analysis identified {optimal_s} temporal states, ")
                template.append("accounting for sequential dependencies in experiential dynamics.")
    
    # Stability
    if results['bootstrap'] is not None and len(results['bootstrap']) > 0:
        bootstrap_df = results['bootstrap']
        
        for _, row in bootstrap_df.iterrows():
            method = row['method']
            mean_ari = row['mean_ari']
            
            template.append(f"\n\n{method} clustering showed ")
            
            if mean_ari >= 0.6:
                template.append(f"good stability (mean ARI = {mean_ari:.3f}), ")
                template.append("indicating robust state identification across bootstrap samples.")
            else:
                template.append(f"moderate stability (mean ARI = {mean_ari:.3f}), ")
                template.append("suggesting some variability in state assignments.")
    
    # Dose effects
    if results['permutation'] is not None:
        perm_df = results['permutation']
        has_fdr = 'p_fdr' in perm_df.columns
        p_col = 'p_fdr' if has_fdr else 'p_value_perm'
        sig_col = 'significant' if 'significant' in perm_df.columns else None
        
        if sig_col is not None:
            sig_results = perm_df[perm_df[sig_col]]
        else:
            sig_results = perm_df[perm_df[p_col] < 0.05]
        
        if len(sig_results) > 0:
            template.append(f"\n\nDose effects were observed for {len(sig_results)} state occupancy metrics. ")
            
            # Summarize by metric type
            for metric in sig_results['metric'].unique():
                metric_results = sig_results[sig_results['metric'] == metric]
                n_sig = len(metric_results)
                
                template.append(f"\n{metric.replace('_', ' ').title()}: {n_sig} significant effect(s)")
        else:
            template.append("\n\nNo significant dose effects were detected on state occupancy metrics.")
    
    # Interaction effects
    if results['interaction'] is not None:
        int_df = results['interaction']
        has_fdr = 'p_fdr' in int_df.columns
        p_col = 'p_fdr' if has_fdr else 'p_value_perm'
        sig_col = 'significant' if 'significant' in int_df.columns else None
        
        if sig_col is not None:
            sig_results = int_df[int_df[sig_col]]
        else:
            sig_results = int_df[int_df[p_col] < 0.05]
        
        if len(sig_results) > 0:
            template.append(f"\n\nSignificant State × Dose interactions were found for ")
            template.append(f"{len(sig_results)} metrics, indicating that dose effects ")
            template.append("differ between DMT and resting state conditions.")
        else:
            template.append("\n\nNo significant State × Dose interactions were detected, ")
            template.append("suggesting dose effects are consistent across experimental conditions.")
    
    template.append("\n")
    
    return "".join(template)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='Inspect clustering analysis results',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--input',
        type=str,
        default='results/tet/clustering',
        help='Input directory containing clustering results (default: results/tet/clustering)'
    )
    
    args = parser.parse_args()
    
    # Check if input directory exists
    if not os.path.exists(args.input):
        print(f"Error: Input directory not found: {args.input}")
        sys.exit(1)
    
    print("\n" + "="*80)
    print("CLUSTERING RESULTS INSPECTION")
    print("="*80)
    print(f"\nInput directory: {args.input}\n")
    
    # Load results
    results = load_results(args.input)
    
    # Display results
    display_model_evaluation(results['evaluation'])
    display_bootstrap_stability(results['bootstrap'])
    display_state_metrics_summary(results['metrics'])
    display_dose_effects(results['permutation'], results['classical'])
    display_interaction_effects(results['interaction'])
    
    # Generate interpretation template
    interpretation = generate_interpretation_template(results)
    print(interpretation)
    
    print("\n" + "="*80)
    print("INSPECTION COMPLETE")
    print("="*80)


if __name__ == '__main__':
    main()
