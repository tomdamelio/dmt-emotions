# -*- coding: utf-8 -*-
"""
Generate Clustering Results Summary Report

This script loads all clustering analysis outputs and generates a comprehensive
markdown report suitable for inclusion in research diary or manuscript.

Usage:
    python scripts/generate_clustering_report.py \
        --input results/tet/clustering \
        --output results/tet/clustering/clustering_summary_report.md \
        --alpha 0.05
"""

import pandas as pd
import numpy as np
import argparse
import os
import sys
from pathlib import Path
from datetime import datetime


def load_results(input_dir: str) -> dict:
    """Load all clustering result files."""
    results = {}
    
    files = {
        'evaluation': 'clustering_evaluation.csv',
        'bootstrap': 'clustering_bootstrap_stability.csv',
        'metrics': 'clustering_state_metrics.csv',
        'classical': 'clustering_dose_tests_classical.csv',
        'permutation': 'clustering_dose_tests_permutation.csv',
        'interaction': 'clustering_interaction_effects.csv'
    }
    
    for key, filename in files.items():
        filepath = os.path.join(input_dir, filename)
        if os.path.exists(filepath):
            results[key] = pd.read_csv(filepath)
        else:
            results[key] = None
    
    return results


def generate_report(results: dict, alpha: float = 0.05) -> str:
    """Generate markdown report from clustering results."""
    report = []
    
    # Header
    report.append("# Clustering Analysis Summary Report")
    report.append(f"\n**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"\n**Significance Level**: α = {alpha}")
    report.append("\n---\n")
    
    # Executive Summary
    report.append("## Executive Summary\n")
    report.extend(generate_executive_summary(results, alpha))
    
    # Model Evaluation
    report.append("\n## Model Evaluation\n")
    report.extend(generate_model_evaluation_section(results))
    
    # Bootstrap Stability
    report.append("\n## Bootstrap Stability\n")
    report.extend(generate_stability_section(results))
    
    # State Occupancy Metrics
    report.append("\n## State Occupancy Metrics\n")
    report.extend(generate_metrics_section(results))
    
    # Dose Effects
    report.append("\n## Dose Effects\n")
    report.extend(generate_dose_effects_section(results, alpha))
    
    # Interaction Effects
    report.append("\n## Interaction Effects (State × Dose)\n")
    report.extend(generate_interaction_section(results, alpha))
    
    # Key Findings
    report.append("\n## Key Findings and Interpretations\n")
    report.extend(generate_key_findings(results, alpha))
    
    # Figures
    report.append("\n## Generated Figures\n")
    report.extend(generate_figures_section())
    
    return "\n".join(report)


def generate_executive_summary(results: dict, alpha: float) -> list:
    """Generate executive summary section."""
    summary = []
    
    # Optimal models
    if results['evaluation'] is not None:
        eval_df = results['evaluation']
        
        kmeans_results = eval_df[eval_df['method'] == 'KMeans']
        if len(kmeans_results) > 0:
            optimal_idx = kmeans_results['silhouette_score'].idxmax()
            optimal_k = int(kmeans_results.loc[optimal_idx, 'n_states'])
            optimal_sil = kmeans_results.loc[optimal_idx, 'silhouette_score']
            
            summary.append(f"**Optimal KMeans Model**: k = {optimal_k} clusters "
                          f"(silhouette = {optimal_sil:.3f})")
        
        glhmm_results = eval_df[eval_df['method'] == 'GLHMM']
        if len(glhmm_results) > 0:
            has_bic = glhmm_results['bic'].notna().any()
            if has_bic:
                optimal_idx = glhmm_results['bic'].idxmin()
                optimal_s = int(glhmm_results.loc[optimal_idx, 'n_states'])
                optimal_bic = glhmm_results.loc[optimal_idx, 'bic']
                summary.append(f"\n**Optimal GLHMM Model**: S = {optimal_s} states "
                              f"(BIC = {optimal_bic:.2f})")
    
    # Stability
    if results['bootstrap'] is not None and len(results['bootstrap']) > 0:
        bootstrap_df = results['bootstrap']
        for _, row in bootstrap_df.iterrows():
            method = row['method']
            mean_ari = row['mean_ari']
            stability = "highly stable" if mean_ari >= 0.8 else \
                       "moderately stable" if mean_ari >= 0.6 else \
                       "weakly stable" if mean_ari >= 0.4 else "unstable"
            summary.append(f"\n**{method} Stability**: {stability} "
                          f"(mean ARI = {mean_ari:.3f})")
    
    # Significant effects count
    n_dose_effects = 0
    n_interactions = 0
    
    if results['permutation'] is not None:
        perm_df = results['permutation']
        p_col = 'p_fdr' if 'p_fdr' in perm_df.columns else 'p_value_perm'
        n_dose_effects = np.sum(perm_df[p_col] < alpha)
    
    if results['interaction'] is not None:
        int_df = results['interaction']
        p_col = 'p_fdr' if 'p_fdr' in int_df.columns else 'p_value_perm'
        n_interactions = np.sum(int_df[p_col] < alpha)
    
    summary.append(f"\n\n**Significant Dose Effects**: {n_dose_effects}")
    summary.append(f"\n**Significant Interactions**: {n_interactions}")
    
    return summary


def generate_model_evaluation_section(results: dict) -> list:
    """Generate model evaluation section."""
    section = []
    
    if results['evaluation'] is None or len(results['evaluation']) == 0:
        section.append("No model evaluation results available.")
        return section
    
    eval_df = results['evaluation']
    
    # KMeans table
    kmeans_results = eval_df[eval_df['method'] == 'KMeans']
    if len(kmeans_results) > 0:
        section.append("### KMeans Clustering\n")
        section.append("| k | Silhouette Score | Interpretation |")
        section.append("|---|------------------|----------------|")
        
        for _, row in kmeans_results.iterrows():
            k = int(row['n_states'])
            sil = row['silhouette_score']
            
            if sil >= 0.7:
                interp = "Strong structure"
            elif sil >= 0.5:
                interp = "Reasonable structure"
            elif sil >= 0.25:
                interp = "Weak structure"
            else:
                interp = "No substantial structure"
            
            marker = "**" if sil == kmeans_results['silhouette_score'].max() else ""
            section.append(f"| {marker}{k}{marker} | {marker}{sil:.3f}{marker} | {marker}{interp}{marker} |")
    
    # GLHMM table
    glhmm_results = eval_df[eval_df['method'] == 'GLHMM']
    if len(glhmm_results) > 0:
        section.append("\n### GLHMM State Models\n")
        
        has_bic = glhmm_results['bic'].notna().any()
        has_fe = glhmm_results['free_energy'].notna().any()
        
        if has_bic:
            section.append("| S | BIC | Interpretation |")
            section.append("|---|-----|----------------|")
            
            for _, row in glhmm_results.iterrows():
                s = int(row['n_states'])
                bic = row['bic']
                marker = "**" if bic == glhmm_results['bic'].min() else ""
                section.append(f"| {marker}{s}{marker} | {marker}{bic:.2f}{marker} | "
                              f"{marker}Lower is better{marker} |")
        
        elif has_fe:
            section.append("| S | Free Energy | Interpretation |")
            section.append("|---|-------------|----------------|")
            
            for _, row in glhmm_results.iterrows():
                s = int(row['n_states'])
                fe = row['free_energy']
                marker = "**" if fe == glhmm_results['free_energy'].max() else ""
                section.append(f"| {marker}{s}{marker} | {marker}{fe:.2f}{marker} | "
                              f"{marker}Higher is better{marker} |")
    
    return section


def generate_stability_section(results: dict) -> list:
    """Generate bootstrap stability section."""
    section = []
    
    if results['bootstrap'] is None or len(results['bootstrap']) == 0:
        section.append("No bootstrap stability results available.")
        return section
    
    bootstrap_df = results['bootstrap']
    
    section.append("| Method | n_states | Mean ARI | 95% CI | Interpretation |")
    section.append("|--------|----------|----------|--------|----------------|")
    
    for _, row in bootstrap_df.iterrows():
        method = row['method']
        n_states = int(row['n_states'])
        mean_ari = row['mean_ari']
        ci_lower = row['ci_lower']
        ci_upper = row['ci_upper']
        
        if mean_ari >= 0.8:
            interp = "Highly stable"
        elif mean_ari >= 0.6:
            interp = "Moderately stable"
        elif mean_ari >= 0.4:
            interp = "Weakly stable"
        else:
            interp = "Unstable"
        
        section.append(f"| {method} | {n_states} | {mean_ari:.3f} | "
                      f"[{ci_lower:.3f}, {ci_upper:.3f}] | {interp} |")
    
    section.append("\n**Interpretation**: ARI (Adjusted Rand Index) measures clustering "
                  "similarity. Higher values indicate more stable, reproducible clustering.")
    
    return section



def generate_metrics_section(results: dict) -> list:
    """Generate state occupancy metrics section."""
    section = []
    
    if results['metrics'] is None or len(results['metrics']) == 0:
        section.append("No state occupancy metrics available.")
        return section
    
    metrics_df = results['metrics']
    
    for method in metrics_df['method'].unique():
        method_data = metrics_df[metrics_df['method'] == method]
        
        section.append(f"### {method}\n")
        section.append("| Cluster/State | Fractional Occupancy | Number of Visits | Mean Dwell Time (bins) |")
        section.append("|---------------|---------------------|------------------|------------------------|")
        
        for cluster_state in sorted(method_data['cluster_state'].unique()):
            cluster_data = method_data[method_data['cluster_state'] == cluster_state]
            
            frac_occ_mean = cluster_data['fractional_occupancy'].mean()
            frac_occ_std = cluster_data['fractional_occupancy'].std()
            
            n_visits_mean = cluster_data['n_visits'].mean()
            n_visits_std = cluster_data['n_visits'].std()
            
            dwell_mean = cluster_data['mean_dwell_time'].mean()
            dwell_std = cluster_data['mean_dwell_time'].std()
            
            section.append(f"| {cluster_state} | {frac_occ_mean:.3f} ± {frac_occ_std:.3f} | "
                          f"{n_visits_mean:.1f} ± {n_visits_std:.1f} | "
                          f"{dwell_mean:.1f} ± {dwell_std:.1f} |")
        
        section.append("")
    
    section.append("**Note**: Values shown as mean ± SD across all subject-sessions. "
                  "Dwell times are in bins (multiply by 4 for seconds at 0.25 Hz sampling).")
    
    return section


def generate_dose_effects_section(results: dict, alpha: float) -> list:
    """Generate dose effects section."""
    section = []
    
    # Permutation test results
    if results['permutation'] is not None and len(results['permutation']) > 0:
        perm_df = results['permutation']
        p_col = 'p_fdr' if 'p_fdr' in perm_df.columns else 'p_value_perm'
        
        sig_results = perm_df[perm_df[p_col] < alpha]
        
        section.append(f"### Significant Effects (Permutation Tests, p < {alpha})\n")
        
        if len(sig_results) > 0:
            section.append("| Metric | Method | State | Observed Δ | p-value | Direction |")
            section.append("|--------|--------|-------|------------|---------|-----------|")
            
            for _, row in sig_results.iterrows():
                metric = row['metric'].replace('_', ' ').title()
                method = row['method']
                cluster_state = int(row['cluster_state'])
                observed = row['observed_stat']
                p_value = row[p_col]
                direction = "High > Low" if observed > 0 else "High < Low"
                
                section.append(f"| {metric} | {method} | {cluster_state} | "
                              f"{observed:.4f} | {p_value:.4f} | {direction} |")
            
            section.append(f"\n**Summary**: {len(sig_results)} significant dose effects detected.")
        else:
            section.append("No significant dose effects detected at p < {alpha}.")
    
    # Classical test results
    if results['classical'] is not None and len(results['classical']) > 0:
        classical_df = results['classical']
        p_col = 'p_fdr' if 'p_fdr' in classical_df.columns else 'p_value'
        
        sig_results = classical_df[classical_df[p_col] < alpha]
        
        section.append(f"\n### Classical t-test Results (p < {alpha})\n")
        
        if len(sig_results) > 0:
            section.append("| Metric | Method | State | Mean Diff | t-statistic | p-value | 95% CI |")
            section.append("|--------|--------|-------|-----------|-------------|---------|--------|")
            
            for _, row in sig_results.iterrows():
                metric = row['metric'].replace('_', ' ').title()
                method = row['method']
                cluster_state = int(row['cluster_state'])
                mean_diff = row['mean_diff']
                t_stat = row['t_statistic']
                p_value = row[p_col]
                ci_lower = row['ci_lower']
                ci_upper = row['ci_upper']
                
                section.append(f"| {metric} | {method} | {cluster_state} | "
                              f"{mean_diff:.4f} | {t_stat:.2f} | {p_value:.4f} | "
                              f"[{ci_lower:.4f}, {ci_upper:.4f}] |")
            
            section.append(f"\n**Summary**: {len(sig_results)} significant effects (classical tests).")
        else:
            section.append(f"No significant effects detected at p < {alpha}.")
    
    return section


def generate_interaction_section(results: dict, alpha: float) -> list:
    """Generate interaction effects section."""
    section = []
    
    if results['interaction'] is None or len(results['interaction']) == 0:
        section.append("No interaction effect results available.")
        return section
    
    int_df = results['interaction']
    p_col = 'p_fdr' if 'p_fdr' in int_df.columns else 'p_value_perm'
    
    sig_results = int_df[int_df[p_col] < alpha]
    
    if len(sig_results) > 0:
        section.append(f"### Significant Interactions (p < {alpha})\n")
        section.append("| Metric | Method | State | Interaction | DMT Δ | RS Δ | p-value | Interpretation |")
        section.append("|--------|--------|-------|-------------|-------|------|---------|----------------|")
        
        for _, row in sig_results.iterrows():
            metric = row['metric'].replace('_', ' ').title()
            method = row['method']
            cluster_state = int(row['cluster_state'])
            interaction = row['interaction_effect']
            dmt_diff = row['dmt_diff']
            rs_diff = row['rs_diff']
            p_value = row[p_col]
            
            if interaction > 0:
                interp = "Stronger in DMT"
            else:
                interp = "Weaker in DMT"
            
            section.append(f"| {metric} | {method} | {cluster_state} | "
                          f"{interaction:.4f} | {dmt_diff:.4f} | {rs_diff:.4f} | "
                          f"{p_value:.4f} | {interp} |")
        
        section.append(f"\n**Summary**: {len(sig_results)} significant State × Dose interactions.")
        section.append("\n**Interpretation**: Positive interaction indicates dose effect is "
                      "stronger in DMT than RS. Negative interaction indicates dose effect is "
                      "weaker in DMT than RS.")
    else:
        section.append(f"No significant State × Dose interactions detected at p < {alpha}.")
    
    return section


def generate_key_findings(results: dict, alpha: float) -> list:
    """Generate key findings and interpretations."""
    findings = []
    
    # Model selection findings
    if results['evaluation'] is not None:
        eval_df = results['evaluation']
        
        kmeans_results = eval_df[eval_df['method'] == 'KMeans']
        if len(kmeans_results) > 0:
            optimal_idx = kmeans_results['silhouette_score'].idxmax()
            optimal_k = int(kmeans_results.loc[optimal_idx, 'n_states'])
            optimal_sil = kmeans_results.loc[optimal_idx, 'silhouette_score']
            
            findings.append(f"1. **Experiential State Structure**: KMeans clustering identified "
                           f"{optimal_k} distinct experiential states with ")
            
            if optimal_sil >= 0.5:
                findings.append("well-separated clusters, suggesting clear boundaries between "
                               "experiential states.")
            else:
                findings.append("moderate cluster separation, suggesting some overlap between "
                               "experiential states.")
    
    # Stability findings
    if results['bootstrap'] is not None and len(results['bootstrap']) > 0:
        bootstrap_df = results['bootstrap']
        
        for _, row in bootstrap_df.iterrows():
            method = row['method']
            mean_ari = row['mean_ari']
            
            findings.append(f"\n2. **Clustering Stability**: {method} clustering showed ")
            
            if mean_ari >= 0.6:
                findings.append(f"good stability (mean ARI = {mean_ari:.3f}), indicating "
                               "robust state identification across bootstrap samples.")
            else:
                findings.append(f"moderate stability (mean ARI = {mean_ari:.3f}), suggesting "
                               "some variability in state assignments.")
    
    # Dose effect findings
    if results['permutation'] is not None:
        perm_df = results['permutation']
        p_col = 'p_fdr' if 'p_fdr' in perm_df.columns else 'p_value_perm'
        sig_results = perm_df[perm_df[p_col] < alpha]
        
        if len(sig_results) > 0:
            findings.append(f"\n3. **Dose Effects**: Significant dose effects were observed for "
                           f"{len(sig_results)} state occupancy metrics. ")
            
            # Summarize by metric type
            for metric in sig_results['metric'].unique():
                metric_results = sig_results[sig_results['metric'] == metric]
                n_sig = len(metric_results)
                metric_name = metric.replace('_', ' ').title()
                
                findings.append(f"\n   - {metric_name}: {n_sig} significant effect(s)")
        else:
            findings.append("\n3. **Dose Effects**: No significant dose effects were detected "
                           "on state occupancy metrics, suggesting dose-independent state dynamics.")
    
    # Interaction findings
    if results['interaction'] is not None:
        int_df = results['interaction']
        p_col = 'p_fdr' if 'p_fdr' in int_df.columns else 'p_value_perm'
        sig_results = int_df[int_df[p_col] < alpha]
        
        if len(sig_results) > 0:
            findings.append(f"\n4. **State × Dose Interactions**: Significant interactions were "
                           f"found for {len(sig_results)} metrics, indicating that dose effects "
                           "differ between DMT and resting state conditions. This suggests that "
                           "the drug modulates dose sensitivity.")
        else:
            findings.append("\n4. **State × Dose Interactions**: No significant interactions were "
                           "detected, suggesting dose effects are consistent across experimental "
                           "conditions.")
    
    return findings


def generate_figures_section() -> list:
    """Generate figures reference section."""
    section = []
    
    section.append("The following figures were generated from the clustering analysis:\n")
    section.append("- **clustering_kmeans_centroids_k2.png**: Centroid profile plots showing "
                  "characteristic dimension patterns for each cluster (replicates Fig. 3.5)")
    section.append("- **clustering_kmeans_prob_timecourses_dmt_only.png**: Time-course cluster "
                  "probability plots showing temporal dynamics for DMT sessions (replicates Fig. 3.6)")
    
    section.append("\n**Note**: GLHMM-related figures (state probability time courses and "
                  "KMeans-GLHMM correspondence) were not generated because the GLHMM library "
                  "is not installed. To generate these figures, install GLHMM with: "
                  "`pip install git+https://github.com/vidaurre/glhmm` and rerun the analysis.")
    
    section.append("\nRefer to these figures for visual interpretation of the clustering results.")
    
    return section


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='Generate clustering results summary report',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--input',
        type=str,
        default='results/tet/clustering',
        help='Input directory containing clustering results (default: results/tet/clustering)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='results/tet/clustering/clustering_summary_report.md',
        help='Output path for markdown report (default: results/tet/clustering/clustering_summary_report.md)'
    )
    
    parser.add_argument(
        '--alpha',
        type=float,
        default=0.05,
        help='Significance threshold (default: 0.05)'
    )
    
    args = parser.parse_args()
    
    # Check if input directory exists
    if not os.path.exists(args.input):
        print(f"Error: Input directory not found: {args.input}")
        sys.exit(1)
    
    print(f"\nGenerating clustering summary report...")
    print(f"Input directory: {args.input}")
    print(f"Output file: {args.output}")
    print(f"Significance level: α = {args.alpha}\n")
    
    # Load results
    print("Loading clustering results...")
    results = load_results(args.input)
    
    # Count loaded files
    n_loaded = sum(1 for v in results.values() if v is not None)
    print(f"Loaded {n_loaded}/6 result files\n")
    
    # Generate report
    print("Generating report...")
    report_content = generate_report(results, args.alpha)
    
    # Create output directory if needed
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Write report
    with open(args.output, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    print(f"\n✓ Report generated successfully: {args.output}")
    print(f"  Report length: {len(report_content)} characters")
    print(f"  Report lines: {len(report_content.splitlines())}")


if __name__ == '__main__':
    main()
