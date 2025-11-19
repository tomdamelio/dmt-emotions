"""
Comparison utilities for ICA vs PCA analysis.

This module compares ICA and PCA results to assess whether ICA reveals
experiential structure beyond the variance explained by principal components.
"""

from typing import Dict, List
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns


class TETICAComparator:
    """
    Compare ICA and PCA results to assess complementarity.
    
    Key questions:
    1. Do ICA components align with PCA components?
    2. Do ICA mixing patterns differ from PCA loadings?
    3. Do ICA and PCA show similar LME effects?
    4. Does ICA reveal patterns beyond PC1 and PC2?
    
    Example:
        >>> comparator = TETICAComparator(
        ...     ica_results={
        ...         'mixing_matrix': ica_mixing_df,
        ...         'scores': ic_scores_df,
        ...         'lme_results': ica_lme_df,
        ...         'pca_correlation': ica_pca_corr_df
        ...     },
        ...     pca_results={
        ...         'loadings': pca_loadings_df,
        ...         'scores': pc_scores_df,
        ...         'lme_results': pca_lme_df
        ...     }
        ... )
        >>> loading_comparison = comparator.compare_loadings()
        >>> lme_comparison = comparator.compare_lme_results()
        >>> comparator.generate_visualizations('results/tet/ica/figures')
        >>> comparator.generate_report('results/tet/ica/comparison_report.md')
    """
    
    def __init__(
        self,
        ica_results: Dict[str, pd.DataFrame],
        pca_results: Dict[str, pd.DataFrame]
    ):
        """
        Initialize ICA-PCA comparator.
        
        Args:
            ica_results: Dict with keys:
                - 'mixing_matrix': ICA mixing coefficients
                - 'scores': IC scores
                - 'lme_results': ICA LME results
                - 'pca_correlation': IC-PC correlations
            pca_results: Dict with keys:
                - 'loadings': PCA loadings
                - 'scores': PC scores
                - 'lme_results': PCA LME results
        """
        self.ica_results = ica_results
        self.pca_results = pca_results
        
        # Storage for comparison results
        self.comparison_report = {}
    
    def compare_loadings(self) -> pd.DataFrame:
        """
        Compare ICA mixing patterns with PCA loadings.
        
        Process:
        1. Align ICA and PCA components by correlation
        2. For each aligned pair, compute pattern similarity
        3. Identify dimensions with divergent contributions
        
        Returns:
            DataFrame with columns:
            - ic_component: ICA component name
            - pc_component: PCA component name (aligned)
            - pattern_correlation: Correlation between mixing/loading vectors
            - divergent_dimensions: List of dimensions with |mixing - loading| > 0.3
        """
        mixing_df = self.ica_results['mixing_matrix']
        loadings_df = self.pca_results['loadings']
        pca_corr_df = self.ica_results['pca_correlation']
        
        # Align components by correlation (highest absolute correlation)
        # For each IC, find the PC with highest absolute correlation
        alignment = {}
        for ic in pca_corr_df['ic_component'].unique():
            ic_corrs = pca_corr_df[pca_corr_df['ic_component'] == ic]
            best_match = ic_corrs.loc[ic_corrs['abs_correlation'].idxmax()]
            alignment[ic] = best_match['pc_component']
        
        # Compare patterns for each aligned pair
        results = []
        for ic_comp, pc_comp in alignment.items():
            # Get mixing coefficients for this IC
            ic_mixing = mixing_df[mixing_df['component'] == ic_comp].copy()
            ic_mixing = ic_mixing.set_index('dimension')['mixing_coef']
            
            # Get loadings for this PC
            pc_loading = loadings_df[loadings_df['component'] == pc_comp].copy()
            pc_loading = pc_loading.set_index('dimension')['loading']
            
            # Align by dimension
            common_dims = ic_mixing.index.intersection(pc_loading.index)
            ic_vec = ic_mixing.loc[common_dims]
            pc_vec = pc_loading.loc[common_dims]
            
            # Compute pattern correlation
            pattern_corr = ic_vec.corr(pc_vec)
            
            # Identify divergent dimensions (|mixing - loading| > 0.3)
            diff = np.abs(ic_vec - pc_vec)
            divergent_dims = diff[diff > 0.3].index.tolist()
            
            results.append({
                'ic_component': ic_comp,
                'pc_component': pc_comp,
                'pattern_correlation': pattern_corr,
                'divergent_dimensions': ', '.join(divergent_dims) if divergent_dims else 'None'
            })
        
        comparison_df = pd.DataFrame(results)
        self.comparison_report['loading_comparison'] = comparison_df
        return comparison_df
    
    def compare_lme_results(self) -> pd.DataFrame:
        """
        Compare LME results between ICA and PCA.
        
        Process:
        1. Align IC and PC components by score correlation
        2. Compare effect sizes (beta coefficients) for each fixed effect
        3. Identify effects that differ between ICA and PCA
        
        Returns:
            DataFrame with columns:
            - effect: Fixed effect name
            - ic_component: ICA component name
            - pc_component: PCA component name (aligned)
            - ic_beta: ICA coefficient
            - pc_beta: PCA coefficient
            - ic_pvalue: ICA p-value
            - pc_pvalue: PCA p-value
            - beta_difference: |ic_beta - pc_beta|
            - agreement: 'convergent' if both significant, 'divergent' otherwise
        """
        ica_lme = self.ica_results['lme_results']
        pca_lme = self.pca_results['lme_results']
        pca_corr_df = self.ica_results['pca_correlation']
        
        # Align components by correlation
        alignment = {}
        for ic in pca_corr_df['ic_component'].unique():
            ic_corrs = pca_corr_df[pca_corr_df['ic_component'] == ic]
            best_match = ic_corrs.loc[ic_corrs['abs_correlation'].idxmax()]
            alignment[ic] = best_match['pc_component']
        
        # Compare LME results for each aligned pair
        results = []
        for ic_comp, pc_comp in alignment.items():
            # Get LME results for this IC
            ic_results = ica_lme[ica_lme['component'] == ic_comp]
            
            # Get LME results for this PC
            pc_results = pca_lme[pca_lme['component'] == pc_comp]
            
            # Compare each fixed effect
            for effect in ic_results['effect'].unique():
                ic_row = ic_results[ic_results['effect'] == effect]
                pc_row = pc_results[pc_results['effect'] == effect]
                
                if ic_row.empty or pc_row.empty:
                    continue
                
                ic_beta = ic_row['beta'].values[0]
                pc_beta = pc_row['beta'].values[0]
                ic_pval = ic_row['p_value'].values[0]
                pc_pval = pc_row['p_value'].values[0]
                
                beta_diff = abs(ic_beta - pc_beta)
                
                # Determine agreement
                alpha = 0.05
                ic_sig = ic_pval < alpha
                pc_sig = pc_pval < alpha
                
                if ic_sig and pc_sig:
                    # Both significant - check if same direction
                    same_direction = np.sign(ic_beta) == np.sign(pc_beta)
                    agreement = 'convergent' if same_direction else 'divergent_direction'
                elif not ic_sig and not pc_sig:
                    agreement = 'both_ns'
                else:
                    agreement = 'divergent_significance'
                
                results.append({
                    'effect': effect,
                    'ic_component': ic_comp,
                    'pc_component': pc_comp,
                    'ic_beta': ic_beta,
                    'pc_beta': pc_beta,
                    'ic_pvalue': ic_pval,
                    'pc_pvalue': pc_pval,
                    'beta_difference': beta_diff,
                    'agreement': agreement
                })
        
        comparison_df = pd.DataFrame(results)
        self.comparison_report['lme_comparison'] = comparison_df
        return comparison_df
    
    def generate_visualizations(self, output_dir: str) -> List[str]:
        """
        Generate comparison visualizations.
        
        Creates:
        1. Side-by-side heatmaps: ICA mixing vs PCA loadings
        2. Scatter plots: IC scores vs PC scores (for aligned pairs)
        3. Effect size comparison: IC vs PC LME coefficients
        4. Component interpretation plots: dimension contributions
        
        Args:
            output_dir: Directory to save figures
        
        Returns:
            List of generated figure paths
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        generated_files = []
        
        # 1. Side-by-side heatmaps: ICA mixing vs PCA loadings
        try:
            fig = self._plot_mixing_loading_heatmaps()
            heatmap_path = output_path / 'ica_pca_mixing_loadings_heatmap.png'
            fig.savefig(heatmap_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            generated_files.append(str(heatmap_path))
        except Exception as e:
            print(f"Error generating heatmap: {e}")
        
        # 2. Scatter plots: IC vs PC scores
        try:
            fig = self._plot_ic_pc_scatter()
            scatter_path = output_path / 'ica_pca_score_scatter.png'
            fig.savefig(scatter_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            generated_files.append(str(scatter_path))
        except Exception as e:
            print(f"Error generating scatter plot: {e}")
        
        # 3. Effect size comparison
        try:
            fig = self._plot_lme_comparison()
            lme_path = output_path / 'ica_pca_lme_comparison.png'
            fig.savefig(lme_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            generated_files.append(str(lme_path))
        except Exception as e:
            print(f"Error generating LME comparison: {e}")
        
        return generated_files
    
    def _plot_mixing_loading_heatmaps(self):
        """Create side-by-side heatmaps of ICA mixing and PCA loadings."""
        mixing_df = self.ica_results['mixing_matrix']
        loadings_df = self.pca_results['loadings']
        
        # Pivot to wide format
        mixing_wide = mixing_df.pivot(index='dimension', columns='component', values='mixing_coef')
        loadings_wide = loadings_df.pivot(index='dimension', columns='component', values='loading')
        
        # Create figure with two subplots
        fig, axes = plt.subplots(1, 2, figsize=(14, 8))
        
        # ICA mixing heatmap
        sns.heatmap(mixing_wide, cmap='RdBu_r', center=0, ax=axes[0], 
                    cbar_kws={'label': 'Mixing Coefficient'})
        axes[0].set_title('ICA Mixing Matrix', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Independent Component')
        axes[0].set_ylabel('Dimension')
        
        # PCA loadings heatmap
        sns.heatmap(loadings_wide, cmap='RdBu_r', center=0, ax=axes[1],
                    cbar_kws={'label': 'Loading'})
        axes[1].set_title('PCA Loadings', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Principal Component')
        axes[1].set_ylabel('')
        
        plt.tight_layout()
        return fig
    
    def _plot_ic_pc_scatter(self):
        """Create scatter plots of IC vs PC scores for aligned pairs."""
        pca_corr_df = self.ica_results['pca_correlation']
        ic_scores = self.ica_results['scores']
        pc_scores = self.pca_results['scores']
        
        # Merge scores
        metadata_cols = ['subject', 'session_id', 't_bin']
        merged = ic_scores.merge(pc_scores, on=metadata_cols, how='inner')
        
        # Get top 2 IC-PC pairs by correlation
        top_pairs = pca_corr_df.nlargest(2, 'abs_correlation')
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        for idx, (_, row) in enumerate(top_pairs.iterrows()):
            ic_col = row['ic_component']
            pc_col = row['pc_component']
            corr = row['correlation']
            
            axes[idx].scatter(merged[pc_col], merged[ic_col], alpha=0.3, s=10)
            axes[idx].set_xlabel(f'{pc_col} Score')
            axes[idx].set_ylabel(f'{ic_col} Score')
            axes[idx].set_title(f'{ic_col} vs {pc_col}\nr = {corr:.3f}')
            axes[idx].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def _plot_lme_comparison(self):
        """Create comparison plot of LME effect sizes."""
        if 'lme_comparison' not in self.comparison_report:
            self.compare_lme_results()
        
        lme_comp = self.comparison_report['lme_comparison']
        
        # Filter for main effects only (exclude intercept)
        main_effects = lme_comp[~lme_comp['effect'].str.contains('Intercept')]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Scatter plot: IC beta vs PC beta
        colors = {'convergent': 'green', 'divergent_significance': 'orange', 
                  'divergent_direction': 'red', 'both_ns': 'gray'}
        
        for agreement, group in main_effects.groupby('agreement'):
            ax.scatter(group['pc_beta'], group['ic_beta'], 
                      label=agreement, alpha=0.6, s=50,
                      color=colors.get(agreement, 'blue'))
        
        # Add diagonal line
        lims = [
            np.min([ax.get_xlim(), ax.get_ylim()]),
            np.max([ax.get_xlim(), ax.get_ylim()])
        ]
        ax.plot(lims, lims, 'k--', alpha=0.5, zorder=0)
        
        ax.set_xlabel('PCA Beta Coefficient')
        ax.set_ylabel('ICA Beta Coefficient')
        ax.set_title('LME Effect Size Comparison: ICA vs PCA')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def generate_report(self, output_path: str) -> str:
        """
        Generate comprehensive comparison report.
        
        Report sections:
        1. Executive Summary: Do ICs reveal patterns beyond PC1/PC2?
        2. Component Alignment: IC-PC correlation matrix
        3. Pattern Comparison: Dimension contribution differences
        4. LME Effect Comparison: Convergent and divergent findings
        5. Interpretation Guidelines: When to use ICA vs PCA
        6. Recommendations: Should ICA be included in main analysis?
        
        Args:
            output_path: Path to save markdown report
        
        Returns:
            Path to generated report
        """
        # Ensure comparisons are computed
        if 'loading_comparison' not in self.comparison_report:
            self.compare_loadings()
        if 'lme_comparison' not in self.comparison_report:
            self.compare_lme_results()
        
        # Build report
        report_lines = []
        
        # Header
        report_lines.append("# ICA vs PCA Comparison Report")
        report_lines.append("")
        report_lines.append("This report compares Independent Component Analysis (ICA) and Principal Component Analysis (PCA) results to assess whether ICA reveals experiential structure beyond the variance explained by principal components.")
        report_lines.append("")
        
        # 1. Executive Summary
        report_lines.append("## 1. Executive Summary")
        report_lines.append("")
        report_lines.append(self._generate_executive_summary())
        report_lines.append("")
        
        # 2. Component Alignment
        report_lines.append("## 2. Component Alignment")
        report_lines.append("")
        report_lines.append("### IC-PC Correlation Matrix")
        report_lines.append("")
        pca_corr = self.ica_results['pca_correlation']
        corr_pivot = pca_corr.pivot(index='ic_component', columns='pc_component', values='correlation')
        report_lines.append(corr_pivot.to_markdown())
        report_lines.append("")
        
        # 3. Pattern Comparison
        report_lines.append("## 3. Pattern Comparison: Dimension Contributions")
        report_lines.append("")
        loading_comp = self.comparison_report['loading_comparison']
        report_lines.append(loading_comp.to_markdown(index=False))
        report_lines.append("")
        
        # 4. LME Effect Comparison
        report_lines.append("## 4. LME Effect Comparison")
        report_lines.append("")
        lme_comp = self.comparison_report['lme_comparison']
        
        if not lme_comp.empty and 'agreement' in lme_comp.columns:
            # Convergent findings
            convergent = lme_comp[lme_comp['agreement'] == 'convergent']
            report_lines.append(f"### Convergent Findings (n={len(convergent)})")
            report_lines.append("")
            if not convergent.empty:
                report_lines.append(convergent[['effect', 'ic_component', 'pc_component', 'ic_beta', 'pc_beta']].to_markdown(index=False))
            else:
                report_lines.append("No convergent findings.")
            report_lines.append("")
            
            # Divergent findings
            divergent = lme_comp[lme_comp['agreement'].str.contains('divergent')]
            report_lines.append(f"### Divergent Findings (n={len(divergent)})")
            report_lines.append("")
            if not divergent.empty:
                report_lines.append(divergent[['effect', 'ic_component', 'pc_component', 'ic_beta', 'pc_beta', 'agreement']].to_markdown(index=False))
            else:
                report_lines.append("No divergent findings.")
            report_lines.append("")
        else:
            report_lines.append("LME comparison not available (no comparable effects found).")
            report_lines.append("")
        
        # 5. Interpretation Guidelines
        report_lines.append("## 5. Interpretation Guidelines")
        report_lines.append("")
        report_lines.append("### When to use ICA vs PCA")
        report_lines.append("")
        report_lines.append("- **PCA**: Best for identifying sources of maximum variance. Use when interested in dominant patterns.")
        report_lines.append("- **ICA**: Best for identifying statistically independent sources. Use when interested in separating mixed signals.")
        report_lines.append("")
        report_lines.append("### Interpreting Correlations")
        report_lines.append("")
        report_lines.append("- **High correlation (|r| > 0.7)**: IC and PC capture similar structure")
        report_lines.append("- **Moderate correlation (0.3 < |r| < 0.7)**: Partial overlap")
        report_lines.append("- **Low correlation (|r| < 0.3)**: IC reveals distinct structure")
        report_lines.append("")
        
        # 6. Recommendations
        report_lines.append("## 6. Recommendations")
        report_lines.append("")
        report_lines.append(self._generate_recommendations())
        report_lines.append("")
        
        # Write report
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        
        return str(output_file)
    
    def _generate_executive_summary(self) -> str:
        """Generate executive summary based on comparison results."""
        pca_corr = self.ica_results['pca_correlation']
        loading_comp = self.comparison_report['loading_comparison']
        lme_comp = self.comparison_report['lme_comparison']
        
        # Compute key metrics
        max_corr = pca_corr['abs_correlation'].max()
        n_high_corr = (pca_corr['abs_correlation'] > 0.7).sum()
        n_low_corr = (pca_corr['abs_correlation'] < 0.3).sum()
        
        # Handle empty LME comparison
        if not lme_comp.empty and 'agreement' in lme_comp.columns:
            n_convergent = (lme_comp['agreement'] == 'convergent').sum()
            n_divergent = lme_comp['agreement'].str.contains('divergent').sum()
        else:
            n_convergent = 0
            n_divergent = 0
        
        summary = []
        summary.append(f"**Key Findings:**")
        summary.append(f"- Maximum IC-PC correlation: {max_corr:.3f}")
        summary.append(f"- High correlations (|r| > 0.7): {n_high_corr}")
        summary.append(f"- Low correlations (|r| < 0.3): {n_low_corr}")
        summary.append(f"- Convergent LME effects: {n_convergent}")
        summary.append(f"- Divergent LME effects: {n_divergent}")
        summary.append("")
        
        # Interpretation
        if n_high_corr > n_low_corr:
            summary.append("**Interpretation:** ICA components show substantial overlap with PCA components, suggesting that variance-based and independence-based decompositions capture similar structure in this dataset.")
        else:
            summary.append("**Interpretation:** ICA components reveal distinct structure beyond PCA, suggesting that independence-based decomposition uncovers latent sources masked by the dominant variance structure.")
        
        return '\n'.join(summary)
    
    def _generate_recommendations(self) -> str:
        """Generate recommendations based on comparison results."""
        pca_corr = self.ica_results['pca_correlation']
        lme_comp = self.comparison_report['lme_comparison']
        
        n_low_corr = (pca_corr['abs_correlation'] < 0.3).sum()
        
        # Handle empty LME comparison
        if not lme_comp.empty and 'agreement' in lme_comp.columns:
            n_divergent = lme_comp['agreement'].str.contains('divergent').sum()
        else:
            n_divergent = 0
        
        recommendations = []
        
        if n_low_corr > 0 or n_divergent > 2:
            recommendations.append("**Recommendation: Include ICA in main analysis**")
            recommendations.append("")
            recommendations.append("ICA reveals experiential structure beyond PCA, particularly:")
            if n_low_corr > 0:
                recommendations.append(f"- {n_low_corr} independent components with low PC correlation")
            if n_divergent > 0:
                recommendations.append(f"- {n_divergent} divergent LME effects")
            recommendations.append("")
            recommendations.append("Including ICA will provide complementary insights into the independent sources of experiential variation.")
        else:
            recommendations.append("**Recommendation: PCA sufficient for main analysis**")
            recommendations.append("")
            recommendations.append("ICA components show substantial overlap with PCA components. The variance-based decomposition captures the main experiential structure.")
            recommendations.append("")
            recommendations.append("ICA can be reported as supplementary analysis for completeness.")
        
        return '\n'.join(recommendations)

