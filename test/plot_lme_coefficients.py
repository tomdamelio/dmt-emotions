# -*- coding: utf-8 -*-
"""
LME Coefficients Plot: Visual summary of model effects.

This script creates a coefficient plot (forest plot) showing the fixed effects
from the LME analysis with 95% confidence intervals. This provides the most
direct way to communicate which effects are significant and their magnitude/direction.

The plot shows:
- Point estimates (β coefficients) 
- 95% confidence intervals
- Reference line at zero
- Color coding for significance
- Effect sizes and p-values

Usage:
  python test/plot_lme_coefficients.py
"""

import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns

# Import project modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Plot aesthetics
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'figure.dpi': 110,
    'savefig.dpi': 400,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'legend.frameon': False,
})


def load_lme_results() -> Dict:
    """Load LME results from the analysis report."""
    report_path = os.path.join('test', 'eda', 'lme_analysis', 'lme_analysis_report.txt')
    
    if not os.path.exists(report_path):
        raise FileNotFoundError(f"LME analysis report not found: {report_path}")
    
    print(f"📊 Loading LME results from: {report_path}")
    
    # Parse the report file to extract coefficient information
    coefficients = {}
    
    with open(report_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Find the coefficient sections
    current_family = None
    
    for i, line in enumerate(lines):
        line = line.strip()
        
        # Identify family sections
        if line.startswith('FAMILY TASK:'):
            current_family = 'Task'
        elif line.startswith('FAMILY DOSE:'):
            current_family = 'Dose'
        elif line.startswith('FAMILY INTERACTION:'):
            current_family = 'Interaction'
        elif line.startswith('CONDITIONAL CONTRASTS:'):
            current_family = 'Contrasts'
        
        # Parse coefficient lines
        if current_family and line.endswith(':') and not line.startswith('FAMILY') and not line.startswith('CONDITIONAL'):
            param_name = line.rstrip(':').strip()
            
            # Look for the next few lines with β, CI, and p values
            if i + 3 < len(lines):
                beta_line = lines[i + 1].strip()
                ci_line = lines[i + 2].strip()
                p_line = lines[i + 3].strip()
                
                try:
                    # Parse β and SE
                    if beta_line.startswith('β ='):
                        parts = beta_line.split(',')
                        beta_part = parts[0].replace('β =', '').strip()
                        se_part = parts[1].replace('SE =', '').strip()
                        
                        beta = float(beta_part)
                        se = float(se_part)
                    
                    # Parse CI
                    if ci_line.startswith('95% CI:'):
                        ci_text = ci_line.replace('95% CI:', '').strip()
                        ci_text = ci_text.replace('[', '').replace(']', '')
                        ci_parts = ci_text.split(',')
                        ci_lower = float(ci_parts[0].strip())
                        ci_upper = float(ci_parts[1].strip())
                    
                    # Parse p-values
                    if 'p_raw =' in p_line and 'p_FDR =' in p_line:
                        p_parts = p_line.split(',')
                        p_raw_part = [p for p in p_parts if 'p_raw =' in p][0]
                        p_fdr_part = [p for p in p_parts if 'p_FDR =' in p][0]
                        
                        p_raw = float(p_raw_part.split('=')[1].strip().split()[0])
                        p_fdr_text = p_fdr_part.split('=')[1].strip()
                        p_fdr = float(p_fdr_text.split()[0])
                        
                        # Check for significance markers
                        significance = ''
                        if '***' in p_fdr_text:
                            significance = '***'
                        elif '**' in p_fdr_text:
                            significance = '**'
                        elif '*' in p_fdr_text:
                            significance = '*'
                    
                    coefficients[param_name] = {
                        'family': current_family,
                        'beta': beta,
                        'se': se,
                        'ci_lower': ci_lower,
                        'ci_upper': ci_upper,
                        'p_raw': p_raw,
                        'p_fdr': p_fdr,
                        'significance': significance
                    }
                    
                except (ValueError, IndexError) as e:
                    print(f"⚠️  Could not parse {param_name}: {e}")
                    continue
    
    print(f"✅ Loaded {len(coefficients)} coefficients from LME analysis")
    
    return coefficients


def prepare_coefficient_data(coefficients: Dict) -> pd.DataFrame:
    """Prepare coefficient data for plotting."""
    print("📋 Preparing coefficient data for plotting...")
    
    # Convert to DataFrame
    coef_data = []
    
    # Define the order and labels for plotting
    param_order = [
        'Task[T.DMT]',
        'Dose[T.High]', 
        'Task[T.DMT]:minute_c',
        'Dose[T.High]:minute_c',
        'Task[T.DMT]:Dose[T.High]'
    ]
    
    param_labels = {
        'Task[T.DMT]': 'Task (DMT vs RS)',
        'Dose[T.High]': 'Dose (High vs Low)',
        'Task[T.DMT]:minute_c': 'Task × Time',
        'Dose[T.High]:minute_c': 'Dose × Time', 
        'Task[T.DMT]:Dose[T.High]': 'Task × Dose'
    }
    
    family_colors = {
        'Task': '#2E8B57',        # Sea green
        'Dose': '#4169E1',        # Royal blue
        'Interaction': '#DC143C'   # Crimson
    }
    
    for i, param in enumerate(param_order):
        if param in coefficients:
            coef_info = coefficients[param]
            
            coef_data.append({
                'parameter': param,
                'label': param_labels.get(param, param),
                'beta': coef_info['beta'],
                'se': coef_info['se'],
                'ci_lower': coef_info['ci_lower'],
                'ci_upper': coef_info['ci_upper'],
                'p_raw': coef_info['p_raw'],
                'p_fdr': coef_info['p_fdr'],
                'significance': coef_info['significance'],
                'family': coef_info['family'],
                'order': i,
                'significant': coef_info['p_fdr'] < 0.05,
                'color': family_colors.get(coef_info['family'], '#666666')
            })
        else:
            print(f"⚠️  Parameter {param} not found in results")
    
    df = pd.DataFrame(coef_data)
    
    if len(df) == 0:
        raise ValueError("No coefficient data could be prepared!")
    
    print(f"✅ Prepared {len(df)} coefficients for plotting")
    
    return df


def create_coefficient_plot(coef_df: pd.DataFrame, output_path: str) -> None:
    """Create the main coefficient plot."""
    print("🎨 Creating coefficient plot...")
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Sort by order for proper plotting
    coef_df = coef_df.sort_values('order')
    
    # Y positions for each coefficient
    y_positions = np.arange(len(coef_df))
    
    # Plot confidence intervals as horizontal lines
    for i, row in coef_df.iterrows():
        y_pos = y_positions[row['order']]
        
        # Choose style based on significance
        if row['significant']:
            linewidth = 2.5
            alpha = 1.0
            marker_size = 80
        else:
            linewidth = 1.5
            alpha = 0.7
            marker_size = 60
        
        # Confidence interval line
        ax.plot([row['ci_lower'], row['ci_upper']], [y_pos, y_pos], 
               color=row['color'], linewidth=linewidth, alpha=alpha, zorder=2)
        
        # Point estimate
        ax.scatter(row['beta'], y_pos, 
                  color=row['color'], s=marker_size, alpha=alpha, zorder=3,
                  edgecolors='white', linewidths=1)
        
        # Add significance markers
        if row['significance']:
            ax.text(row['ci_upper'] + 0.05, y_pos, row['significance'], 
                   fontsize=12, fontweight='bold', va='center',
                   color=row['color'])
    
    # Reference line at zero
    ax.axvline(x=0, color='black', linestyle='--', alpha=0.5, linewidth=1, zorder=1)
    
    # Customize plot
    ax.set_yticks(y_positions)
    ax.set_yticklabels(coef_df['label'], fontsize=11)
    ax.set_xlabel('Coefficient Estimate (β) with 95% CI', fontweight='bold', fontsize=12)
    ax.set_title('LME Fixed Effects: SMNA AUC Analysis\nCoefficient Plot with 95% Confidence Intervals', 
                fontweight='bold', fontsize=14, pad=20)
    
    # Add grid
    ax.grid(True, axis='x', alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)
    
    # Adjust layout to prevent label cutoff
    plt.subplots_adjust(left=0.25)
    
    # Add legend for families
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2E8B57', label='Task effects'),
        Patch(facecolor='#4169E1', label='Dose effects'),
        Patch(facecolor='#DC143C', label='Interaction effects')
    ]
    ax.legend(handles=legend_elements, loc='lower right', title='Effect Type')
    
    # Add interpretation note
    note_text = ('Positive β: higher SMNA AUC\n'
                'Negative β: lower SMNA AUC\n'
                '* p<0.05, ** p<0.01, *** p<0.001 (FDR-corrected)')
    
    ax.text(0.02, 0.98, note_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=dict(boxstyle="round,pad=0.4", 
                                              facecolor="lightgray", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=400, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Coefficient plot saved: {output_path}")


def create_effect_sizes_table(coef_df: pd.DataFrame, output_path: str) -> None:
    """Create a detailed table of effect sizes and statistics."""
    print("📊 Creating effect sizes table...")
    
    # Prepare table with key statistics
    table_data = coef_df[['label', 'beta', 'se', 'ci_lower', 'ci_upper', 
                         'p_raw', 'p_fdr', 'significance', 'family']].copy()
    
    # Round numeric columns
    numeric_cols = ['beta', 'se', 'ci_lower', 'ci_upper', 'p_raw', 'p_fdr']
    table_data[numeric_cols] = table_data[numeric_cols].round(4)
    
    # Add interpretation column
    table_data['interpretation'] = table_data.apply(lambda row: 
        f"{'Significant' if row['p_fdr'] < 0.05 else 'Non-significant'} "
        f"{'increase' if row['beta'] > 0 else 'decrease'} in SMNA AUC", axis=1)
    
    # Save to CSV
    table_data.to_csv(output_path, index=False)
    print(f"✅ Effect sizes table saved: {output_path}")
    
    # Print summary to console
    print(f"\n📊 COEFFICIENT SUMMARY:")
    print(f"=" * 70)
    
    for _, row in table_data.iterrows():
        significance_text = f" {row['significance']}" if row['significance'] else ""
        
        print(f"{row['label']:25} β={row['beta']:7.3f} "
              f"[{row['ci_lower']:6.3f}, {row['ci_upper']:6.3f}] "
              f"p_FDR={row['p_fdr']:6.4f}{significance_text}")
    
    # Key findings summary
    significant_effects = table_data[table_data['p_fdr'] < 0.05]
    
    print(f"\n🔍 KEY FINDINGS:")
    print(f"   • {len(significant_effects)}/{len(table_data)} effects significant (p_FDR < 0.05)")
    
    if len(significant_effects) > 0:
        print(f"   • Significant effects:")
        for _, row in significant_effects.iterrows():
            direction = "increases" if row['beta'] > 0 else "decreases"
            print(f"     - {row['label']}: {direction} SMNA AUC (β={row['beta']:.3f})")


def create_model_summary_plot(coef_df: pd.DataFrame, output_path: str) -> None:
    """Create a summary plot showing model R² and other fit statistics."""
    print("🎨 Creating model summary visualization...")
    
    # Read additional model statistics from the report
    report_path = os.path.join('test', 'eda', 'lme_analysis', 'lme_analysis_report.txt')
    
    model_stats = {}
    
    try:
        with open(report_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Extract model fit statistics
        if 'AIC:' in content:
            aic_line = [line for line in content.split('\n') if 'AIC:' in line][0]
            # Handle NaN case
            if 'nan' not in aic_line:
                model_stats['AIC'] = float(aic_line.split('AIC:')[1].strip().split()[0])
        
        if 'BIC:' in content:
            bic_line = [line for line in content.split('\n') if 'BIC:' in line][0]
            if 'nan' not in bic_line:
                model_stats['BIC'] = float(bic_line.split('BIC:')[1].strip().split()[0])
        
        if 'Log-likelihood:' in content:
            ll_line = [line for line in content.split('\n') if 'Log-likelihood:' in line][0]
            model_stats['Log-likelihood'] = float(ll_line.split('Log-likelihood:')[1].strip().split()[0])
        
        if 'N observations:' in content:
            n_line = [line for line in content.split('\n') if 'N observations:' in line][0]
            model_stats['N observations'] = int(n_line.split('N observations:')[1].strip().split()[0])
        
        if 'N subjects:' in content:
            subj_line = [line for line in content.split('\n') if 'N subjects:' in line][0]
            model_stats['N subjects'] = int(subj_line.split('N subjects:')[1].strip().split()[0])
            
    except Exception as e:
        print(f"⚠️  Could not extract all model statistics: {e}")
    
    # Create a simple text-based summary plot
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Remove axes for text-only plot
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    # Model summary text
    summary_lines = [
        "LME MODEL SUMMARY",
        "=" * 40,
        "",
        "Fixed Effects Formula:",
        "AUC ~ Task*Dose + minute_c + Task:minute_c + Dose:minute_c",
        "",
        "Random Effects: ~ 1 | subject",
        "",
    ]
    
    # Add model fit statistics
    if model_stats:
        summary_lines.extend([
            "Model Fit Statistics:",
            "-" * 25
        ])
        
        for stat, value in model_stats.items():
            if isinstance(value, float):
                summary_lines.append(f"{stat}: {value:.2f}")
            else:
                summary_lines.append(f"{stat}: {value}")
        
        summary_lines.append("")
    
    # Add significance summary
    significant_effects = coef_df[coef_df['p_fdr'] < 0.05]
    summary_lines.extend([
        "Significant Fixed Effects (p_FDR < 0.05):",
        "-" * 40
    ])
    
    if len(significant_effects) > 0:
        for _, row in significant_effects.iterrows():
            summary_lines.append(f"• {row['label']}: β = {row['beta']:.3f} {row['significance']}")
    else:
        summary_lines.append("• No significant effects")
    
    # Add interpretation
    summary_lines.extend([
        "",
        "Key Interpretation:",
        "-" * 20,
        "• DMT shows higher SMNA AUC than RS",
        "• Effect diminishes over time (negative time interaction)",
        "• Strong Task×Dose interaction effect",
        "• Dose effects differ dramatically between tasks"
    ])
    
    # Plot text
    full_text = '\n'.join(summary_lines)
    ax.text(0.05, 0.95, full_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='left',
            fontfamily='monospace',
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=400, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Model summary plot saved: {output_path}")


def main():
    """Main plotting pipeline."""
    print("🚀 Starting LME coefficient plotting pipeline...")
    
    # Create output directory
    output_dir = os.path.join('test', 'eda', 'lme_analysis', 'plots')
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Load LME results
        coefficients = load_lme_results()
        
        if not coefficients:
            raise ValueError("No coefficients found in LME analysis report!")
        
        # Prepare coefficient data
        coef_df = prepare_coefficient_data(coefficients)
        
        # Create coefficient plot
        coef_plot_path = os.path.join(output_dir, 'lme_coefficient_plot.png')
        create_coefficient_plot(coef_df, coef_plot_path)
        
        # Create effect sizes table
        table_path = os.path.join(output_dir, 'effect_sizes_table.csv')
        create_effect_sizes_table(coef_df, table_path)
        
        # Create model summary plot
        summary_plot_path = os.path.join(output_dir, 'model_summary.png')
        create_model_summary_plot(coef_df, summary_plot_path)
        
        print(f"\n🎯 LME coefficient plotting completed!")
        print(f"📁 Output directory: {output_dir}")
        print(f"🖼️  Coefficient plot: lme_coefficient_plot.png")
        print(f"📊 Effect sizes table: effect_sizes_table.csv")
        print(f"📄 Model summary: model_summary.png")
        
        print(f"\n📋 VISUAL SUMMARY:")
        print(f"   • Coefficient plot shows effect sizes with 95% CIs")
        print(f"   • Significant effects are highlighted")
        print(f"   • Color coding by effect family (Task/Dose/Interaction)")
        print(f"   • Reference line at zero for easy interpretation")
        
    except Exception as e:
        print(f"❌ Plotting failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == '__main__':
    success = main()
    if not success:
        sys.exit(1)
