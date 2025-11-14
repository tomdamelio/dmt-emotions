"""
Figure Caption Generator for TET Analysis

This module generates descriptive captions for TET analysis figures.
Captions are saved as text files in results/tet/figures/captions/
"""

from pathlib import Path
from typing import Dict, Optional

# Caption templates for different figure types
CAPTION_TEMPLATES = {
    'timeseries_all_dimensions': """
Figure: Temporal Dynamics of Subjective Experience Dimensions

Time series plots showing mean trajectories (±SEM) for all 15 TET dimensions across 
DMT sessions. Blue lines represent Low dose (20mg), red lines represent High dose (40mg).
Grey dashed vertical line indicates DMT onset (end of resting state baseline). Grey 
background shading indicates time bins where DMT significantly differs from resting 
state baseline (p < .05, FDR-corrected). Black horizontal bars indicate time bins 
with significant State×Dose interaction effects (p < .05, FDR-corrected). Dimensions 
are ordered by strength of main State effect from linear mixed effects models.

Statistical Analysis: Linear mixed effects models with random subject intercepts.
Multiple comparison correction: Benjamini-Hochberg FDR.
Time window: 0-20 minutes for DMT, 0-10 minutes for resting state.
""",
    
    'lme_coefficients_forest': """
Figure: Linear Mixed Effects Model Coefficients

Forest plots showing standardized regression coefficients (β) with 95% confidence 
intervals for fixed effects in linear mixed effects models. Separate panels show 
State effects (DMT vs RS), Dose effects (High vs Low), and State×Dose interaction 
effects. Filled circles indicate statistically significant effects (p_fdr < .05), 
open circles indicate non-significant effects. Vertical line at β=0 represents null 
effect. Dimensions are ordered by magnitude of State main effect.

Statistical Analysis: Linear mixed effects models with formula:
  Dimension ~ State*Dose + Time_c + State:Time_c + Dose:Time_c + (1|Subject)
Multiple comparison correction: Benjamini-Hochberg FDR across 15 dimensions.
""",
    
    'peak_dose_comparison': """
Figure: Peak Intensity Comparison Between Dose Conditions

Boxplots comparing peak z-scored intensity values between Low dose (20mg, blue) and 
High dose (40mg, red) conditions within DMT sessions. Grey lines connect paired 
observations from the same subject. Significance stars indicate FDR-corrected p-values 
from Wilcoxon signed-rank tests: * p < .05, ** p < .01, *** p < .001. Effect sizes 
(r) with 95% bootstrap confidence intervals are displayed for significant comparisons.

Statistical Analysis: Wilcoxon signed-rank tests (paired, two-tailed).
Effect size: Rank-biserial correlation (r) with 2000 bootstrap iterations.
Multiple comparison correction: Benjamini-Hochberg FDR across 15 dimensions.
""",
    
    'time_to_peak_dose_comparison': """
Figure: Time to Peak Comparison Between Dose Conditions

Boxplots comparing time to peak intensity (in minutes) between Low dose (20mg, blue) 
and High dose (40mg, red) conditions within DMT sessions. Grey lines connect paired 
observations from the same subject. Significance stars indicate FDR-corrected p-values 
from Wilcoxon signed-rank tests: * p < .05, ** p < .01, *** p < .001. Effect sizes 
(r) with 95% bootstrap confidence intervals are displayed for significant comparisons.

Statistical Analysis: Wilcoxon signed-rank tests (paired, two-tailed).
Effect size: Rank-biserial correlation (r) with 2000 bootstrap iterations.
Multiple comparison correction: Benjamini-Hochberg FDR across 15 dimensions.
""",
    
    'auc_dose_comparison': """
Figure: Area Under Curve Comparison Between Dose Conditions

Boxplots comparing area under the curve (AUC, 0-9 minutes) between Low dose (20mg, blue) 
and High dose (40mg, red) conditions within DMT sessions. AUC represents cumulative 
intensity over time (z-score × minutes). Grey lines connect paired observations from 
the same subject. Significance stars indicate FDR-corrected p-values from Wilcoxon 
signed-rank tests: * p < .05, ** p < .01, *** p < .001. Effect sizes (r) with 95% 
bootstrap confidence intervals are displayed for significant comparisons.

Statistical Analysis: Wilcoxon signed-rank tests (paired, two-tailed).
Effect size: Rank-biserial correlation (r) with 2000 bootstrap iterations.
Multiple comparison correction: Benjamini-Hochberg FDR across 15 dimensions.
""",
    
    'clustering_kmeans_centroids': """
Figure: KMeans Cluster Centroid Profiles

Bar plots showing normalized dimension contributions for each cluster identified by 
KMeans clustering (k=2 solution). Each centroid is normalized by its maximum dimension 
value to show relative contributions on a 0-1 scale. Elevated dimensions (positive 
contributions) and suppressed dimensions (negative contributions) characterize each 
cluster's experiential profile.

Statistical Analysis: KMeans clustering on z-scored dimension values.
Stability assessment: Bootstrap resampling with 1000 iterations (Adjusted Rand Index).
Model selection: Silhouette score and within-cluster sum of squares.
""",
    
    'clustering_kmeans_prob_timecourses': """
Figure: Cluster Probability Time Courses

Time series plots showing mean cluster probability (±SEM) as a function of time for 
each cluster state. Separate panels show High dose (40mg) and Low dose (20mg) conditions 
within DMT sessions. Probabilities represent the likelihood of each time point belonging 
to each cluster based on soft KMeans assignments. Temporal dynamics reveal how cluster 
prevalence changes throughout the DMT experience.

Statistical Analysis: Soft KMeans clustering probabilities averaged across subjects.
Time window: 0-20 minutes for DMT sessions.
Comparison: Resting state sessions included as baseline reference.
"""
}


def generate_caption(figure_name: str, custom_caption: Optional[str] = None) -> str:
    """
    Generate caption for a figure.
    
    Args:
        figure_name: Name of the figure (without extension)
        custom_caption: Optional custom caption text
        
    Returns:
        Caption text
    """
    if custom_caption:
        return custom_caption
    
    # Remove file extension if present
    figure_name = Path(figure_name).stem
    
    # Get template caption
    caption = CAPTION_TEMPLATES.get(figure_name)
    
    if caption:
        return caption.strip()
    else:
        # Generate generic caption
        return f"""
Figure: {figure_name.replace('_', ' ').title()}

[Caption to be added]

Statistical details and interpretation should be added based on the specific 
analysis and results shown in this figure.
""".strip()


def save_caption(figure_path: str, caption: Optional[str] = None, 
                captions_dir: str = 'results/tet/figures/captions') -> Path:
    """
    Save caption for a figure.
    
    Args:
        figure_path: Path to the figure file
        caption: Optional custom caption (if None, uses template)
        captions_dir: Directory to save captions
        
    Returns:
        Path to saved caption file
    """
    figure_path = Path(figure_path)
    captions_dir = Path(captions_dir)
    
    # Create captions directory if needed
    captions_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate caption
    caption_text = generate_caption(figure_path.stem, caption)
    
    # Save caption
    caption_file = captions_dir / f"{figure_path.stem}.txt"
    with open(caption_file, 'w', encoding='utf-8') as f:
        f.write(caption_text)
    
    return caption_file


def generate_all_captions(figures_dir: str = 'results/tet/figures',
                         captions_dir: str = 'results/tet/figures/captions') -> Dict[str, Path]:
    """
    Generate captions for all figures in directory.
    
    Args:
        figures_dir: Directory containing figures
        captions_dir: Directory to save captions
        
    Returns:
        Dictionary mapping figure names to caption file paths
    """
    figures_dir = Path(figures_dir)
    caption_files = {}
    
    if not figures_dir.exists():
        return caption_files
    
    # Find all PNG files
    for figure_file in figures_dir.glob('*.png'):
        caption_file = save_caption(figure_file, captions_dir=captions_dir)
        caption_files[figure_file.stem] = caption_file
    
    return caption_files


if __name__ == '__main__':
    # Generate captions for all existing figures
    caption_files = generate_all_captions()
    
    print(f"Generated {len(caption_files)} caption files:")
    for figure_name, caption_path in caption_files.items():
        print(f"  {figure_name} -> {caption_path}")
