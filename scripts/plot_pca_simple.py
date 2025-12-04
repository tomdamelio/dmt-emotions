import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os

# Nature Human Behaviour style configuration
AXES_TITLE_SIZE = 24
AXES_LABEL_SIZE = 22
TICK_LABEL_SIZE = 18
LEGEND_FONTSIZE = 18

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'figure.dpi': 110,
    'savefig.dpi': 400,
    'axes.titlesize': AXES_TITLE_SIZE,
    'axes.labelsize': AXES_LABEL_SIZE,
    'axes.linewidth': 1.5,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'legend.frameon': False,
    'legend.fontsize': LEGEND_FONTSIZE,
    'xtick.labelsize': TICK_LABEL_SIZE,
    'ytick.labelsize': TICK_LABEL_SIZE,
})

# TET uses purple/violet color scheme from tab20c palette
# tab20c has 20 colors in 5 groups of 4 gradients each
# Purple group: indices 12-15 (darkest to lightest)
tab20c_colors = plt.cm.tab20c.colors
COLOR_PRIMARY = tab20c_colors[12]  # Darkest purple for primary elements
COLOR_SECONDARY = tab20c_colors[13]  # Medium purple
COLOR_TERTIARY = tab20c_colors[14]  # Lighter purple

print("="*80)
print("TET PCA VISUALIZATION")
print("="*80)

# Load data
variance_df = pd.read_csv('results/tet/pca/pca_variance_explained.csv')
loadings_df = pd.read_csv('results/tet/pca/pca_loadings.csv')

print(f"Loaded variance: {len(variance_df)} components")
print(f"Loaded loadings: {len(loadings_df)} rows")

# Prepare loadings data for heatmap
loadings_wide = loadings_df.pivot(index='dimension', columns='component', values='loading')
# Use only available components (don't hardcode PC4, PC5 if they don't exist)
available_components = sorted([col for col in loadings_wide.columns if col.startswith('PC')])
loadings_wide = loadings_wide[available_components]
loadings_wide.index = [d.replace('_z', '').replace('_', ' ').title() for d in loadings_wide.index]

print(f"Available components: {available_components}")

# Create scree plot with only variance explained (single panel)
# Height set to match pca_loadings_heatmap.png
SCREE_HEIGHT = 5
fig, ax1 = plt.subplots(1, 1, figsize=(8, SCREE_HEIGHT))

# Panel 1: Variance Explained
# Use violet for PC1-PC2, grey for PC3-PC5
bar_colors = []
text_colors = []
for i in range(len(variance_df)):
    if i < 2:
        # PC1-PC2: violet (darkest purple from tab20c)
        bar_colors.append(COLOR_PRIMARY)
        text_colors.append(COLOR_PRIMARY)
    else:
        # PC3-PC5: grey
        bar_colors.append('#808080')  # Medium grey
        text_colors.append('#808080')

ax1.bar(variance_df['component'], variance_df['variance_explained'] * 100, 
        color=bar_colors, alpha=0.9, edgecolor='none')
ax1.set_xlabel('Principal Component', fontweight='bold')
ax1.set_ylabel('Variance Explained (%)', fontweight='bold')
# Eliminar cuadrícula del fondo
ax1.grid(False)
ax1.set_axisbelow(True)
# Eliminar bordes superior y derecho
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

for i, var in enumerate(variance_df['variance_explained']):
    # Use violet for PC1-PC2, grey for PC3-PC5
    text_color = text_colors[i]
    ax1.text(i, var * 100 + 0.5, f'{var*100:.1f}%', ha='center', va='bottom', 
             fontsize=TICK_LABEL_SIZE-4, fontweight='bold', color=text_color)

plt.tight_layout()
plt.savefig('results/tet/figures/pca_scree_plot.png', dpi=400, bbox_inches='tight')
plt.close()
print("Saved: pca_scree_plot.png (variance explained only, 50% shorter)")

# Create separate loadings heatmap (standalone figure) - Only PC1 and PC2
# TRANSPOSED: Components on Y-axis (vertical), Dimensions on X-axis (horizontal)
# This creates a wider-than-tall figure
loadings_wide_full = loadings_df.pivot(index='dimension', columns='component', values='loading')
# Use only PC1 and PC2 for standalone heatmap
components_standalone = [c for c in ['PC1', 'PC2'] if c in available_components]
loadings_wide_full = loadings_wide_full[components_standalone]
loadings_wide_full.index = [d.replace('_z', '').replace('_', ' ').title() for d in loadings_wide_full.index]

# Transpose: now rows=components, columns=dimensions
loadings_transposed = loadings_wide_full.T

# Horizontal layout: wider than tall, height matches pca_scree_plot.png (SCREE_HEIGHT)
# Increased width to give more space for dimension labels
fig, ax = plt.subplots(figsize=(14, SCREE_HEIGHT))

# Create custom divergent colormap: Black (negative) -> White (center) -> Violet (positive)
from matplotlib.colors import LinearSegmentedColormap

# Use black for negative and darkest violet from tab20c (index 12) for positive
black = 'black'  # Pure black for negative loadings
violet_dark = tab20c_colors[12]  # (0.459, 0.420, 0.694)

# Create custom colormap
colors_custom = [black, 'white', violet_dark]
n_bins = 256
cmap_custom = LinearSegmentedColormap.from_list('black_violet', colors_custom, N=n_bins)

# Larger annotation font and larger colorbar
sns.heatmap(loadings_transposed, cmap=cmap_custom, center=0, annot=True, fmt='.2f', 
            cbar_kws={'label': 'Loading', 'shrink': 0.85, 'aspect': 25, 'pad': 0.02}, 
            linewidths=1.5, linecolor='white', vmin=-1, vmax=1, ax=ax,
            annot_kws={'fontsize': TICK_LABEL_SIZE, 'fontweight': 'bold'},
            square=False)
ax.set_xlabel('Affective Dimension', fontweight='bold', fontsize=AXES_LABEL_SIZE)
ax.tick_params(labelsize=TICK_LABEL_SIZE)
# Rotate x-axis labels for better readability
plt.xticks(rotation=30, ha='right')
# Rotate y-axis labels (PC1, PC2) to horizontal (90 degrees to the right)
plt.yticks(rotation=0)
# Eliminar bordes superior y derecho
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Remove ylabel completely (PC1, PC2 labels are already on the y-axis ticks)
ax.set_ylabel('', fontweight='bold', fontsize=AXES_LABEL_SIZE)

plt.tight_layout()
plt.savefig('results/tet/figures/pca_loadings_heatmap.png', dpi=400, bbox_inches='tight')
plt.close()
print("Saved: pca_loadings_heatmap.png (horizontal layout, components vertical)")

print("="*80)
print("PCA VISUALIZATION COMPLETE")
print("="*80)
