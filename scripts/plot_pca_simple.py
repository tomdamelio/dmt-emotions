import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os

print("="*80)
print("TET PCA VISUALIZATION")
print("="*80)

# Load data
variance_df = pd.read_csv('results/tet/pca/pca_variance_explained.csv')
loadings_df = pd.read_csv('results/tet/pca/pca_loadings.csv')

print(f"Loaded variance: {len(variance_df)} components")
print(f"Loaded loadings: {len(loadings_df)} rows")

# Create scree plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.bar(variance_df['component'], variance_df['variance_explained'] * 100, 
        color='steelblue', alpha=0.8, edgecolor='black')
ax1.set_xlabel('Principal Component', fontsize=12, fontweight='bold')
ax1.set_ylabel('Variance Explained (%)', fontsize=12, fontweight='bold')
ax1.set_title('Variance Explained by Each Component', fontsize=14, fontweight='bold')
ax1.grid(axis='y', alpha=0.3)

for i, var in enumerate(variance_df['variance_explained']):
    ax1.text(i, var * 100 + 1, f'{var*100:.1f}%', ha='center', va='bottom', fontsize=10)

ax2.plot(range(len(variance_df)), variance_df['cumulative_variance'] * 100, 
        marker='o', linewidth=2, markersize=8, color='darkred')
ax2.axhline(y=75, color='gray', linestyle='--', linewidth=1.5, label='75% threshold')
ax2.fill_between(range(len(variance_df)), 0, variance_df['cumulative_variance'] * 100, 
                 alpha=0.2, color='darkred')
ax2.set_xlabel('Number of Components', fontsize=12, fontweight='bold')
ax2.set_ylabel('Cumulative Variance (%)', fontsize=12, fontweight='bold')
ax2.set_title('Cumulative Variance Explained', fontsize=14, fontweight='bold')
ax2.set_xticks(range(len(variance_df)))
ax2.set_xticklabels(variance_df['component'])
ax2.grid(alpha=0.3)
ax2.legend()

plt.tight_layout()
plt.savefig('results/tet/figures/pca_scree_plot.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: pca_scree_plot.png")

# Create loadings heatmap
loadings_wide = loadings_df.pivot(index='dimension', columns='component', values='loading')
components = ['PC1', 'PC2', 'PC3', 'PC4', 'PC5']
loadings_wide = loadings_wide[components]
loadings_wide.index = [d.replace('_z', '').replace('_', ' ').title() for d in loadings_wide.index]

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(loadings_wide, cmap='RdBu_r', center=0, annot=True, fmt='.2f', 
            cbar_kws={'label': 'Loading'}, linewidths=0.5, vmin=-1, vmax=1, ax=ax)
ax.set_xlabel('Principal Component', fontsize=12, fontweight='bold')
ax.set_ylabel('TET Dimension', fontsize=12, fontweight='bold')
ax.set_title('PCA Loadings Heatmap', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('results/tet/figures/pca_loadings_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: pca_loadings_heatmap.png")

print("="*80)
print("PCA VISUALIZATION COMPLETE")
print("="*80)
