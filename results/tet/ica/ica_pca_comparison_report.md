# ICA vs PCA Comparison Report

This report compares Independent Component Analysis (ICA) and Principal Component Analysis (PCA) results to assess whether ICA reveals experiential structure beyond the variance explained by principal components.

## 1. Executive Summary

**Key Findings:**
- Maximum IC-PC correlation: 0.348
- High correlations (|r| > 0.7): 0
- Low correlations (|r| < 0.3): 33
- Convergent LME effects: 0
- Divergent LME effects: 0

**Interpretation:** ICA components reveal distinct structure beyond PCA, suggesting that independence-based decomposition uncovers latent sources masked by the dominant variance structure.

## 2. Component Alignment

### IC-PC Correlation Matrix

| ic_component   |        PC1 |        PC2 |        PC3 |        PC4 |         PC5 |         PC6 |
|:---------------|-----------:|-----------:|-----------:|-----------:|------------:|------------:|
| IC1            |  0.14061   | -0.20585   | -0.327275  |  0.142358  |  0.173962   | -0.133919   |
| IC2            | -0.0775793 | -0.0663459 | -0.107638  |  0.0793998 | -0.283596   | -0.00127228 |
| IC3            |  0.227038  | -0.348078  |  0.10048   | -0.0331595 |  0.00908307 | -0.023499   |
| IC4            | -0.1431    | -0.343868  |  0.0947626 | -0.0700627 |  0.0759191  | -0.0452454  |
| IC5            | -0.139445  | -0.0251316 | -0.122251  | -0.109084  | -0.0254507  |  0.00872679 |
| IC6            | -0.0326965 |  0.274691  | -0.256168  |  0.0593487 |  0.198362   |  0.025427   |

## 3. Pattern Comparison: Dimension Contributions

| ic_component   | pc_component   |   pattern_correlation | divergent_dimensions                                                |
|:---------------|:---------------|----------------------:|:--------------------------------------------------------------------|
| IC1            | PC3            |           -0.567897   | unpleasantness_z, interoception_z, bliss_z, anxiety_z               |
| IC2            | PC5            |           -0.571137   | pleasantness_z, anxiety_z                                           |
| IC3            | PC2            |           -0.927559   | pleasantness_z, unpleasantness_z, bliss_z, anxiety_z                |
| IC4            | PC2            |           -0.522713   | pleasantness_z, emotional_intensity_z, interoception_z, bliss_z     |
| IC5            | PC1            |           -0.00572912 | unpleasantness_z, emotional_intensity_z, interoception_z, anxiety_z |
| IC6            | PC2            |            0.908252   | emotional_intensity_z, bliss_z                                      |

## 4. LME Effect Comparison

LME comparison not available (no comparable effects found).

## 5. Interpretation Guidelines

### When to use ICA vs PCA

- **PCA**: Best for identifying sources of maximum variance. Use when interested in dominant patterns.
- **ICA**: Best for identifying statistically independent sources. Use when interested in separating mixed signals.

### Interpreting Correlations

- **High correlation (|r| > 0.7)**: IC and PC capture similar structure
- **Moderate correlation (0.3 < |r| < 0.7)**: Partial overlap
- **Low correlation (|r| < 0.3)**: IC reveals distinct structure

## 6. Recommendations

**Recommendation: Include ICA in main analysis**

ICA reveals experiential structure beyond PCA, particularly:
- 33 independent components with low PC correlation

Including ICA will provide complementary insights into the independent sources of experiential variation.
