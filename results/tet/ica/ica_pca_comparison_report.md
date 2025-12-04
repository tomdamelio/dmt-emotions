# ICA vs PCA Comparison Report

This report compares Independent Component Analysis (ICA) and Principal Component Analysis (PCA) results to assess whether ICA reveals experiential structure beyond the variance explained by principal components.

## 1. Executive Summary

**Key Findings:**
- Maximum IC-PC correlation: 0.315
- High correlations (|r| > 0.7): 0
- Low correlations (|r| < 0.3): 24
- Convergent LME effects: 1
- Divergent LME effects: 6

**Interpretation:** ICA components reveal distinct structure beyond PCA, suggesting that independence-based decomposition uncovers latent sources masked by the dominant variance structure.

## 2. Component Alignment

### IC-PC Correlation Matrix

| ic_component   |          PC1 |       PC2 |        PC3 |         PC4 |        PC5 |
|:---------------|-------------:|----------:|-----------:|------------:|-----------:|
| IC1            |  0.000126517 | -0.279014 | 0.19992    | -0.159535   |  0.0346668 |
| IC2            |  0.0391999   |  0.280402 | 0.00388091 | -0.00616009 |  0.315154  |
| IC3            | -0.183308    |  0.295326 | 0.21384    | -0.139699   | -0.174855  |
| IC4            | -0.153775    | -0.23825  | 0.167323   | -0.0527498  |  0.0143879 |
| IC5            |  0.230811    | -0.254382 | 0.181215   | -0.0258769  |  0.0194556 |

## 3. Pattern Comparison: Dimension Contributions

| ic_component   | pc_component   |   pattern_correlation | divergent_dimensions                                            |
|:---------------|:---------------|----------------------:|:----------------------------------------------------------------|
| IC1            | PC2            |             -0.937419 | pleasantness_z, unpleasantness_z, bliss_z, anxiety_z            |
| IC2            | PC5            |              0.323882 | emotional_intensity_z, interoception_z, bliss_z, anxiety_z      |
| IC3            | PC2            |              0.853289 | pleasantness_z, interoception_z                                 |
| IC4            | PC2            |             -0.174079 | pleasantness_z, emotional_intensity_z, interoception_z, bliss_z |
| IC5            | PC2            |             -0.923164 | pleasantness_z, unpleasantness_z, bliss_z, anxiety_z            |

## 4. LME Effect Comparison

### Convergent Findings (n=1)

| effect    | ic_component   | pc_component   |   ic_beta |   pc_beta |
|:----------|:---------------|:---------------|----------:|----------:|
| Group Var | IC1            | PC2            |  0.595207 |  0.559334 |

### Divergent Findings (n=6)

| effect                                                                     | ic_component   | pc_component   |      ic_beta |     pc_beta | agreement              |
|:---------------------------------------------------------------------------|:---------------|:---------------|-------------:|------------:|:-----------------------|
| Intercept                                                                  | IC1            | PC2            | -0.0863809   | -0.823212   | divergent_significance |
| C(state, Treatment('RS'))[T.DMT]                                           | IC1            | PC2            |  0.190256    |  0.0118673  | divergent_significance |
| C(dose, Treatment('Baja'))[T.Alta]                                         | IC1            | PC2            | -0.0846886   | -0.0782414  | divergent_significance |
| time_c                                                                     | IC1            | PC2            | -0.000476969 |  0.00540495 | divergent_significance |
| C(state, Treatment('RS'))[T.DMT]:time_c                                    | IC1            | PC2            | -0.000739709 |  0.023463   | divergent_significance |
| C(state, Treatment('RS'))[T.DMT]:C(dose, Treatment('Baja'))[T.Alta]:time_c | IC1            | PC2            | -0.000604494 | -0.0238918  | divergent_significance |

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
- 24 independent components with low PC correlation
- 6 divergent LME effects

Including ICA will provide complementary insights into the independent sources of experiential variation.
