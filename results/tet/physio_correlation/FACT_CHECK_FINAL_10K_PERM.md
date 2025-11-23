# Fact-Checking Final: CCA Results with 10,000 Permutations

**Date:** November 22, 2025  
**Analysis:** 10,000 permutations (publication-ready)  
**Status:** ✅ COMPLETE VERIFICATION

---

## 📊 Data Sources (10K Permutations)

From current analysis run:
- Permutation test: 10,000 iterations
- Cross-validation: 7 folds (LOSO)
- Loadings: From `cca_loadings.csv`

---

## ✅ VERIFIED CLAIMS

### 1. Resting State Failure to Generalize
**Manuscript:** "mean out-of-sample r = −.28"  
**Data:** RS CV1: mean_r_oos = -0.276  
**Status:** ✅ **CORRECT**

---

### 2. DMT In-Sample Correlation
**Manuscript:** "robs = .68"  
**Data:** DMT CV1: r_observed = 0.678  
**Status:** ✅ **CORRECT**

---

### 3. DMT Out-of-Sample Correlation
**Manuscript:** "roos = .49 (SD = .31)"  
**Data:** DMT CV1: mean_r_oos = 0.494, sd_r_oos = 0.306  
**Status:** ✅ **CORRECT**

---

### 4. Cross-Validation Significance
**Manuscript:** "pcv = .008"  
**Data:** DMT CV1: p_value_t_test = 0.008  
**Status:** ✅ **CORRECT**

---

### 5. Redundancy Analysis
**Manuscript:** "10.3% of the total variance in the physiological data"  
**Data:** DMT CV1: redundancy_X_given_Y = 0.103 (10.3%)  
**Status:** ✅ **CORRECT**

---

### 6. Physiological Loadings
**Manuscript Claims:**
- "respiratory volume (r = .85)"
- "heart rate (r = .66)"
- "sympathetic tone (r = .56)"

**Actual Data (DMT CV1):**
- RVT: **0.850** ✅
- HR: **0.655** ✅
- SMNA_AUC: **0.558** ✅

**Status:** ✅ **ALL CORRECT**

---

### 7. TET Affective Loadings
**Manuscript Claims:**
- "emotional intensity (r = .86)"
- "unpleasantness (r = .27)"
- "interoception (r = .28)"

**Actual Data (DMT CV1):**
- emotional_intensity: **0.857** ✅
- unpleasantness: **0.267** ✅
- interoception: **0.284** ✅

**Status:** ✅ **ALL CORRECT**

---

## 🚨 CRITICAL ISSUE: Permutation Test Result

### With 10,000 Permutations:

**DMT CV1 Permutation Test:**
- **p_perm = 0.162** (NOT significant at α = 0.05)

**Comparison with Previous Results:**
- 100 permutations: p = 0.119
- 1,000 permutations: p = 0.152
- 10,000 permutations: p = 0.162

**Trend:** The p-value is **increasing** with more permutations, stabilizing around p ≈ 0.16

---

## 📊 Complete Validation Summary (10K Permutations)

| Metric | Value | Significant? |
|--------|-------|--------------|
| In-sample r | 0.678 | ✅ Yes (Wilks' Lambda) |
| Permutation p | 0.162 | ❌ No (p > 0.05) |
| Mean r_oos | 0.494 | ✅ Moderate |
| CV significance p | 0.008 | ✅ Yes (p < 0.01) |
| Overfitting index | 0.271 | ✅ Acceptable (<0.3) |
| Redundancy | 10.3% | ✅ Meaningful (>10%) |
| Success rate | 100% | ✅ All folds positive |

---

## 🎯 INTERPRETATION

### The Discrepancy:

You have **two significance tests with conflicting results:**

1. **Permutation test (cross-subject coupling):** p = 0.162 (NOT significant)
   - Tests: Is there systematic coupling across subjects?
   - Answer: Borderline/weak evidence

2. **CV significance test (generalization):** p = 0.008 (SIGNIFICANT)
   - Tests: Does the model generalize to new subjects?
   - Answer: Yes, strong evidence

### Why the Discrepancy?

These tests answer **different questions:**

- **Permutation test:** Tests if the coupling exists at the population level (across all subjects simultaneously)
- **CV significance test:** Tests if the coupling pattern, once learned, generalizes to held-out subjects

**Possible explanation:** The physiological-affective coupling may be **subject-specific** (different patterns per subject) but **consistently present** (generalizes within-subject structure).

---

## 📝 RECOMMENDED MANUSCRIPT TEXT

### Current Version (Problematic):
Your text omits the permutation result entirely, which is **not acceptable** for peer review.

### Recommended Revision:

> "Canonical correlation analysis, validated via leave-one-subject-out cross-validation (LOOCV), identified a latent structure linking physiological signals with affective experience during the psychedelic state. In contrast to the resting state, where the model failed to generalize to new participants (mean out-of-sample r = −.28), the DMT state revealed a latent dimension coupling autonomic dynamics with subjective phenomenology.
> 
> The in-sample correlation for this first canonical variate was high (r_obs = .68). Crucially, the cross-validation procedure demonstrated robust generalization: the model successfully predicted the physiological–affective coupling of held-out participants, yielding a mean out-of-sample correlation of r_oos = .49 (SD = .31). Statistical testing of the cross-validation coefficients using Fisher Z-transformed correlations confirmed that this predictive capacity was significantly greater than chance (p_cv = .008, one-sample t-test, N = 7 folds). **Subject-level permutation testing (10,000 iterations) did not reach conventional significance (p_perm = .162), suggesting that while the coupling pattern generalizes within subjects, there may be heterogeneity in the specific physiological-affective mappings across individuals.** Redundancy analysis indicated that this affective dimension explained 10.3% of the total variance in the physiological data, consistent with a coherent psychophysiological synchronization mechanism.
> 
> Examination of the canonical loadings revealed that this latent dimension reflects a "general autonomic arousal" factor (Figure 4). On the physiological side, the variate was defined by strong positive loadings across all three markers: respiratory volume (r = .85), heart rate (r = .66), and sympathetic tone (r = .56). On the phenomenological side, this broad autonomic activation was coupled primarily with emotional intensity (r = .86), with moderate contributions from interoception (r = .28) and unpleasantness (r = .27). This pattern indicates that the subjective intensity of the emotional experience under DMT maps onto a coherent, systemic upregulation of cardiorespiratory and electrodermal activity."

---

## 🔬 Alternative Interpretation (More Conservative)

If you want to be more conservative given the non-significant permutation test:

> "...Statistical testing of the cross-validation coefficients confirmed significant generalization to held-out subjects (p_cv = .008). However, subject-level permutation testing (10,000 iterations) did not reach significance (p_perm = .162), indicating that **the evidence for robust cross-subject coupling should be interpreted with caution**. The significant cross-validation result suggests that the model captures meaningful within-subject structure that generalizes, but the non-significant permutation result indicates potential heterogeneity in physiological-affective mappings across individuals. Future studies with larger samples are needed to clarify whether this represents a universal coupling mechanism or subject-specific patterns."

---

## 📋 COMPLETE FACT-CHECK TABLE

| Claim | Manuscript | Actual | Status | Notes |
|-------|-----------|--------|--------|-------|
| RS r_oos | -0.28 | -0.276 | ✅ | Correct |
| DMT r_obs | 0.68 | 0.678 | ✅ | Correct |
| DMT r_oos | 0.49 | 0.494 | ✅ | Correct |
| DMT SD r_oos | 0.31 | 0.306 | ✅ | Correct |
| DMT p_cv | 0.008 | 0.008 | ✅ | Correct |
| **DMT p_perm** | **NOT MENTIONED** | **0.162** | ❌ | **MUST ADD** |
| Redundancy | 10.3% | 10.3% | ✅ | Correct |
| RVT loading | 0.85 | 0.850 | ✅ | Correct |
| HR loading | 0.66 | 0.655 | ✅ | Correct |
| SMNA loading | 0.56 | 0.558 | ✅ | Correct |
| Emotional intensity | 0.86 | 0.857 | ✅ | Correct |
| Unpleasantness | 0.27 | 0.267 | ✅ | Correct |
| Interoception | 0.28 | 0.284 | ✅ | Correct |

---

## 🎯 FINAL RECOMMENDATIONS

### 1. **MANDATORY: Address Permutation Result**
You **must** mention p_perm = 0.162 in the manuscript. Omitting it is cherry-picking.

### 2. **Interpretation Options:**

**Option A (Balanced):** Acknowledge both tests, explain the discrepancy, emphasize CV significance.

**Option B (Conservative):** Downgrade claim from "robust" to "promising" or "preliminary evidence."

**Option C (Mechanistic):** Explain that CV tests generalization (within-subject patterns) while permutation tests population-level coupling (across-subject consistency).

### 3. **Supplementary Material:**
Add a supplementary table with all validation metrics to show transparency.

### 4. **Figure 4:**
The figure is now ready in Nature Human Behaviour format:
- `Figure4_CCA_Loadings_DMT_CV1.png` (600 DPI)
- `Figure4_CCA_Loadings_DMT_CV1.pdf` (vector)
- `Figure4_CCA_Loadings_DMT_CV1.tiff` (600 DPI)

---

## ✅ FINAL VERDICT

**Numerical Accuracy:** 100% (12/12 values correct)

**Statistical Interpretation:** ⚠️ **NEEDS REVISION**
- The manuscript overstates the evidence by omitting the non-significant permutation test
- The CV significance (p = 0.008) is real and important, but must be contextualized with the permutation result

**Recommendation:** Use the revised text above that acknowledges both tests and provides a balanced interpretation.

---

**Fact-Check Completed:** November 22, 2025  
**Verified By:** Kiro AI with 10,000 permutation analysis  
**Confidence:** 100% (all values cross-referenced with source data)
