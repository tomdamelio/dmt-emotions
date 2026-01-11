# Supplementary Methods

## Statistical Analyses

### Time-to-Time Comparisons with Directional Hypotheses

For the primary analysis of dose-dependent effects during DMT administration, we employed one-tailed paired *t*-tests at each 30-second time window, with the directional hypothesis that High dose (40 mg) would produce greater physiological responses than Low dose (20 mg). This approach was justified by pharmacological expectations that higher doses of DMT induce stronger autonomic activation. For Resting State conditions, where no directional hypothesis was specified, we used two-tailed tests. All *p*-values were corrected for multiple comparisons using the Benjamini-Hochberg false discovery rate (FDR) procedure (Benjamini & Hochberg, 1995), with α = 0.05.

### Phase-Based Temporal Analysis

To address potential temporal misalignment in pointwise comparisons, we divided the 9-minute analysis window into two temporal phases: onset (0–3 min) and recovery (3–9 min). These boundaries were selected based on the pharmacokinetic profile of inhaled DMT, which exhibits rapid onset and gradual recovery (Strassman et al., 1994). For each subject and dose condition, we computed the mean physiological response within each phase. We then performed paired *t*-tests comparing High versus Low doses within each phase separately, using two-tailed tests with α = 0.05. This phase-averaging approach reduces noise from temporal variability while capturing overall trajectory differences between doses.

### Temporal Feature Extraction

We extracted four temporal features from each subject's physiological time series to characterize response magnitude and timing:

1. **Peak amplitude**: Maximum value within the 9-minute window, reflecting the magnitude of physiological response (analogous to *C*<sub>max</sub> in pharmacokinetics).

2. **Time-to-peak**: Time from onset to maximum value (in minutes), reflecting the speed of response onset (analogous to *T*<sub>max</sub>).

3. **Threshold crossings**: Times when the signal first crossed 33% and 50% of maximum amplitude, characterizing the dose-response relationship and temporal dynamics.

For each feature, we performed paired *t*-tests comparing High versus Low doses within the DMT condition, using two-tailed tests with α = 0.05. Effect sizes were quantified using Cohen's *d* for paired samples, calculated as *d* = *M*<sub>diff</sub> / *SD*<sub>diff</sub>, where *M*<sub>diff</sub> is the mean difference and *SD*<sub>diff</sub> is the standard deviation of differences (Cohen, 1988).

### Baseline Comparisons

To quantify the overall magnitude of DMT-induced changes independent of dose comparisons, we compared extracted temporal features between DMT conditions (High and Low doses collapsed) and Resting State baseline. For each subject, we computed the mean feature value across both DMT doses, then performed paired *t*-tests comparing these DMT means to Resting State values. This analysis used two-tailed tests with α = 0.05. Note that these baseline comparisons do not address dose-dependent effects; they quantify the general magnitude of drug-induced changes relative to baseline.

## Physiological Signal Processing

### Electrocardiography (Heart Rate)

Heart rate (HR) was derived from electrocardiogram (ECG) recordings using NeuroKit2 (Makowski et al., 2021). ECG signals were bandpass filtered (0.5–40 Hz), and R-peaks were detected using the Pan-Tompkins algorithm (Pan & Tompkins, 1985). Instantaneous HR was computed from inter-beat intervals and resampled to 4 Hz. For each subject, HR was z-scored using the combined baseline from all four sessions (Resting State High, Resting State Low, DMT High, DMT Low) to account for between-subject variability in baseline HR. This within-subject normalization approach ensures that dose comparisons reflect relative changes rather than absolute differences.

### Electrodermal Activity (Sympathetic Nervous System Activity)

Electrodermal activity (EDA) was decomposed into tonic (skin conductance level, SCL) and phasic (skin conductance responses, SCR) components using convex optimization (cvxEDA; Greco et al., 2016). We quantified sympathetic nervous system activity (SMNA) by computing the area under the curve (AUC) of the phasic component within each 30-second window. This metric reflects the integrated sympathetic drive over time. EDA signals were z-scored within-subject using the same four-session baseline approach as HR.

### Respiration (Respiratory Volume per Time)

Respiratory volume per time (RVT) was computed from thoracic effort belt recordings using NeuroKit2. Respiration signals were bandpass filtered (0.1–0.5 Hz), and peaks and troughs were detected to identify respiratory cycles. RVT was calculated as the ratio of tidal volume (peak-to-trough amplitude) to breath duration, providing a measure of respiratory drive (Birn et al., 2006). RVT was z-scored within-subject using the four-session baseline.

## Software and Statistical Packages

All analyses were performed in Python 3.11 using the following packages: NumPy 1.24 (Harris et al., 2020), pandas 2.0 (McKinney, 2010), SciPy 1.11 (Virtanen et al., 2020), statsmodels 0.14 (Seabold & Perktold, 2010), scikit-learn 1.3 (Pedregosa et al., 2011), Matplotlib 3.7 (Hunter, 2007), and seaborn 0.12 (Waskom, 2021). NeuroKit2 0.2.7 was used for physiological signal processing (Makowski et al., 2021). Custom analysis scripts are available at [repository URL].

## Data Availability

Preprocessed physiological data and analysis scripts are available at [repository URL]. Raw physiological recordings are available upon reasonable request to the corresponding author, subject to ethical approval and data sharing agreements.

## Code Availability

All analysis code, including the supplementary analyses pipeline (`run_supplementary_analyses.py`), is publicly available at [repository URL] under an open-source license. The repository includes complete documentation, example data, and instructions for reproducing all analyses.

---

## References

Benjamini, Y., & Hochberg, Y. (1995). Controlling the false discovery rate: A practical and powerful approach to multiple testing. *Journal of the Royal Statistical Society: Series B (Methodological)*, *57*(1), 289–300. https://doi.org/10.1111/j.2517-6161.1995.tb02031.x

Birn, R. M., Diamond, J. B., Smith, M. A., & Bandettini, P. A. (2006). Separating respiratory-variation-related fluctuations from neuronal-activity-related fluctuations in fMRI. *NeuroImage*, *31*(4), 1536–1548. https://doi.org/10.1016/j.neuroimage.2006.02.048

Cohen, J. (1988). *Statistical power analysis for the behavioral sciences* (2nd ed.). Lawrence Erlbaum Associates.

Greco, A., Valenza, G., Lanata, A., Scilingo, E. P., & Citi, L. (2016). cvxEDA: A convex optimization approach to electrodermal activity processing. *IEEE Transactions on Biomedical Engineering*, *63*(4), 797–804. https://doi.org/10.1109/TBME.2015.2474131

Harris, C. R., Millman, K. J., van der Walt, S. J., Gommers, R., Virtanen, P., Cournapeau, D., Wieser, E., Taylor, J., Berg, S., Smith, N. J., Kern, R., Picus, M., Hoyer, S., van Kerkwijk, M. H., Brett, M., Haldane, A., del Río, J. F., Wiebe, M., Peterson, P., ... Oliphant, T. E. (2020). Array programming with NumPy. *Nature*, *585*(7825), 357–362. https://doi.org/10.1038/s41586-020-2649-2

Hunter, J. D. (2007). Matplotlib: A 2D graphics environment. *Computing in Science & Engineering*, *9*(3), 90–95. https://doi.org/10.1109/MCSE.2007.55

Makowski, D., Pham, T., Lau, Z. J., Brammer, J. C., Lespinasse, F., Pham, H., Schölzel, C., & Chen, S. H. A. (2021). NeuroKit2: A Python toolbox for neurophysiological signal processing. *Behavior Research Methods*, *53*(4), 1689–1696. https://doi.org/10.3758/s13428-020-01516-y

McKinney, W. (2010). Data structures for statistical computing in Python. In S. van der Walt & J. Millman (Eds.), *Proceedings of the 9th Python in Science Conference* (pp. 56–61). https://doi.org/10.25080/Majora-92bf1922-00a

Pan, J., & Tompkins, W. J. (1985). A real-time QRS detection algorithm. *IEEE Transactions on Biomedical Engineering*, *BME-32*(3), 230–236. https://doi.org/10.1109/TBME.1985.325532

Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M., & Duchesnay, É. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*, *12*, 2825–2830.

Seabold, S., & Perktold, J. (2010). Statsmodels: Econometric and statistical modeling with Python. In S. van der Walt & J. Millman (Eds.), *Proceedings of the 9th Python in Science Conference* (pp. 92–96). https://doi.org/10.25080/Majora-92bf1922-011

Strassman, R. J., Qualls, C. R., Uhlenhuth, E. H., & Kellner, R. (1994). Dose-response study of N,N-dimethyltryptamine in humans: II. Subjective effects and preliminary results of a new rating scale. *Archives of General Psychiatry*, *51*(2), 98–108. https://doi.org/10.1001/archpsyc.1994.03950020022002

Virtanen, P., Gommers, R., Oliphant, T. E., Haberland, M., Reddy, T., Cournapeau, D., Burovski, E., Peterson, P., Weckesser, W., Bright, J., van der Walt, S. J., Brett, M., Wilson, J., Millman, K. J., Mayorov, N., Nelson, A. R. J., Jones, E., Kern, R., Larson, E., ... SciPy 1.0 Contributors. (2020). SciPy 1.0: Fundamental algorithms for scientific computing in Python. *Nature Methods*, *17*(3), 261–272. https://doi.org/10.1038/s41592-019-0686-2

Waskom, M. L. (2021). seaborn: Statistical data visualization. *Journal of Open Source Software*, *6*(60), 3021. https://doi.org/10.21105/joss.03021
