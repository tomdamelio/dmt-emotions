import pandas as pd

df = pd.read_csv('test/resp/validation_results/validation_summary.csv')
problematic = df[df['subject'].isin(['S09', 'S17', 'S19'])]

print('\n=== ANÁLISIS DETALLADO DE SUJETOS PROBLEMÁTICOS ===\n')

for subj in ['S09', 'S17', 'S19']:
    print(f'\n{subj}:')
    subj_data = problematic[problematic['subject'] == subj]
    
    for _, row in subj_data.iterrows():
        print(f"  {row['condition']:12s}: {row['pct_valid']:5.1f}% válido, "
              f"Rate={row['mean_rate']:5.1f}±{row['std_rate']:4.1f} rpm, "
              f"Peaks={int(row['peaks_n_peaks']):3d}, "
              f"NaN={row['artifact_pct_nan_rate']:5.2f}%")
    
    print(f"  {'─'*60}")
    print(f"  Media válido: {subj_data['pct_valid'].mean():.1f}%")
    worst_idx = subj_data['pct_valid'].idxmin()
    best_idx = subj_data['pct_valid'].idxmax()
    print(f"  Peor condición: {subj_data.loc[worst_idx, 'condition']} ({subj_data['pct_valid'].min():.1f}%)")
    print(f"  Mejor condición: {subj_data.loc[best_idx, 'condition']} ({subj_data['pct_valid'].max():.1f}%)")
    print(f"  Rango de frecuencias: {subj_data['mean_rate'].min():.1f} - {subj_data['mean_rate'].max():.1f} rpm")
    print(f"  Variabilidad entre condiciones: {subj_data['pct_valid'].std():.1f}%")

print('\n\n=== RANKING DE PROBLEMAS (de peor a mejor) ===\n')

# Calcular score de problemas por sujeto
summary = []
for subj in ['S09', 'S17', 'S19']:
    subj_data = problematic[problematic['subject'] == subj]
    summary.append({
        'subject': subj,
        'mean_pct_valid': subj_data['pct_valid'].mean(),
        'min_pct_valid': subj_data['pct_valid'].min(),
        'n_conditions_below_70': (subj_data['pct_valid'] < 70).sum(),
        'n_conditions_below_90': (subj_data['pct_valid'] < 90).sum(),
        'rate_range': subj_data['mean_rate'].max() - subj_data['mean_rate'].min(),
        'variability': subj_data['pct_valid'].std(),
    })

summary_df = pd.DataFrame(summary).sort_values('mean_pct_valid')

for _, row in summary_df.iterrows():
    print(f"{row['subject']}:")
    print(f"  Media válido: {row['mean_pct_valid']:.1f}%")
    print(f"  Mínimo válido: {row['min_pct_valid']:.1f}%")
    print(f"  Condiciones < 70%: {int(row['n_conditions_below_70'])}/4")
    print(f"  Condiciones < 90%: {int(row['n_conditions_below_90'])}/4")
    print(f"  Rango de frecuencias: {row['rate_range']:.1f} rpm")
    print(f"  Variabilidad: {row['variability']:.1f}%")
    print()

print('\n=== RECOMENDACIÓN ===\n')
print(f"Los 2 sujetos MÁS PROBLEMÁTICOS a considerar para exclusión son:")
print(f"  1. {summary_df.iloc[0]['subject']} (media: {summary_df.iloc[0]['mean_pct_valid']:.1f}% válido)")
print(f"  2. {summary_df.iloc[1]['subject']} (media: {summary_df.iloc[1]['mean_pct_valid']:.1f}% válido)")
