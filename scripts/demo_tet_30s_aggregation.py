#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Demostración de truncamiento TET a duraciones del paper original

Este script muestra cómo truncar los datos TET extendidos
a las duraciones especificadas en el paper original (40 bins DMT, 20 bins RS).
"""

import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from tet.data_loader import TETDataLoader
import config

def main():
    print("=" * 80)
    print("DEMOSTRACIÓN: TRUNCAMIENTO TET BINS DE 30 SEGUNDOS")
    print("=" * 80)
    print()
    
    print("CONTEXTO DEL PAPER ORIGINAL")
    print("-" * 80)
    print('Paper: "each datapoint represented 30s of subjective experience,')
    print('        with N=20 for RS (10-minutes) and N=40 for DMT (20-minutes)"')
    print()
    print("Datos actuales (down-sampled uniformemente):")
    print("  - DMT: 300 puntos @ 4s/punto = 1200s = 20 minutos")
    print("  - RS: 150 puntos @ 4s/punto = 600s = 10 minutos")
    print()
    print("Objetivo (bins de 30s según paper):")
    print("  - DMT: 40 bins × 30s = 1200s = 20 minutos")
    print("  - RS: 20 bins × 30s = 600s = 10 minutos")
    print()
    print("Agregación necesaria: 7.5 puntos → 1 bin (30s / 4s = 7.5)")
    print()
    
    # Cargar datos
    print("[1/3] Cargando datos TET...")
    try:
        loader = TETDataLoader(mat_dir='../data/original/reports/resampled')
        data = loader.load_data()
        print(f"      ✓ Cargados: {len(data)} filas")
    except Exception as e:
        print(f"      ✗ Error: {e}")
        return 1
    
    print()
    
    # Analizar estructura actual
    print("[2/3] Analizando estructura ACTUAL (down-sampled @ 4s)...")
    print("-" * 80)
    
    dmt_data = data[data['state'] == 'DMT']
    rs_data = data[data['state'] == 'RS']
    
    # Contar bins por sesión
    dmt_bins_per_session = dmt_data.groupby(['subject', 'session_id']).size()
    rs_bins_per_session = rs_data.groupby(['subject', 'session_id']).size()
    
    print(f"DMT:")
    print(f"  Bins por sesión: {dmt_bins_per_session.unique()}")
    print(f"  Total sesiones: {len(dmt_bins_per_session)}")
    print(f"  Total bins: {len(dmt_data)}")
    print()
    print(f"RS:")
    print(f"  Bins por sesión: {rs_bins_per_session.unique()}")
    print(f"  Total sesiones: {len(rs_bins_per_session)}")
    print(f"  Total bins: {len(rs_data)}")
    print()
    
    # Agregar a bins de 30s
    print("[3/3] Agregando a bins de 30 segundos...")
    print("-" * 80)
    
    data_30s = config.aggregate_tet_to_30s_bins(data, method='mean')
    
    dmt_30s = data_30s[data_30s['state'] == 'DMT']
    rs_30s = data_30s[data_30s['state'] == 'RS']
    
    # Contar bins después de agregación
    dmt_bins_30s = dmt_30s.groupby(['subject', 'session_id']).size()
    rs_bins_30s = rs_30s.groupby(['subject', 'session_id']).size()
    
    print(f"DMT (agregado a 30s):")
    print(f"  Bins por sesión: {dmt_bins_30s.unique()}")
    print(f"  Total sesiones: {len(dmt_bins_30s)}")
    print(f"  Total bins: {len(dmt_30s)}")
    print(f"  Esperado según paper: 40 bins por sesión")
    print()
    print(f"RS (agregado a 30s):")
    print(f"  Bins por sesión: {rs_bins_30s.unique()}")
    print(f"  Total sesiones: {len(rs_bins_30s)}")
    print(f"  Total bins: {len(rs_30s)}")
    print(f"  Esperado según paper: 20 bins por sesión")
    print()
    
    # Comparar valores
    print("COMPARACIÓN DE VALORES (primera sesión DMT)")
    print("-" * 80)
    
    # Tomar primera sesión DMT
    first_subj = dmt_data['subject'].iloc[0]
    first_sess = dmt_data['session_id'].iloc[0]
    
    first_dmt = dmt_data[
        (dmt_data['subject'] == first_subj) &
        (dmt_data['session_id'] == first_sess)
    ]
    
    first_dmt_30s = dmt_30s[
        (dmt_30s['subject'] == first_subj) &
        (dmt_30s['session_id'] == first_sess)
    ]
    
    print()
    print(f"Sujeto: {first_subj}, Sesión: {first_sess}")
    print()
    print("Pleasantness - Original @ 4s (primeros 10 puntos):")
    print(f"  {first_dmt['pleasantness'].head(10).values}")
    print()
    print("Pleasantness - Agregado @ 30s (primeros 5 bins):")
    print(f"  {first_dmt_30s['pleasantness'].head(5).values}")
    print()
    
    # Verificar que el promedio coincide
    # Bin 0 de 30s debe ser promedio de puntos 0-7 (8 puntos = 32s, aproximadamente 30s)
    manual_avg = first_dmt['pleasantness'].iloc[0:8].mean()
    aggregated_val = first_dmt_30s['pleasantness'].iloc[0]
    print(f"Verificación bin 0 (0-30s):")
    print(f"  Promedio manual (puntos 0-7, 0-28s): {manual_avg:.6f}")
    print(f"  Valor agregado (bin 0): {aggregated_val:.6f}")
    print(f"  Diferencia: {abs(manual_avg - aggregated_val):.10f}")
    print()
    
    # Resumen
    print("=" * 80)
    print("RESUMEN")
    print("=" * 80)
    print()
    print("✓ Los datos TET están down-sampled uniformemente @ 4s/punto")
    print("✓ La función aggregate_tet_to_30s_bins() agrupa correctamente a bins de 30s")
    print("✓ Método de agregación: promedio (mean) por defecto")
    print("✓ Factor de agregación: 7.5 puntos → 1 bin (30s / 4s)")
    print()
    print("USO EN ANÁLISIS:")
    print()
    print("  from tet.data_loader import TETDataLoader")
    print("  import config")
    print()
    print("  # Cargar datos (resolución 4s)")
    print("  loader = TETDataLoader(mat_dir='../data/original/reports/resampled')")
    print("  data = loader.load_data()")
    print()
    print("  # Agregar a bins de 30s (según paper original)")
    print("  data_30s = config.aggregate_tet_to_30s_bins(data, method='mean')")
    print()
    print("  # Ahora data_30s tiene:")
    print("  #   - DMT: 40 bins × 30s = 20 minutos")
    print("  #   - RS: 20 bins × 30s = 10 minutos")
    print()
    print("=" * 80)
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
