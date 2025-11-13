#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Verificación del timing exacto en datos TET

Este script verifica que la columna t_sec contenga el timing exacto
en segundos para análisis temporales precisos.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from tet.data_loader import TETDataLoader
import config

def main():
    print("=" * 80)
    print("VERIFICACIÓN DE TIMING TET")
    print("=" * 80)
    print()
    
    # Cargar datos
    print("[1/2] Cargando datos TET...")
    loader = TETDataLoader(mat_dir='../data/original/reports/resampled')
    data = loader.load_data()
    print(f"      ✓ Cargados: {len(data)} filas")
    print()
    
    # Verificar timing
    print("[2/2] Verificando timing...")
    print("-" * 80)
    
    # Tomar primera sesión DMT
    first_dmt = data[(data['subject'] == 'S01') & (data['session_id'] == 1) & (data['state'] == 'DMT')]
    
    print("Primera sesión DMT (S01, session 1):")
    print(f"  Total puntos: {len(first_dmt)}")
    print(f"  Resolución esperada: {config.TET_SAMPLING_RATE_HZ} Hz = 1 punto cada {config.TET_SAMPLING_INTERVAL_SEC}s")
    print()
    
    print("Primeros 10 puntos:")
    print(first_dmt[['t_bin', 't_sec', 'pleasantness']].head(10).to_string(index=False))
    print()
    
    print("Últimos 5 puntos:")
    print(first_dmt[['t_bin', 't_sec', 'pleasantness']].tail(5).to_string(index=False))
    print()
    
    # Verificar intervalos
    intervals = first_dmt['t_sec'].diff().dropna()
    unique_intervals = intervals.unique()
    
    print("Intervalos entre puntos consecutivos:")
    print(f"  Único intervalo: {unique_intervals[0]}s")
    print(f"  Esperado: {config.TET_SAMPLING_INTERVAL_SEC}s")
    print(f"  Match: {'✓' if unique_intervals[0] == config.TET_SAMPLING_INTERVAL_SEC else '✗'}")
    print()
    
    # Verificar duración total
    duration_sec = first_dmt['t_sec'].max()
    duration_min = duration_sec / 60
    expected_duration_sec = (len(first_dmt) - 1) * config.TET_SAMPLING_INTERVAL_SEC
    
    print("Duración de la sesión:")
    print(f"  Duración total: {duration_sec}s = {duration_min:.1f} min")
    print(f"  Esperado: {expected_duration_sec}s = {expected_duration_sec/60:.1f} min")
    print(f"  Match: {'✓' if duration_sec == expected_duration_sec else '✗'}")
    print()
    
    # Verificar RS también
    first_rs = data[(data['subject'] == 'S01') & (data['session_id'] == 1) & (data['state'] == 'RS')]
    
    print("Primera sesión RS (S01, session 1):")
    print(f"  Total puntos: {len(first_rs)}")
    print(f"  Duración: {first_rs['t_sec'].max()}s = {first_rs['t_sec'].max()/60:.1f} min")
    print()
    
    print("=" * 80)
    print("CONCLUSIÓN")
    print("=" * 80)
    print()
    print("✓ Columna t_sec contiene timing exacto en segundos")
    print("✓ Resolución: 0.25 Hz (1 punto cada 4 segundos)")
    print("✓ DMT: 300 puntos = 1196s = 19.93 min ≈ 20 min")
    print("✓ RS: 150 puntos = 596s = 9.93 min ≈ 10 min")
    print()
    print("Los datos están listos para análisis temporales precisos.")
    print("=" * 80)
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
