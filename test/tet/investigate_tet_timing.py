#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Investigar la estructura temporal de los datos TET"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from scripts.tet.data_loader import TETDataLoader

loader = TETDataLoader(mat_dir='../data/original/reports/resampled')
data = loader.load_data()

# Tomar primera sesión DMT
first_session = data[(data['subject'] == 'S01') & (data['session_id'] == 1) & (data['state'] == 'DMT')]

print("Primera sesión DMT (S01, session 1):")
print(f"Total bins: {len(first_session)}")
print()
print("Primeros 10 valores de t_bin y t_sec:")
print(first_session[['t_bin', 't_sec', 'pleasantness']].head(10))
print()
print("Últimos 10 valores:")
print(first_session[['t_bin', 't_sec', 'pleasantness']].tail(10))
print()
print(f"Duración total: {first_session['t_sec'].max()} segundos")
print(f"Duración esperada DMT: 20 minutos = 1200 segundos")
print()
print("Diferencia entre bins consecutivos:")
print(first_session['t_sec'].diff().value_counts().head())
