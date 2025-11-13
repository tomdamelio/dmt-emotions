#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script de comparación de dimensiones TET

Compara el orden de dimensiones en config.py con el script original
para verificar consistencia.
"""

import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import config

# Dimensiones del script original (old_scripts/Visualizacion Reportes y Clustering.py)
DIMENSIONES_ORIGINALES = [
    'Pleasantness', 'Unpleasantness', 'Emotional_Intensity', 
    'Elementary_Imagery', 'Complex_Imagery', 'Auditory', 
    'Interoception', 'Bliss', 'Anxiety', 'Entity', 'Selfhood', 
    'Disembodiment', 'Salience', 'Temporality', 'General_Intensity'
]

def main():
    print("=" * 80)
    print("COMPARACIÓN DE DIMENSIONES TET")
    print("=" * 80)
    print()
    
    # Normalizar a minúsculas para comparación
    config_dims = [d.lower() for d in config.TET_DIMENSION_COLUMNS]
    original_dims = [d.lower() for d in DIMENSIONES_ORIGINALES]
    
    # Verificar longitud
    if len(config_dims) != len(original_dims):
        print(f"❌ ERROR: Diferente número de dimensiones")
        print(f"   Config: {len(config_dims)}")
        print(f"   Original: {len(original_dims)}")
        return 1
    
    # Comparar dimensión por dimensión
    all_match = True
    print("Comparación dimensión por dimensión:")
    print("-" * 80)
    print(f"{'#':<3} {'Config':<25} {'Original':<25} {'Match':<10}")
    print("-" * 80)
    
    for i, (cfg, orig) in enumerate(zip(config_dims, original_dims), 1):
        match = "✓" if cfg == orig else "✗"
        if cfg != orig:
            all_match = False
        print(f"{i:<3} {cfg:<25} {orig:<25} {match:<10}")
    
    print("-" * 80)
    print()
    
    if all_match:
        print("✅ ÉXITO: Todas las dimensiones coinciden con el script original")
        print()
        print("Fuente de verificación:")
        print("  old_scripts/Visualizacion Reportes y Clustering.py")
        return 0
    else:
        print("❌ ERROR: Hay discrepancias entre config.py y el script original")
        print()
        print("Por favor, revisa las diferencias arriba y corrige config.py")
        return 1

if __name__ == '__main__':
    sys.exit(main())
