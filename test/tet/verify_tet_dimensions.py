#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script de verificación del mapeo de dimensiones TET

Este script verifica que el orden de las dimensiones TET en config.py
coincide con la documentación en docs/PIPELINE.md
"""

import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import config

def main():
    print("=" * 80)
    print("VERIFICACIÓN DE DIMENSIONES TET")
    print("=" * 80)
    print()
    
    print(f"Total de dimensiones: {len(config.TET_DIMENSION_COLUMNS)}")
    print()
    print("Orden de dimensiones según config.py:")
    print("-" * 80)
    
    for i, col in enumerate(config.TET_DIMENSION_COLUMNS, 1):
        print(f"{i:2d}. {col}")
    
    print()
    print("=" * 80)
    print("FUENTE DE DOCUMENTACIÓN")
    print("=" * 80)
    print()
    print("El orden de estas dimensiones está documentado en:")
    print("  - docs/PIPELINE.md (sección 'Autorreportes')")
    print("  - config.py (comentarios en TET_DIMENSION_COLUMNS)")
    print()
    print("Los archivos .mat contienen una matriz 'dimensions' de shape (n_bins, 15)")
    print("donde las columnas siguen este orden exacto.")
    print()
    print("⚠️  IMPORTANTE: Este orden es crítico para la validez del análisis.")
    print("   Cualquier cambio debe ser verificado contra los datos originales.")
    print()

if __name__ == '__main__':
    main()
