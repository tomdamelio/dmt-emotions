"""
Script para ejecutar todos los análisis principales en orden.

Este script ejecuta el pipeline completo de análisis DMT:
1. Análisis TET (datos fenomenológicos)
2. Análisis HR (frecuencia cardíaca)
3. Análisis SMNA (actividad sudomotora)
4. Análisis RVT (volumen respiratorio)
5. Índice compuesto de arousal (PC1)
6. Análisis de acoplamiento (TET-Fisiología)
7. Generación de figuras finales

Autor: DMT Analysis Team
Fecha: 2026-01-11
"""

import subprocess
import sys
from pathlib import Path
from datetime import datetime


def run_script(script_path: str, description: str) -> bool:
    """
    Ejecuta un script Python y reporta el resultado.
    
    Args:
        script_path: Ruta al script
        description: Descripción del análisis
    
    Returns:
        True si exitoso, False si falló
    """
    print("\n" + "="*80)
    print(f"EJECUTANDO: {description}")
    print(f"Script: {script_path}")
    print(f"Hora: {datetime.now().strftime('%H:%M:%S')}")
    print("="*80 + "\n")
    
    try:
        result = subprocess.run(
            ['micromamba', 'run', '-n', 'dmt-emotions', 'python', script_path],
            capture_output=True,
            text=True,
            timeout=600  # 10 minutos timeout por script
        )
        
        if result.returncode == 0:
            print(f"\n✅ {description} - COMPLETADO")
            return True
        else:
            print(f"\n❌ {description} - FALLÓ")
            print(f"Error: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"\n⏱️ {description} - TIMEOUT (>10 min)")
        return False
    except Exception as e:
        print(f"\n❌ {description} - ERROR: {str(e)}")
        return False


def main():
    """Ejecuta todos los análisis en orden."""
    
    print("\n" + "="*80)
    print("PIPELINE COMPLETO DE ANÁLISIS DMT")
    print("="*80)
    print(f"Inicio: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    # Definir scripts en orden de ejecución
    analyses = [
        ('src/run_tet_analysis.py', '1. Análisis TET (Datos Fenomenológicos)'),
        ('src/run_ecg_hr_analysis.py', '2. Análisis HR (Frecuencia Cardíaca)'),
        ('src/run_eda_smna_analysis.py', '3. Análisis SMNA (Actividad Sudomotora)'),
        ('src/run_resp_rvt_analysis.py', '4. Análisis RVT (Volumen Respiratorio)'),
        ('src/run_composite_arousal_index.py', '5. Índice Compuesto de Arousal (PC1)'),
        ('src/run_coupling_analysis.py', '6. Análisis de Acoplamiento (TET-Fisiología)'),
        ('src/run_figures.py', '7. Generación de Figuras Finales'),
    ]
    
    results = {}
    
    # Ejecutar cada análisis
    for script_path, description in analyses:
        success = run_script(script_path, description)
        results[description] = success
        
        # Si un análisis crítico falla, preguntar si continuar
        if not success and script_path != 'src/run_figures.py':
            print(f"\n⚠️  ADVERTENCIA: {description} falló.")
            response = input("¿Continuar con los siguientes análisis? (s/n): ")
            if response.lower() != 's':
                print("\n🛑 Pipeline interrumpido por el usuario.")
                break
    
    # Resumen final
    print("\n" + "="*80)
    print("RESUMEN DE EJECUCIÓN")
    print("="*80)
    
    for description, success in results.items():
        status = "✅ EXITOSO" if success else "❌ FALLÓ"
        print(f"{status} - {description}")
    
    print("\n" + "="*80)
    print(f"Fin: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    # Contar éxitos
    total = len(results)
    exitosos = sum(1 for success in results.values() if success)
    
    print(f"\nResultado: {exitosos}/{total} análisis completados exitosamente")
    
    if exitosos == total:
        print("\n🎉 ¡TODOS LOS ANÁLISIS COMPLETADOS EXITOSAMENTE!")
        return 0
    else:
        print(f"\n⚠️  {total - exitosos} análisis fallaron. Revisar logs arriba.")
        return 1


if __name__ == '__main__':
    sys.exit(main())
