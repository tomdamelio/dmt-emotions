# -*- coding: utf-8 -*-
"""
Ejemplo de flujo de trabajo para validación de datos fisiológicos

Este script muestra el flujo completo para validar datos preprocesados:
1. Generar log automático de validación
2. Comparar con anotaciones manuales
3. Usar validación para filtrar datos en análisis posteriores

Usage:
    python scripts/validation_workflow_example.py
"""

import os
import sys
import subprocess

def run_command(command, description):
    """Ejecuta un comando y muestra el resultado"""
    print(f"\n🔄 {description}")
    print(f"   Comando: {command}")
    
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"   ✅ Completado exitosamente")
            if result.stdout:
                print(f"   📄 Output: {result.stdout[:200]}...")
        else:
            print(f"   ❌ Error (código {result.returncode})")
            if result.stderr:
                print(f"   📄 Error: {result.stderr[:200]}...")
    except Exception as e:
        print(f"   ❌ Excepción: {e}")

def main():
    """Flujo de trabajo completo"""
    print("🏗️  FLUJO DE TRABAJO DE VALIDACIÓN DE DATOS FISIOLÓGICOS")
    print("=" * 60)
    
    print("""
📋 PASOS DEL FLUJO DE TRABAJO:

1. 🏃 Ejecutar preprocesamiento fisiológico (si no está hecho)
   → python scripts/preprocess_phys.py
   
2. 🧪 Generar log automático de validación  
   → python test/test_phys_preprocessing.py
   
3. ✏️  Completar anotaciones manuales en validation_log.json
   
4. 🔍 Comparar validaciones manual vs automática
   → python scripts/compare_validation.py
   
5. 📊 Usar validación para filtrar datos en análisis
   → python scripts/process_eda.py (usa validation_log.json automáticamente)
""")
    
    # Verificar archivos necesarios
    print("\n📁 VERIFICANDO ARCHIVOS:")
    
    files_to_check = [
        ('validation_log.json', 'Archivo de anotaciones manuales'),
        ('automatic_validation_log.json', 'Log automático (generado por test)'),
        ('scripts/compare_validation.py', 'Script de comparación'),
        ('test/test_phys_preprocessing.py', 'Script de testing')
    ]
    
    for file_path, description in files_to_check:
        exists = "✅" if os.path.exists(file_path) else "❌"
        print(f"   {exists} {description}: {file_path}")
    
    # Mostrar comandos de ejemplo
    print(f"\n🚀 COMANDOS DE EJEMPLO:")
    print(f"   # Generar validación automática:")
    print(f"   python test/test_phys_preprocessing.py")
    print(f"   ")
    print(f"   # Comparar validaciones:")
    print(f"   python scripts/compare_validation.py")
    print(f"   ")
    print(f"   # Procesar EDA con filtrado:")
    print(f"   python scripts/process_eda.py")
    
    # Ejemplo de estructura del archivo de validación
    print(f"\n📝 EJEMPLO DE ANOTACIÓN MANUAL EN validation_log.json:")
    print(f'''   "S04": {{
     "eda": {{
       "dmt_session1_high": {{
         "category": "good",
         "notes": "Señal limpia, sin artefactos visibles"
       }},
       "rs_session1_high": {{
         "category": "acceptable", 
         "notes": "Algunos artefactos menores al inicio"
       }},
       "dmt_session2_low": {{
         "category": "bad",
         "notes": "Señal saturada, no usable para análisis"
       }}
     }}
   }}''')
    
    print(f"\n✅ Flujo de trabajo listo para usar!")
    print(f"📖 Consulta validation_log.json para completar las anotaciones manuales")

if __name__ == "__main__":
    main()
