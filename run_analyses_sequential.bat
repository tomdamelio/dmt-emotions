@echo off
echo ================================================================================
echo EJECUTANDO PIPELINE DE ANALISIS DMT
echo ================================================================================
echo.

echo [1/7] Ejecutando analisis TET...
micromamba run -n dmt-emotions python src/run_tet_analysis.py
if %errorlevel% neq 0 (
    echo ERROR en analisis TET
    pause
    exit /b 1
)
echo ✓ TET completado
echo.

echo [2/7] Ejecutando analisis HR...
micromamba run -n dmt-emotions python src/run_ecg_hr_analysis.py
if %errorlevel% neq 0 (
    echo ERROR en analisis HR
    pause
    exit /b 1
)
echo ✓ HR completado
echo.

echo [3/7] Ejecutando analisis SMNA...
micromamba run -n dmt-emotions python src/run_eda_smna_analysis.py
if %errorlevel% neq 0 (
    echo ERROR en analisis SMNA
    pause
    exit /b 1
)
echo ✓ SMNA completado
echo.

echo [4/7] Ejecutando analisis RVT...
micromamba run -n dmt-emotions python src/run_resp_rvt_analysis.py
if %errorlevel% neq 0 (
    echo ERROR en analisis RVT
    pause
    exit /b 1
)
echo ✓ RVT completado
echo.

echo [5/7] Ejecutando indice compuesto...
micromamba run -n dmt-emotions python src/run_composite_arousal_index.py
if %errorlevel% neq 0 (
    echo ERROR en indice compuesto
    pause
    exit /b 1
)
echo ✓ Indice compuesto completado
echo.

echo [6/7] Ejecutando analisis de acoplamiento...
micromamba run -n dmt-emotions python src/run_coupling_analysis.py
if %errorlevel% neq 0 (
    echo ERROR en analisis de acoplamiento
    pause
    exit /b 1
)
echo ✓ Acoplamiento completado
echo.

echo [7/7] Generando figuras finales...
micromamba run -n dmt-emotions python src/run_figures.py
if %errorlevel% neq 0 (
    echo ERROR en generacion de figuras
    pause
    exit /b 1
)
echo ✓ Figuras completadas
echo.

echo ================================================================================
echo PIPELINE COMPLETADO EXITOSAMENTE
echo ================================================================================
echo.
echo Resultados guardados en: ./results/
echo Figuras guardadas en: ./results/figures/
echo.
pause
