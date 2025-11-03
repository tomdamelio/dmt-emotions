@echo off
echo Running physiological analyses in parallel...
start "RVT Analysis" python scripts/run_resp_rvt_analysis.py
start "HR Analysis" python scripts/run_ecg_hr_analysis.py
echo Both analyses started in separate windows.
