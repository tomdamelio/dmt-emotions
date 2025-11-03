Write-Host "Running physiological analyses in parallel..." -ForegroundColor Green

# Start both processes in separate windows
$proc1 = Start-Process powershell -ArgumentList "-NoExit", "-Command", "python scripts/run_resp_rvt_analysis.py" -PassThru
$proc2 = Start-Process powershell -ArgumentList "-NoExit", "-Command", "python scripts/run_ecg_hr_analysis.py" -PassThru

Write-Host "Started processes:" -ForegroundColor Cyan
Write-Host "  RVT Analysis (PID: $($proc1.Id))" -ForegroundColor Yellow
Write-Host "  HR Analysis (PID: $($proc2.Id))" -ForegroundColor Yellow
Write-Host "`nBoth analyses running in separate windows..." -ForegroundColor Green
