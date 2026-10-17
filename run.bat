@echo off
title TuneForge - QLoRA Studio
cls
echo ========================================================
echo       TuneForge - QLoRA Studio & Model Registry
echo ========================================================
echo.
echo [1/3] Checking dependencies...
python -c "import uvicorn, fastapi" 2>nul || pip install fastapi uvicorn
if not exist "frontend\node_modules" (
    echo Installing frontend dependencies...
    cd frontend && npm install && cd ..
)

echo [2/3] Starting Backend API (Port 8011)...
start "TuneForge API (Port 8011)" cmd /k "cd /d %~dp0backend && python -m uvicorn src.tuneforge.main:app --port 8011 --reload"

echo [3/3] Starting Frontend UI (Port 3011)...
start "TuneForge UI (Port 3011)" cmd /k "cd /d %~dp0frontend && npm run dev"

timeout /t 3 /nobreak >nul
start http://localhost:3011
echo.
echo TuneForge is running!
echo   Frontend : http://localhost:3011
echo   Backend  : http://localhost:8011/docs
echo.
echo Press any key to stop all TuneForge processes...
pause >nul
taskkill /fi "WINDOWTITLE eq TuneForge API*" /f >nul 2>&1
taskkill /fi "WINDOWTITLE eq TuneForge UI*" /f >nul 2>&1
echo Done.
