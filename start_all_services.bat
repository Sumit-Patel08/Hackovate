@echo off
echo Starting Complete Cattle Management System...
echo ==========================================

echo Starting Model 1 (Milk Yield Prediction) on port 8000...
start "Model 1 - Milk Yield" cmd /k "cd /d "FineTuned Model1" && python -m uvicorn app.fastapi_app:app --host 0.0.0.0 --port 8000 --reload"

timeout /t 3 /nobreak >nul

echo Starting Model 2 (Disease Detection) on port 8001...
start "Model 2 - Disease Detection" cmd /k "cd /d "FineTuned Model2" && python -m uvicorn app.fastapi_app:app --host 0.0.0.0 --port 8001 --reload"

timeout /t 3 /nobreak >nul

echo Starting React Frontend on port 3000...
start "React Frontend" cmd /k "cd /d Frontend && npm run dev"

echo.
echo ==========================================
echo All services are starting...
echo.
echo Model 1 (Milk Yield):     http://localhost:8000
echo Model 2 (Disease Detection): http://localhost:8001
echo React Frontend:           http://localhost:3000
echo.
echo API Documentation:
echo Model 1 Docs:            http://localhost:8000/docs
echo Model 2 Docs:            http://localhost:8001/docs
echo ==========================================

pause
