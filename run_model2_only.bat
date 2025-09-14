@echo off
echo Starting Model 2 - Disease Detection API...
cd "FineTuned Model2"
python -m uvicorn app.fastapi_app:app --host 0.0.0.0 --port 8001 --reload
pause
