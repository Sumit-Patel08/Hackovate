@echo off
echo Starting Model 1 - Milk Yield Prediction API...
cd "FineTuned Model1"
python -m uvicorn app.fastapi_app:app --host 0.0.0.0 --port 8000 --reload
pause
