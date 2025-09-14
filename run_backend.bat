@echo off
echo Starting FastAPI Backend for Cattle Milk Yield Prediction...
cd "FineTuned Model1"
python -m uvicorn app.fastapi_app:app --host 0.0.0.0 --port 8000 --reload
pause
