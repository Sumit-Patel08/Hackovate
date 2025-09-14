"""
FastAPI backend for AI/ML-Based Cattle Milk Yield Prediction System
Model 1: Comprehensive milk yield prediction API
"""

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
import pandas as pd
import sys
import os
from datetime import datetime
import io

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from predict import MilkYieldPredictor, create_sample_cow_data

# Initialize FastAPI app
app = FastAPI(
    title="AI/ML Cattle Milk Yield Prediction API",
    description="Comprehensive API for predicting daily milk yield per cattle based on multiple factors",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global predictor instance
predictor = None

class CowData(BaseModel):
    """Comprehensive cow data model for prediction."""
    
    # Animal-related data
    age_months: float = Field(..., ge=12, le=200, description="Age of cow in months")
    weight_kg: float = Field(..., ge=300, le=1200, description="Weight of cow in kg")
    breed: str = Field(..., description="Breed of cow")
    lactation_stage: str = Field(..., description="Lactation stage")
    lactation_day: Optional[int] = Field(150, ge=1, le=365, description="Days since lactation start")
    parity: Optional[int] = Field(2, ge=1, le=10, description="Number of calvings")
    historical_yield_7d: Optional[float] = Field(25.0, ge=0, description="Average yield last 7 days (L)")
    historical_yield_30d: Optional[float] = Field(24.0, ge=0, description="Average yield last 30 days (L)")
    
    # Feed and nutrition data
    feed_type: str = Field(..., description="Type of feed")
    feed_quantity_kg: float = Field(..., ge=5, le=25, description="Daily feed quantity in kg")
    feeding_frequency: Optional[int] = Field(3, ge=1, le=6, description="Feeding times per day")
    
    # Activity & behavioral data
    walking_distance_km: Optional[float] = Field(3.0, ge=0, le=15, description="Daily walking distance")
    grazing_hours: Optional[float] = Field(6.0, ge=0, le=12, description="Daily grazing hours")
    rumination_hours: Optional[float] = Field(7.0, ge=4, le=12, description="Daily rumination hours")
    resting_hours: Optional[float] = Field(11.0, ge=6, le=16, description="Daily resting hours")
    
    # Health data
    body_temperature: Optional[float] = Field(38.5, ge=36, le=42, description="Body temperature (°C)")
    heart_rate: Optional[float] = Field(60.0, ge=40, le=100, description="Heart rate (bpm)")
    health_score: Optional[float] = Field(0.9, ge=0, le=1, description="Health score (0-1)")
    
    # Environmental data
    temperature: float = Field(..., ge=-20, le=50, description="Ambient temperature (°C)")
    humidity: float = Field(..., ge=0, le=100, description="Relative humidity (%)")
    season: Optional[str] = Field("summer", description="Current season")
    housing_type: Optional[str] = Field("free_stall", description="Housing type")
    ventilation_score: Optional[float] = Field(0.8, ge=0, le=1, description="Ventilation quality")
    cleanliness_score: Optional[float] = Field(0.8, ge=0, le=1, description="Cleanliness score")
    day_of_year: Optional[int] = Field(180, ge=1, le=365, description="Day of year")
    
    @validator('breed')
    def validate_breed(cls, v):
        allowed_breeds = ['Holstein', 'Jersey', 'Guernsey', 'Ayrshire', 'Brown Swiss', 'Simmental']
        if v not in allowed_breeds:
            raise ValueError(f'Breed must be one of: {allowed_breeds}')
        return v
    
    @validator('lactation_stage')
    def validate_lactation_stage(cls, v):
        allowed_stages = ['early', 'peak', 'mid', 'late', 'dry']
        if v not in allowed_stages:
            raise ValueError(f'Lactation stage must be one of: {allowed_stages}')
        return v
    
    @validator('feed_type')
    def validate_feed_type(cls, v):
        allowed_types = ['green_fodder', 'dry_fodder', 'concentrates', 'silage', 'mixed']
        if v not in allowed_types:
            raise ValueError(f'Feed type must be one of: {allowed_types}')
        return v

class PredictionResponse(BaseModel):
    """Response model for predictions."""
    predicted_milk_yield: Optional[float] = None
    status: str
    timestamp: str
    error: Optional[str] = None
    validation_warnings: Optional[List[str]] = None

class BatchPredictionRequest(BaseModel):
    """Request model for batch predictions."""
    cows_data: List[CowData]

class BatchPredictionResponse(BaseModel):
    """Response model for batch predictions."""
    predictions: List[float]
    count: int
    status: str
    timestamp: str
    error: Optional[str] = None

@app.on_event("startup")
async def startup_event():
    """Initialize the predictor on startup."""
    global predictor
    predictor = MilkYieldPredictor()
    if predictor.model is None:
        print("⚠️ Warning: Model not loaded. Please train the model first.")

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "AI/ML-Based Cattle Milk Yield Prediction API",
        "version": "1.0.0",
        "model": "Model 1 - Comprehensive Milk Yield Prediction",
        "status": "active",
        "endpoints": {
            "/predict": "Single cow milk yield prediction",
            "/predict/batch": "Batch predictions for multiple cows",
            "/validate": "Validate cow data input",
            "/features": "Get feature information and descriptions",
            "/sample": "Get sample input data",
            "/model/info": "Get model performance information",
            "/health": "Health check",
            "/upload": "Upload CSV for batch prediction"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    model_status = "loaded" if predictor and predictor.model is not None else "not_loaded"
    return {
        "status": "healthy",
        "model_status": model_status,
        "timestamp": datetime.now().isoformat(),
        "message": "API is running"
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_milk_yield(cow_data: CowData):
    """Predict milk yield for a single cow."""
    if predictor is None or predictor.model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not available. Please train the model first."
        )
    
    try:
        # Convert to dict
        input_data = cow_data.dict()
        
        # Validate input
        validation = predictor.validate_input(input_data)
        
        # Make prediction
        result = predictor.predict_single_cow(input_data)
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return PredictionResponse(
            predicted_milk_yield=result["predicted_milk_yield"],
            status=result["status"],
            timestamp=result["timestamp"],
            validation_warnings=validation.get("warnings", [])
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest):
    """Predict milk yield for multiple cows."""
    if predictor is None or predictor.model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not available. Please train the model first."
        )
    
    try:
        # Convert to list of dicts
        input_data_list = [cow_data.dict() for cow_data in request.cows_data]
        
        # Make batch predictions
        result = predictor.predict_batch(input_data_list)
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return BatchPredictionResponse(
            predictions=result["predictions"],
            count=result["count"],
            status=result["status"],
            timestamp=result["timestamp"]
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

@app.post("/validate")
async def validate_cow_data(cow_data: CowData):
    """Validate cow data input."""
    if predictor is None:
        raise HTTPException(status_code=503, detail="Predictor not initialized")
    
    input_data = cow_data.dict()
    validation = predictor.validate_input(input_data)
    
    return {
        "valid": validation["valid"],
        "errors": validation["errors"],
        "warnings": validation["warnings"],
        "timestamp": datetime.now().isoformat()
    }

@app.get("/features")
async def get_feature_info():
    """Get information about model features."""
    if predictor is None:
        return {"error": "Predictor not initialized"}
    
    return predictor.get_model_info()

@app.get("/sample")
async def get_sample_input():
    """Get sample input data for testing."""
    return {
        "sample_input": create_sample_cow_data(),
        "description": "Use this sample data to test the prediction endpoints",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/model/info")
async def get_model_info():
    """Get detailed model information and performance metrics."""
    if predictor is None or predictor.model_results is None:
        raise HTTPException(status_code=503, detail="Model information not available")
    
    return {
        "model_performance": predictor.model_results,
        "feature_count": len(predictor.feature_names),
        "training_timestamp": datetime.now().isoformat(),
        "model_type": "Ensemble Regression (Multiple algorithms compared)"
    }

@app.post("/upload")
async def upload_csv_for_prediction(file: UploadFile = File(...)):
    """Upload CSV file for batch prediction."""
    if predictor is None or predictor.model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not available. Please train the model first."
        )
    
    try:
        # Read CSV file
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        
        # Validate CSV structure
        required_columns = ['age_months', 'weight_kg', 'breed', 'lactation_stage', 
                          'feed_type', 'feed_quantity_kg', 'temperature', 'humidity']
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise HTTPException(
                status_code=400,
                detail=f"Missing required columns: {missing_columns}"
            )
        
        # Make predictions
        result = predictor.predict_batch(df)
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        # Add predictions to dataframe
        df['predicted_milk_yield'] = result["predictions"]
        
        return {
            "predictions": result["predictions"],
            "count": result["count"],
            "status": result["status"],
            "timestamp": result["timestamp"],
            "data_with_predictions": df.to_dict('records')
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File processing failed: {str(e)}")

@app.get("/stats")
async def get_prediction_stats():
    """Get prediction statistics and insights."""
    if predictor is None or predictor.model_results is None:
        return {"error": "Model statistics not available"}
    
    # Find best performing model
    best_model = max(predictor.model_results.items(), key=lambda x: x[1]['test_r2'])
    
    return {
        "best_model": {
            "name": best_model[0],
            "r2_score": best_model[1]['test_r2'],
            "rmse": best_model[1]['test_rmse'],
            "mae": best_model[1]['test_mae']
        },
        "all_models": predictor.model_results,
        "feature_count": len(predictor.feature_names),
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
