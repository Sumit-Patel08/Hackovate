"""
FastAPI backend for AI/ML-Based Cattle Disease Detection System
Model 2: Comprehensive disease detection and diagnosis API
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

from predict import CattleDiseasePredictor, create_sample_disease_data

# Initialize FastAPI app
app = FastAPI(
    title="AI/ML Cattle Disease Detection API",
    description="Comprehensive API for detecting and diagnosing cattle diseases based on health parameters",
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

class CowHealthData(BaseModel):
    """Comprehensive cow health data model for disease prediction."""
    
    # Animal-related data
    age_months: float = Field(..., ge=12, le=200, description="Age of cow in months")
    weight_kg: float = Field(..., ge=300, le=1200, description="Weight of cow in kg")
    breed: str = Field(..., description="Breed of cow")
    lactation_stage: str = Field(..., description="Lactation stage")
    lactation_day: Optional[int] = Field(150, ge=0, le=365, description="Days since lactation start")
    parity: Optional[int] = Field(2, ge=1, le=10, description="Number of calvings")
    
    # Vital signs
    body_temperature: float = Field(..., ge=36, le=45, description="Body temperature (°C)")
    heart_rate: float = Field(..., ge=40, le=150, description="Heart rate (bpm)")
    respiratory_rate: float = Field(..., ge=15, le=60, description="Respiratory rate per minute")
    
    # Blood parameters
    white_blood_cells: Optional[float] = Field(7500, ge=2000, le=30000, description="White blood cell count")
    somatic_cell_count: Optional[float] = Field(150000, ge=10000, le=3000000, description="Somatic cell count in milk")
    
    # Rumen health
    rumen_ph: Optional[float] = Field(6.3, ge=5.0, le=7.5, description="Rumen pH level")
    rumen_temperature: Optional[float] = Field(40.0, ge=38, le=43, description="Rumen temperature (°C)")
    
    # Blood chemistry
    calcium_level: Optional[float] = Field(10.0, ge=7.0, le=12.0, description="Blood calcium level mg/dL")
    phosphorus_level: Optional[float] = Field(5.0, ge=2.0, le=8.0, description="Blood phosphorus level mg/dL")
    protein_level: Optional[float] = Field(7.0, ge=5.0, le=10.0, description="Blood protein level g/dL")
    glucose_level: Optional[float] = Field(60, ge=35, le=85, description="Blood glucose level mg/dL")
    
    # Physical examination
    udder_swelling: Optional[int] = Field(0, ge=0, le=1, description="Presence of udder swelling (0/1)")
    lameness_score: Optional[int] = Field(1, ge=0, le=5, description="Lameness severity score (0-5)")
    appetite_score: Optional[int] = Field(4, ge=1, le=5, description="Appetite quality score (1-5)")
    coat_condition: Optional[int] = Field(4, ge=1, le=5, description="Coat condition score (1-5)")
    
    # Feed and activity data
    feed_type: str = Field(..., description="Type of feed")
    feed_quantity_kg: float = Field(..., ge=5, le=25, description="Daily feed quantity in kg")
    feeding_frequency: Optional[int] = Field(3, ge=1, le=6, description="Feeding times per day")
    walking_distance_km: Optional[float] = Field(5.0, ge=0, le=15, description="Daily walking distance")
    grazing_hours: Optional[float] = Field(7.0, ge=0, le=12, description="Daily grazing hours")
    rumination_hours: Optional[float] = Field(7.0, ge=4, le=12, description="Daily rumination hours")
    resting_hours: Optional[float] = Field(10.0, ge=6, le=16, description="Daily resting hours")
    
    # Environmental data
    temperature: float = Field(..., ge=-10, le=45, description="Ambient temperature (°C)")
    humidity: float = Field(..., ge=20, le=95, description="Relative humidity (%)")
    season: Optional[str] = Field("summer", description="Current season")
    
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
    
    @validator('season')
    def validate_season(cls, v):
        allowed_seasons = ['spring', 'summer', 'autumn', 'winter']
        if v not in allowed_seasons:
            raise ValueError(f'Season must be one of: {allowed_seasons}')
        return v

class DiseasePredictionResponse(BaseModel):
    """Response model for disease predictions."""
    predicted_disease: Optional[str] = None
    confidence: Optional[float] = None
    probabilities: Optional[Dict[str, float]] = None
    risk_level: Optional[str] = None
    recommendations: Optional[List[str]] = None
    status: str
    timestamp: str
    error: Optional[str] = None
    validation_warnings: Optional[List[str]] = None

class BatchDiseasePredictionRequest(BaseModel):
    """Request model for batch disease predictions."""
    cows_data: List[CowHealthData]

class BatchDiseasePredictionResponse(BaseModel):
    """Response model for batch disease predictions."""
    predictions: List[DiseasePredictionResponse]
    count: int
    status: str
    timestamp: str
    error: Optional[str] = None

@app.on_event("startup")
async def startup_event():
    """Initialize the disease predictor on startup."""
    global predictor
    try:
        predictor = CattleDiseasePredictor("models/cattle_disease_classifier.joblib")
        if predictor.model is None:
            print("Warning: Disease detection model not found. Please train the model first.")
    except Exception as e:
        print(f"Error initializing disease predictor: {e}")

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "AI/ML Cattle Disease Detection API",
        "version": "1.0.0",
        "description": "Comprehensive disease detection and diagnosis for dairy cattle",
        "endpoints": {
            "predict": "/predict - Single disease prediction",
            "predict_batch": "/predict_batch - Batch disease predictions",
            "health": "/health - API health check",
            "features": "/features - Get feature information",
            "sample": "/sample - Get sample data",
            "diseases": "/diseases - Get supported disease classes"
        }
    }

@app.post("/predict", response_model=DiseasePredictionResponse)
async def predict_disease(cow_data: CowHealthData):
    """Predict disease for a single cow."""
    if predictor is None or predictor.model is None:
        raise HTTPException(
            status_code=503, 
            detail="Disease detection model not available. Please train the model first."
        )
    
    try:
        # Convert Pydantic model to dict
        data_dict = cow_data.dict()
        
        # Make prediction
        result = predictor.predict_disease(data_dict)
        
        return DiseasePredictionResponse(**result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict_batch", response_model=BatchDiseasePredictionResponse)
async def predict_diseases_batch(request: BatchDiseasePredictionRequest):
    """Predict diseases for multiple cows."""
    if predictor is None or predictor.model is None:
        raise HTTPException(
            status_code=503, 
            detail="Disease detection model not available. Please train the model first."
        )
    
    try:
        # Convert Pydantic models to dicts
        cows_data = [cow.dict() for cow in request.cows_data]
        
        # Make batch predictions
        results = predictor.predict_batch(cows_data)
        
        # Convert to response models
        predictions = [DiseasePredictionResponse(**result) for result in results]
        
        return BatchDiseasePredictionResponse(
            predictions=predictions,
            count=len(predictions),
            status="success",
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    model_status = "available" if (predictor and predictor.model) else "not_available"
    
    return {
        "status": "healthy",
        "model_status": model_status,
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    }

@app.get("/features")
async def get_features():
    """Get information about model features."""
    if predictor is None:
        raise HTTPException(status_code=503, detail="Predictor not initialized")
    
    return predictor.get_feature_info()

@app.get("/diseases")
async def get_diseases():
    """Get supported disease classes."""
    if predictor is None or predictor.class_names is None:
        return {
            "diseases": ["healthy", "mastitis", "digestive_disorder", "mineral_deficiency", "lameness"],
            "note": "Default classes - model not loaded"
        }
    
    return {
        "diseases": predictor.class_names,
        "count": len(predictor.class_names),
        "descriptions": {
            "healthy": "No disease detected - normal health status",
            "mastitis": "Inflammation of mammary gland, often bacterial infection",
            "digestive_disorder": "Issues with digestion, rumen function, or acidosis",
            "mineral_deficiency": "Deficiency in essential minerals like calcium, phosphorus",
            "lameness": "Foot problems, injuries, or mobility issues"
        }
    }

@app.get("/sample")
async def get_sample_data():
    """Get sample cow health data for testing."""
    sample_data = create_sample_disease_data()
    
    return {
        "sample_data": sample_data,
        "description": "Sample cow health data for disease prediction testing",
        "usage": "Use this data structure for /predict endpoint"
    }

@app.post("/validate")
async def validate_data(cow_data: CowHealthData):
    """Validate cow health data without making prediction."""
    if predictor is None:
        raise HTTPException(status_code=503, detail="Predictor not initialized")
    
    try:
        # Convert to dict and validate
        data_dict = cow_data.dict()
        validated_data, warnings = predictor.validate_input(data_dict)
        
        return {
            "status": "valid",
            "validated_data": validated_data,
            "warnings": warnings if warnings else None,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Validation failed: {str(e)}")

@app.post("/upload_csv")
async def upload_csv_file(file: UploadFile = File(...)):
    """Upload CSV file for batch disease prediction."""
    if predictor is None or predictor.model is None:
        raise HTTPException(
            status_code=503, 
            detail="Disease detection model not available. Please train the model first."
        )
    
    try:
        # Read CSV file
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        
        # Convert DataFrame to list of dicts
        cows_data = df.to_dict('records')
        
        # Make batch predictions
        results = predictor.predict_batch(cows_data)
        
        # Add predictions to DataFrame
        predictions_df = pd.DataFrame(results)
        
        return {
            "status": "success",
            "file_name": file.filename,
            "rows_processed": len(df),
            "predictions": results,
            "summary": {
                "total_cows": len(results),
                "disease_distribution": pd.Series([r.get('predicted_disease') for r in results]).value_counts().to_dict()
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"CSV processing failed: {str(e)}")

@app.get("/model_info")
async def get_model_info():
    """Get information about the trained model."""
    if predictor is None:
        raise HTTPException(status_code=503, detail="Predictor not initialized")
    
    if predictor.artifacts is None:
        return {"error": "Model artifacts not available"}
    
    training_report = predictor.artifacts.get('training_report', {})
    
    return {
        "model_type": "Cattle Disease Classification",
        "version": "1.0.0",
        "training_info": training_report,
        "feature_count": len(predictor.feature_names) if predictor.feature_names else 0,
        "disease_classes": predictor.class_names if predictor.class_names else [],
        "status": "loaded" if predictor.model else "not_loaded"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
