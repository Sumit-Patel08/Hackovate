# Cattle Disease Detection Model (Model 2)

## Overview
This is a comprehensive AI/ML system for detecting and diagnosing cattle diseases that may cause drops in milk production. The model analyzes various health parameters to classify diseases including mastitis, digestive disorders, mineral deficiencies, and lameness.

## Features
- **Disease Classification**: Detects 5 disease categories (healthy, mastitis, digestive_disorder, mineral_deficiency, lameness)
- **Comprehensive Health Analysis**: Uses 25+ health parameters including vital signs, blood chemistry, and behavioral data
- **Risk Assessment**: Provides risk levels (low, medium, high) and treatment recommendations
- **FastAPI Backend**: RESTful API for easy integration
- **Batch Processing**: Support for multiple cattle analysis
- **Data Validation**: Automatic input validation and error handling

## Disease Categories
1. **Healthy**: Normal health status
2. **Mastitis**: Mammary gland inflammation/infection
3. **Digestive Disorder**: Rumen issues, acidosis, digestive problems
4. **Mineral Deficiency**: Calcium, phosphorus, or other mineral deficiencies
5. **Lameness**: Foot problems, mobility issues

## Installation & Setup

### Step 1: Install Dependencies
```bash
cd "FineTuned Model2"
pip install -r requirements.txt
```

### Step 2: Train the Model
```bash
python run_training.py
```
This will:
- Generate synthetic cattle disease dataset (5000 samples)
- Train multiple classification models
- Select the best performing model
- Save model artifacts and generate reports

### Step 3: Start the API Server
```bash
python run_fastapi.py
```
The API will be available at:
- Server: http://localhost:8001
- Documentation: http://localhost:8001/docs

## API Endpoints

### Main Endpoints
- `POST /predict` - Single disease prediction
- `POST /predict_batch` - Batch disease predictions
- `POST /upload_csv` - Upload CSV for batch processing
- `GET /health` - API health check
- `GET /features` - Get feature information
- `GET /diseases` - Get supported disease classes
- `GET /sample` - Get sample data for testing

### Example Usage

#### Single Prediction
```python
import requests

data = {
    "breed": "Holstein",
    "age_months": 48,
    "weight_kg": 550,
    "lactation_stage": "peak",
    "body_temperature": 39.2,
    "heart_rate": 68,
    "somatic_cell_count": 150000,
    "feed_type": "mixed",
    "feed_quantity_kg": 16.5,
    "temperature": 22,
    "humidity": 65
    # ... other parameters
}

response = requests.post("http://localhost:8001/predict", json=data)
result = response.json()

print(f"Predicted Disease: {result['predicted_disease']}")
print(f"Confidence: {result['confidence']}")
print(f"Risk Level: {result['risk_level']}")
```

## Model Performance
- **Algorithm**: Ensemble of Random Forest, Gradient Boosting, and Logistic Regression
- **Features**: 25+ health and environmental parameters
- **Accuracy**: ~95%+ on synthetic dataset
- **F1 Score**: ~94%+ weighted average

## File Structure
```
FineTuned Model2/
├── app/
│   ├── __init__.py
│   └── fastapi_app.py          # FastAPI backend
├── data/                       # Generated datasets
├── models/                     # Trained model artifacts
├── plots/                      # Training visualizations
├── reports/                    # Training reports
├── data_generator.py           # Synthetic data generation
├── train_model.py             # Model training pipeline
├── predict.py                 # Prediction utilities
├── requirements.txt           # Dependencies
├── run_training.py           # Training script
├── run_fastapi.py            # API server script
└── README.md                 # This file
```

## Health Parameters Used

### Vital Signs
- Body temperature (°C)
- Heart rate (bpm)
- Respiratory rate (per minute)

### Blood Parameters
- White blood cell count
- Somatic cell count
- Calcium level
- Phosphorus level
- Protein level
- Glucose level

### Physical Examination
- Udder swelling (0/1)
- Lameness score (0-5)
- Appetite score (1-5)
- Coat condition (1-5)

### Rumen Health
- Rumen pH
- Rumen temperature

### Activity & Behavior
- Walking distance
- Grazing hours
- Rumination hours
- Resting hours

### Environmental
- Ambient temperature
- Humidity
- Season

## Recommendations System
The model provides specific treatment recommendations based on:
- Predicted disease type
- Confidence level
- Severity of symptoms
- Risk factors present

## Integration with Model 1
This disease detection model complements Model 1 (milk yield prediction) by:
- Identifying health issues that cause milk yield drops
- Providing early disease detection
- Enabling preventive healthcare measures
- Supporting farm management decisions

## Troubleshooting

### Model Not Found Error
If you get "Model not found" errors:
1. Run training first: `python run_training.py`
2. Check that `models/` directory contains the model files
3. Verify all dependencies are installed

### API Connection Issues
- Ensure the API is running on port 8001
- Check firewall settings
- Verify CORS settings for frontend integration

## Future Enhancements
- Integration with IoT sensors for real-time monitoring
- Mobile app for field use
- Integration with veterinary management systems
- Advanced deep learning models
- Multi-language support
