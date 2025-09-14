# 🐄 AI/ML-Based Cattle Milk Yield and Health Prediction

## Problem Statement
Dairy farming faces challenges with fluctuating milk yields due to multiple factors including feed quality, animal health, environmental conditions, and activity levels. This AI/ML platform provides predictive capabilities to optimize farm management and detect health issues early.

## Model 1: Milk Yield Prediction
**Objective**: Build an ML model to estimate daily milk production per cow under given conditions.

### Key Features
**Animal-related Data:**
- Breed, age, weight, lactation stage, parity
- Historical milk yield records
- Reproductive cycle information

**Feed and Nutrition Data:**
- Type of feed (green fodder, dry fodder, concentrates, supplements)
- Quantity consumed daily, feeding frequency

**Activity & Behavioral Data:**
- Walking distance, grazing duration, rumination time
- Resting/sleeping hours

**Health Data:**
- Veterinary records, vaccination history
- Body temperature, heart rate, activity alerts

**Environmental Data:**
- Ambient temperature, humidity, seasonal conditions
- Housing conditions (ventilation, cleanliness)

## Installation
```bash
pip install -r requirements.txt
```

## Usage
1. **Training**: `python train_model.py`
2. **API Server**: `python run_fastapi.py`
3. **Dashboard**: `python run_streamlit.py`

## Project Structure
```
├── train_model.py          # Model 1 training script
├── predict.py              # Prediction utilities
├── data_generator.py       # Synthetic data generation
├── app/                    # Web interfaces
├── models/                 # Trained models
├── data/                   # Datasets
├── reports/               # Generated reports
└── logs/                  # Training logs
```

## Features
- **Regression Model**: Predicts daily milk output per cattle
- **Interactive Dashboard**: Easy-to-use farmer interface
- **Comprehensive Reports**: CSV/Excel/PDF exports
- **Real-time Predictions**: Input data and get instant predictions
