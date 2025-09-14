"""
Prediction utilities for Model 1: Milk Yield Prediction
Comprehensive cattle milk yield prediction system.
"""

import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime

class MilkYieldPredictor:
    """Production-ready milk yield prediction system."""
    
    def __init__(self, model_path="models/best_milk_yield_model.pkl"):
        """Initialize predictor with trained model."""
        self.model_path = model_path
        self.model = None
        self.feature_names = None
        self.model_results = None
        self.load_model_artifacts()
    
    def load_model_artifacts(self):
        """Load trained model and related artifacts."""
        try:
            if os.path.exists(self.model_path):
                self.model = joblib.load(self.model_path)
                print("✅ Model loaded successfully!")
                
                # Load feature names
                self.feature_names = joblib.load('models/feature_names.pkl')
                print(f"✅ Feature names loaded: {len(self.feature_names)} features")
                
                # Load model results
                self.model_results = joblib.load('models/model_results.pkl')
                print("✅ Model performance metrics loaded")
                
                return True
            else:
                print("❌ Model not found. Please train the model first using 'python train_model.py'")
                return False
                
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    
    def predict_single_cow(self, cow_data):
        """Predict milk yield for a single cow."""
        if self.model is None:
            return {"error": "Model not loaded"}
        
        try:
            # Convert to DataFrame
            if isinstance(cow_data, dict):
                df = pd.DataFrame([cow_data])
            else:
                df = cow_data.copy()
            
            # Ensure all required features are present
            for feature in self.feature_names:
                if feature not in df.columns:
                    # Set default values for missing features
                    df[feature] = self._get_default_value(feature)
            
            # Select only required features in correct order
            df = df[self.feature_names]
            
            # Make prediction
            prediction = self.model.predict(df)
            
            return {
                "predicted_milk_yield": float(prediction[0]),
                "status": "success",
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {"error": f"Prediction failed: {str(e)}"}
    
    def predict_batch(self, cows_data):
        """Predict milk yield for multiple cows."""
        if self.model is None:
            return {"error": "Model not loaded"}
        
        try:
            # Convert to DataFrame if needed
            if isinstance(cows_data, list):
                df = pd.DataFrame(cows_data)
            else:
                df = cows_data.copy()
            
            # Ensure all required features are present
            for feature in self.feature_names:
                if feature not in df.columns:
                    df[feature] = self._get_default_value(feature)
            
            # Select only required features
            df = df[self.feature_names]
            
            # Make predictions
            predictions = self.model.predict(df)
            
            return {
                "predictions": predictions.tolist(),
                "count": len(predictions),
                "status": "success",
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {"error": f"Batch prediction failed: {str(e)}"}
    
    def _get_default_value(self, feature):
        """Get default value for missing features."""
        defaults = {
            # Animal data
            'age_months': 48,
            'weight_kg': 550,
            'lactation_day': 150,
            'parity': 2,
            'historical_yield_7d': 25,
            'historical_yield_30d': 24,
            
            # Feed data
            'feed_quantity_kg': 12,
            'feeding_frequency': 3,
            
            # Activity data
            'walking_distance_km': 3,
            'grazing_hours': 6,
            'rumination_hours': 7,
            'resting_hours': 11,
            
            # Health data
            'body_temperature': 38.5,
            'heart_rate': 60,
            'health_score': 0.9,
            
            # Environmental data
            'temperature': 20,
            'humidity': 65,
            'ventilation_score': 0.8,
            'cleanliness_score': 0.8,
            'day_of_year': 180,
            
            # Categorical defaults
            'breed': 'Holstein',
            'lactation_stage': 'mid',
            'feed_type': 'mixed',
            'season': 'summer',
            'housing_type': 'free_stall'
        }
        
        return defaults.get(feature, 0)
    
    def get_model_info(self):
        """Get model information and performance metrics."""
        if self.model_results is None:
            return {"error": "Model results not available"}
        
        return {
            "model_performance": self.model_results,
            "feature_count": len(self.feature_names),
            "required_features": self.feature_names,
            "feature_descriptions": self._get_feature_descriptions()
        }
    
    def _get_feature_descriptions(self):
        """Get descriptions for all features."""
        return {
            # Animal-related data
            'age_months': 'Age of cow in months',
            'weight_kg': 'Weight of cow in kilograms',
            'lactation_day': 'Days since start of current lactation',
            'parity': 'Number of times cow has calved',
            'historical_yield_7d': 'Average milk yield over last 7 days (L)',
            'historical_yield_30d': 'Average milk yield over last 30 days (L)',
            'breed': 'Breed of cow (Holstein, Jersey, etc.)',
            'lactation_stage': 'Current lactation stage (early, peak, mid, late, dry)',
            
            # Feed and nutrition data
            'feed_quantity_kg': 'Daily feed quantity in kg',
            'feeding_frequency': 'Number of feeding times per day',
            'feed_type': 'Type of feed (green_fodder, dry_fodder, concentrates, silage, mixed)',
            
            # Activity & behavioral data
            'walking_distance_km': 'Daily walking distance in km',
            'grazing_hours': 'Hours spent grazing per day',
            'rumination_hours': 'Hours spent ruminating per day',
            'resting_hours': 'Hours spent resting per day',
            
            # Health data
            'body_temperature': 'Body temperature in Celsius',
            'heart_rate': 'Heart rate in beats per minute',
            'health_score': 'Overall health score (0-1, higher is better)',
            
            # Environmental data
            'temperature': 'Ambient temperature in Celsius',
            'humidity': 'Relative humidity percentage',
            'season': 'Current season (spring, summer, autumn, winter)',
            'housing_type': 'Type of housing (free_stall, tie_stall, pasture, compost_barn)',
            'ventilation_score': 'Housing ventilation quality (0-1)',
            'cleanliness_score': 'Housing cleanliness score (0-1)',
            'day_of_year': 'Day of year (1-365)'
        }
    
    def validate_input(self, cow_data):
        """Validate input data for prediction."""
        errors = []
        warnings = []
        
        if isinstance(cow_data, dict):
            data = cow_data
        else:
            data = cow_data.iloc[0].to_dict() if len(cow_data) > 0 else {}
        
        # Check critical ranges
        if 'age_months' in data:
            if data['age_months'] < 12 or data['age_months'] > 200:
                warnings.append("Age outside typical range (12-200 months)")
        
        if 'weight_kg' in data:
            if data['weight_kg'] < 300 or data['weight_kg'] > 1200:
                warnings.append("Weight outside typical range (300-1200 kg)")
        
        if 'body_temperature' in data:
            if data['body_temperature'] < 37 or data['body_temperature'] > 40:
                warnings.append("Body temperature outside normal range (37-40°C)")
        
        if 'health_score' in data:
            if data['health_score'] < 0 or data['health_score'] > 1:
                errors.append("Health score must be between 0 and 1")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings
        }

def create_sample_cow_data():
    """Create sample cow data for testing."""
    return {
        # Animal data
        'age_months': 48,
        'weight_kg': 580,
        'breed': 'Holstein',
        'lactation_stage': 'peak',
        'lactation_day': 60,
        'parity': 3,
        'historical_yield_7d': 32.5,
        'historical_yield_30d': 31.8,
        
        # Feed data
        'feed_type': 'mixed',
        'feed_quantity_kg': 14.5,
        'feeding_frequency': 3,
        
        # Activity data
        'walking_distance_km': 4.2,
        'grazing_hours': 7.5,
        'rumination_hours': 8.2,
        'resting_hours': 8.3,
        
        # Health data
        'body_temperature': 38.6,
        'heart_rate': 65,
        'health_score': 0.95,
        
        # Environmental data
        'temperature': 22,
        'humidity': 60,
        'season': 'summer',
        'housing_type': 'free_stall',
        'ventilation_score': 0.9,
        'cleanliness_score': 0.85,
        'day_of_year': 180
    }

def test_predictions():
    """Test the prediction system."""
    print("🧪 Testing Milk Yield Prediction System")
    print("=" * 50)
    
    # Initialize predictor
    predictor = MilkYieldPredictor()
    
    if predictor.model is None:
        print("❌ Cannot test - model not available")
        print("Please run 'python train_model.py' first to train the model")
        return
    
    # Test single prediction
    print("\n1. Testing single cow prediction...")
    sample_data = create_sample_cow_data()
    
    # Validate input
    validation = predictor.validate_input(sample_data)
    print(f"Input validation: {'✅ Valid' if validation['valid'] else '❌ Invalid'}")
    if validation['warnings']:
        print(f"Warnings: {validation['warnings']}")
    if validation['errors']:
        print(f"Errors: {validation['errors']}")
    
    # Make prediction
    result = predictor.predict_single_cow(sample_data)
    print(f"Prediction result: {result}")
    
    # Test batch prediction
    print("\n2. Testing batch prediction...")
    batch_data = [sample_data.copy() for _ in range(3)]
    
    # Modify some values for variety
    batch_data[1]['age_months'] = 36
    batch_data[1]['lactation_stage'] = 'mid'
    batch_data[2]['breed'] = 'Jersey'
    batch_data[2]['feed_quantity_kg'] = 10.5
    
    batch_result = predictor.predict_batch(batch_data)
    print(f"Batch prediction result: {batch_result}")
    
    # Display model info
    print("\n3. Model information...")
    model_info = predictor.get_model_info()
    if 'error' not in model_info:
        print(f"Number of features: {model_info['feature_count']}")
        print("Model performance:")
        for model_name, metrics in model_info['model_performance'].items():
            print(f"  {model_name}: R² = {metrics['test_r2']:.4f}, RMSE = {metrics['test_rmse']:.2f}L")
    
    print("\n✅ Prediction system test completed!")

if __name__ == "__main__":
    test_predictions()
