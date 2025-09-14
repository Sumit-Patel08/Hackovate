"""
Prediction utilities for Cattle Disease Detection Model (Model 2)
Handles loading trained model and making disease predictions
"""

import pandas as pd
import numpy as np
import joblib
import os
from typing import Dict, List, Optional, Union
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class CattleDiseasePredictor:
    """Cattle Disease Detection and Diagnosis Predictor."""
    
    def __init__(self, model_path: str = "models/cattle_disease_classifier.joblib"):
        """Initialize the predictor with trained model."""
        self.model_path = model_path
        self.model = None
        self.preprocessor = None
        self.label_encoder = None
        self.feature_names = None
        self.class_names = None
        self.artifacts = None
        
        # Load model if path exists
        if os.path.exists(model_path):
            self.load_model()
    
    def load_model(self):
        """Load the trained model and artifacts."""
        try:
            # Load model
            self.model = joblib.load(self.model_path)
            
            # Load preprocessor and label encoder
            base_path = self.model_path.replace('.joblib', '')
            self.preprocessor = joblib.load(f"{base_path}_preprocessor.joblib")
            self.label_encoder = joblib.load(f"{base_path}_label_encoder.joblib")
            
            # Load artifacts
            self.artifacts = joblib.load(f"{base_path}_artifacts.joblib")
            self.feature_names = self.artifacts['feature_names']
            self.class_names = self.artifacts['class_names']
            
            print(f"Model loaded successfully from {self.model_path}")
            print(f"Available disease classes: {self.class_names}")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Please train the model first using train_model.py")
    
    def validate_input(self, data: Dict) -> Dict:
        """Validate and clean input data."""
        validated_data = data.copy()
        
        # Define expected ranges and defaults
        validation_rules = {
            'age_months': {'min': 12, 'max': 200, 'default': 48},
            'weight_kg': {'min': 300, 'max': 1200, 'default': 550},
            'lactation_day': {'min': 0, 'max': 365, 'default': 150},
            'parity': {'min': 1, 'max': 10, 'default': 2},
            'body_temperature': {'min': 36, 'max': 42, 'default': 38.5},
            'heart_rate': {'min': 40, 'max': 120, 'default': 65},
            'respiratory_rate': {'min': 15, 'max': 50, 'default': 28},
            'white_blood_cells': {'min': 2000, 'max': 30000, 'default': 7500},
            'somatic_cell_count': {'min': 10000, 'max': 3000000, 'default': 150000},
            'rumen_ph': {'min': 5.0, 'max': 7.5, 'default': 6.3},
            'rumen_temperature': {'min': 38, 'max': 43, 'default': 40.0},
            'calcium_level': {'min': 7.0, 'max': 12.0, 'default': 10.0},
            'phosphorus_level': {'min': 2.0, 'max': 8.0, 'default': 5.0},
            'protein_level': {'min': 5.0, 'max': 10.0, 'default': 7.0},
            'glucose_level': {'min': 35, 'max': 85, 'default': 60},
            'lameness_score': {'min': 0, 'max': 5, 'default': 1},
            'appetite_score': {'min': 1, 'max': 5, 'default': 4},
            'coat_condition': {'min': 1, 'max': 5, 'default': 4},
            'feed_quantity_kg': {'min': 5, 'max': 25, 'default': 15},
            'feeding_frequency': {'min': 1, 'max': 6, 'default': 3},
            'walking_distance_km': {'min': 0, 'max': 15, 'default': 5},
            'grazing_hours': {'min': 0, 'max': 12, 'default': 7},
            'rumination_hours': {'min': 4, 'max': 12, 'default': 7},
            'resting_hours': {'min': 6, 'max': 16, 'default': 10},
            'temperature': {'min': -10, 'max': 45, 'default': 25},
            'humidity': {'min': 20, 'max': 95, 'default': 65}
        }
        
        # Categorical defaults
        categorical_defaults = {
            'breed': 'Holstein',
            'lactation_stage': 'peak',
            'feed_type': 'mixed',
            'season': 'summer'
        }
        
        # Binary defaults
        binary_defaults = {
            'udder_swelling': 0
        }
        
        warnings = []
        
        # Validate numerical features
        for feature, rules in validation_rules.items():
            if feature in validated_data:
                value = validated_data[feature]
                if pd.isna(value) or value is None:
                    validated_data[feature] = rules['default']
                    warnings.append(f"Missing {feature}, using default: {rules['default']}")
                elif value < rules['min'] or value > rules['max']:
                    validated_data[feature] = max(rules['min'], min(rules['max'], value))
                    warnings.append(f"{feature} out of range, adjusted to: {validated_data[feature]}")
            else:
                validated_data[feature] = rules['default']
                warnings.append(f"Missing {feature}, using default: {rules['default']}")
        
        # Validate categorical features
        valid_categories = {
            'breed': ['Holstein', 'Jersey', 'Guernsey', 'Ayrshire', 'Brown Swiss', 'Simmental'],
            'lactation_stage': ['early', 'peak', 'mid', 'late', 'dry'],
            'feed_type': ['green_fodder', 'dry_fodder', 'concentrates', 'silage', 'mixed'],
            'season': ['spring', 'summer', 'autumn', 'winter']
        }
        
        for feature, default in categorical_defaults.items():
            if feature not in validated_data or validated_data[feature] not in valid_categories[feature]:
                validated_data[feature] = default
                warnings.append(f"Invalid or missing {feature}, using default: {default}")
        
        # Validate binary features
        for feature, default in binary_defaults.items():
            if feature not in validated_data:
                validated_data[feature] = default
            else:
                validated_data[feature] = int(bool(validated_data[feature]))
        
        return validated_data, warnings
    
    def predict_disease(self, cow_data: Dict) -> Dict:
        """Predict disease for a single cow."""
        if self.model is None:
            return {
                'predicted_disease': None,
                'confidence': None,
                'probabilities': None,
                'status': 'error',
                'error': 'Model not loaded. Please train the model first.',
                'timestamp': datetime.now().isoformat()
            }
        
        try:
            # Validate input
            validated_data, warnings = self.validate_input(cow_data)
            
            # Create DataFrame
            df = pd.DataFrame([validated_data])
            
            # Make prediction
            prediction = self.model.predict(df)[0]
            probabilities = self.model.predict_proba(df)[0]
            
            # Get disease name
            predicted_disease = self.label_encoder.inverse_transform([prediction])[0]
            confidence = max(probabilities)
            
            # Create probability dictionary
            prob_dict = {}
            for i, class_name in enumerate(self.class_names):
                prob_dict[class_name] = float(probabilities[i])
            
            # Determine risk level
            risk_level = self._determine_risk_level(predicted_disease, confidence, validated_data)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(predicted_disease, validated_data)
            
            return {
                'predicted_disease': predicted_disease,
                'confidence': float(confidence),
                'probabilities': prob_dict,
                'risk_level': risk_level,
                'recommendations': recommendations,
                'status': 'success',
                'validation_warnings': warnings if warnings else None,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                'predicted_disease': None,
                'confidence': None,
                'probabilities': None,
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def predict_batch(self, cows_data: List[Dict]) -> List[Dict]:
        """Predict diseases for multiple cows."""
        results = []
        for i, cow_data in enumerate(cows_data):
            result = self.predict_disease(cow_data)
            result['cow_index'] = i
            results.append(result)
        return results
    
    def _determine_risk_level(self, disease: str, confidence: float, data: Dict) -> str:
        """Determine risk level based on disease and symptoms."""
        if disease == 'healthy':
            return 'low'
        
        # High risk conditions
        high_risk_conditions = [
            data.get('body_temperature', 38.5) > 40.0,
            data.get('somatic_cell_count', 150000) > 1000000,
            data.get('white_blood_cells', 7500) > 20000,
            data.get('appetite_score', 4) < 2,
            data.get('lameness_score', 1) > 3
        ]
        
        if disease == 'mastitis' and confidence > 0.8:
            return 'high'
        elif any(high_risk_conditions):
            return 'high'
        elif confidence > 0.7:
            return 'medium'
        else:
            return 'low'
    
    def _generate_recommendations(self, disease: str, data: Dict) -> List[str]:
        """Generate treatment and management recommendations."""
        recommendations = []
        
        if disease == 'healthy':
            recommendations.extend([
                "Continue current management practices",
                "Monitor regularly for any changes in health parameters",
                "Maintain good nutrition and hygiene"
            ])
        
        elif disease == 'mastitis':
            recommendations.extend([
                "Immediate veterinary consultation required",
                "Isolate affected cow and milk separately",
                "Implement strict milking hygiene protocols",
                "Consider antibiotic treatment as prescribed by veterinarian",
                "Monitor somatic cell count closely"
            ])
            
            if data.get('somatic_cell_count', 0) > 1000000:
                recommendations.append("Critical: Extremely high somatic cell count - urgent treatment needed")
        
        elif disease == 'digestive_disorder':
            recommendations.extend([
                "Review and adjust feed composition",
                "Ensure adequate fiber in diet",
                "Monitor rumen pH levels",
                "Provide probiotics or rumen buffers",
                "Consult veterinarian for digestive aids"
            ])
            
            if data.get('rumen_ph', 6.3) < 5.5:
                recommendations.append("Critical: Low rumen pH - risk of acidosis")
        
        elif disease == 'mineral_deficiency':
            recommendations.extend([
                "Blood test to confirm specific mineral deficiencies",
                "Supplement diet with appropriate minerals",
                "Review mineral content of feed and water",
                "Consider injectable mineral supplements",
                "Monitor coat condition and appetite"
            ])
            
            if data.get('calcium_level', 10.0) < 8.0:
                recommendations.append("Low calcium detected - risk of milk fever")
        
        elif disease == 'lameness':
            recommendations.extend([
                "Immediate hoof examination and trimming",
                "Provide comfortable, clean bedding",
                "Reduce walking distances",
                "Apply topical treatments as needed",
                "Consider pain management options"
            ])
            
            if data.get('lameness_score', 1) > 3:
                recommendations.append("Severe lameness - immediate veterinary attention required")
        
        # General recommendations based on parameters
        if data.get('body_temperature', 38.5) > 39.5:
            recommendations.append("Elevated temperature detected - monitor for fever")
        
        if data.get('appetite_score', 4) < 3:
            recommendations.append("Poor appetite - investigate underlying causes")
        
        return recommendations
    
    def get_feature_info(self) -> Dict:
        """Get information about model features."""
        if not self.feature_names:
            return {"error": "Model not loaded"}
        
        feature_descriptions = {
            'age_months': 'Age of cow in months (12-200)',
            'weight_kg': 'Weight of cow in kg (300-1200)',
            'lactation_day': 'Days since lactation start (0-365)',
            'parity': 'Number of calvings (1-10)',
            'body_temperature': 'Body temperature in °C (36-42)',
            'heart_rate': 'Heart rate in bpm (40-120)',
            'respiratory_rate': 'Respiratory rate per minute (15-50)',
            'white_blood_cells': 'White blood cell count (2000-30000)',
            'somatic_cell_count': 'Somatic cell count in milk (10000-3000000)',
            'rumen_ph': 'Rumen pH level (5.0-7.5)',
            'rumen_temperature': 'Rumen temperature in °C (38-43)',
            'calcium_level': 'Blood calcium level mg/dL (7.0-12.0)',
            'phosphorus_level': 'Blood phosphorus level mg/dL (2.0-8.0)',
            'protein_level': 'Blood protein level g/dL (5.0-10.0)',
            'glucose_level': 'Blood glucose level mg/dL (35-85)',
            'udder_swelling': 'Presence of udder swelling (0/1)',
            'lameness_score': 'Lameness severity score (0-5)',
            'appetite_score': 'Appetite quality score (1-5)',
            'coat_condition': 'Coat condition score (1-5)',
            'breed': 'Cattle breed',
            'lactation_stage': 'Current lactation stage',
            'feed_type': 'Type of feed provided',
            'feed_quantity_kg': 'Daily feed quantity in kg (5-25)',
            'feeding_frequency': 'Feeding times per day (1-6)',
            'walking_distance_km': 'Daily walking distance in km (0-15)',
            'grazing_hours': 'Daily grazing hours (0-12)',
            'rumination_hours': 'Daily rumination hours (4-12)',
            'resting_hours': 'Daily resting hours (6-16)',
            'temperature': 'Ambient temperature in °C (-10-45)',
            'humidity': 'Relative humidity % (20-95)',
            'season': 'Current season'
        }
        
        return {
            'feature_count': len(self.feature_names),
            'features': self.feature_names,
            'descriptions': feature_descriptions,
            'disease_classes': self.class_names
        }

def create_sample_disease_data() -> Dict:
    """Create sample data for testing disease prediction."""
    return {
        'breed': 'Holstein',
        'age_months': 48.0,
        'weight_kg': 550.0,
        'lactation_stage': 'peak',
        'lactation_day': 120,
        'parity': 3,
        'body_temperature': 39.2,
        'heart_rate': 68.0,
        'respiratory_rate': 28.0,
        'white_blood_cells': 7500.0,
        'somatic_cell_count': 150000.0,
        'rumen_ph': 6.3,
        'rumen_temperature': 40.1,
        'calcium_level': 10.2,
        'phosphorus_level': 5.5,
        'protein_level': 7.2,
        'glucose_level': 62.0,
        'udder_swelling': 0,
        'lameness_score': 1,
        'appetite_score': 4,
        'coat_condition': 4,
        'feed_type': 'mixed',
        'feed_quantity_kg': 16.5,
        'feeding_frequency': 3,
        'walking_distance_km': 5.2,
        'grazing_hours': 7.5,
        'rumination_hours': 7.0,
        'resting_hours': 9.5,
        'temperature': 22.0,
        'humidity': 65.0,
        'season': 'summer'
    }

if __name__ == "__main__":
    # Test the predictor
    predictor = CattleDiseasePredictor()
    
    if predictor.model is not None:
        # Test with sample data
        sample_data = create_sample_disease_data()
        result = predictor.predict_disease(sample_data)
        
        print("Disease Prediction Test:")
        print(f"Predicted Disease: {result['predicted_disease']}")
        print(f"Confidence: {result['confidence']:.3f}")
        print(f"Risk Level: {result['risk_level']}")
        print(f"Status: {result['status']}")
        
        if result['recommendations']:
            print("\nRecommendations:")
            for rec in result['recommendations']:
                print(f"- {rec}")
    else:
        print("Model not found. Please train the model first using train_model.py")
