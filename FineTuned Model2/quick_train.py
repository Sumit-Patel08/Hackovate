"""
Quick Training Script for Cattle Disease Detection Model
Optimized for faster training with reduced complexity
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, f1_score
import joblib
import os
import time
from datetime import datetime

def quick_train():
    """Quick training with optimized parameters."""
    print("Starting Quick Training for Cattle Disease Detection...")
    
    # Load existing dataset
    data_path = "data/cattle_disease_dataset.csv"
    if not os.path.exists(data_path):
        print(f"Dataset not found at {data_path}")
        print("Please run the full training script first to generate data.")
        return
    
    print(f"Loading dataset from {data_path}")
    df = pd.read_csv(data_path)
    
    # Use smaller subset for quick training
    df_sample = df.sample(n=min(500, len(df)), random_state=42)
    print(f"Using sample of {len(df_sample)} records for quick training")
    
    # Define features
    numerical_features = [
        'age_months', 'weight_kg', 'lactation_day', 'parity',
        'body_temperature', 'heart_rate', 'respiratory_rate',
        'white_blood_cells', 'somatic_cell_count', 'rumen_ph', 'rumen_temperature',
        'calcium_level', 'phosphorus_level', 'protein_level', 'glucose_level',
        'lameness_score', 'appetite_score', 'coat_condition',
        'feed_quantity_kg', 'feeding_frequency',
        'walking_distance_km', 'grazing_hours', 'rumination_hours', 'resting_hours',
        'temperature', 'humidity'
    ]
    
    categorical_features = ['breed', 'lactation_stage', 'feed_type', 'season']
    binary_features = ['udder_swelling']
    
    # Prepare features
    feature_columns = numerical_features + categorical_features + binary_features
    existing_features = [f for f in feature_columns if f in df_sample.columns]
    
    X = df_sample[existing_features].copy()
    y = df_sample['disease_status'].copy()
    
    print(f"Features: {len(existing_features)}")
    print(f"Samples: {len(X)}")
    print(f"Classes: {y.unique()}")
    
    # Handle missing values
    for col in numerical_features:
        if col in X.columns:
            X[col] = X[col].fillna(X[col].median())
    
    for col in categorical_features:
        if col in X.columns:
            X[col] = X[col].fillna(X[col].mode()[0] if len(X[col].mode()) > 0 else 'unknown')
    
    # Create preprocessor
    existing_numerical = [f for f in numerical_features if f in X.columns]
    existing_categorical = [f for f in categorical_features if f in X.columns]
    existing_binary = [f for f in binary_features if f in X.columns]
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), existing_numerical),
            ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), existing_categorical),
            ('bin', 'passthrough', existing_binary)
        ]
    )
    
    # Encode target
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    print(f"Training set: {len(X_train)}")
    print(f"Test set: {len(X_test)}")
    
    # Quick models with minimal parameters
    models = {
        'Quick Random Forest': RandomForestClassifier(
            n_estimators=20,  # Very small for speed
            max_depth=5,
            random_state=42,
            n_jobs=-1
        ),
        'Quick Gradient Boosting': GradientBoostingClassifier(
            n_estimators=20,  # Very small for speed
            learning_rate=0.3,  # Higher for faster convergence
            max_depth=3,  # Shallow trees
            random_state=42
        )
    }
    
    best_model = None
    best_score = 0
    best_name = ""
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        # Create pipeline
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])
        
        # Train with timing
        start_time = time.time()
        pipeline.fit(X_train, y_train)
        end_time = time.time()
        
        # Evaluate
        y_pred = pipeline.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        print(f"Training time: {end_time - start_time:.2f} seconds")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1 Score: {f1:.4f}")
        
        if f1 > best_score:
            best_score = f1
            best_model = pipeline
            best_name = name
    
    print(f"\nBest model: {best_name} (F1: {best_score:.4f})")
    
    # Save the quick model
    os.makedirs("models", exist_ok=True)
    model_path = "models/cattle_disease_classifier.joblib"
    joblib.dump(best_model, model_path)
    
    # Save artifacts
    joblib.dump(preprocessor, "models/cattle_disease_classifier_preprocessor.joblib")
    joblib.dump(label_encoder, "models/cattle_disease_classifier_label_encoder.joblib")
    
    artifacts = {
        'feature_names': existing_features,
        'class_names': label_encoder.classes_.tolist(),
        'training_report': {
            'model_type': best_name,
            'f1_score': best_score,
            'training_date': datetime.now().isoformat(),
            'samples_used': len(X_train)
        }
    }
    
    joblib.dump(artifacts, "models/cattle_disease_classifier_artifacts.joblib")
    
    print(f"\nQuick model saved to: {model_path}")
    print("Model is ready for use!")
    
    # Test prediction
    print("\nTesting prediction...")
    sample_data = X_test.iloc[0:1]
    prediction = best_model.predict(sample_data)
    probabilities = best_model.predict_proba(sample_data)
    
    predicted_class = label_encoder.inverse_transform(prediction)[0]
    print(f"Sample prediction: {predicted_class}")
    print(f"Confidence: {probabilities.max():.3f}")
    
    return model_path

if __name__ == "__main__":
    quick_train()
