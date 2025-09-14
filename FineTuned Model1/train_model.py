"""
Model 1: Comprehensive Milk Yield Prediction Training
AI/ML-Based Cattle Milk Yield and Health Prediction System

This script trains a regression model to predict daily milk output per cattle
based on comprehensive input variables including animal data, feed/nutrition,
activity/behavioral data, health data, and environmental conditions.
"""

import pandas as pd
import numpy as np
import os
import joblib
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import xgboost as xgb
import lightgbm as lgb

from data_generator import CattleDataGenerator

class MilkYieldPredictor:
    """Comprehensive milk yield prediction model."""
    
    def __init__(self):
        """Initialize the predictor."""
        self.models = {}
        self.best_model = None
        self.preprocessor = None
        self.feature_names = None
        self.target_name = 'milk_yield_liters'
        
        # Create directories
        os.makedirs('models', exist_ok=True)
        os.makedirs('logs', exist_ok=True)
        os.makedirs('reports', exist_ok=True)
    
    def load_or_generate_data(self, use_existing=True):
        """Load existing data or generate new dataset."""
        data_path = 'data/cattle_milk_yield_dataset.csv'
        
        if use_existing and os.path.exists(data_path):
            print(f"📊 Loading existing dataset from {data_path}")
            df = pd.read_csv(data_path)
            print(f"✅ Loaded {len(df)} records")
        else:
            print("🔄 Generating new comprehensive dataset...")
            generator = CattleDataGenerator()
            
            # Generate larger dataset for better model training
            df = generator.generate_cattle_dataset(n_cows=200, days_per_cow=180)
            generator.save_dataset(df, 'cattle_milk_yield_dataset.csv')
        
        return df
    
    def explore_data(self, df):
        """Perform exploratory data analysis."""
        print("\n📈 Exploratory Data Analysis")
        print("=" * 50)
        
        print(f"Dataset shape: {df.shape}")
        print(f"Date range: {df['date'].min()} to {df['date'].max()}")
        print(f"Number of unique cows: {df['cow_id'].nunique()}")
        
        # Target variable statistics
        print(f"\n🎯 Target Variable (Milk Yield) Statistics:")
        print(f"Mean: {df[self.target_name].mean():.2f} L")
        print(f"Std: {df[self.target_name].std():.2f} L")
        print(f"Min: {df[self.target_name].min():.2f} L")
        print(f"Max: {df[self.target_name].max():.2f} L")
        
        # Missing values
        missing_values = df.isnull().sum()
        if missing_values.sum() > 0:
            print(f"\n⚠️ Missing values found:")
            print(missing_values[missing_values > 0])
        else:
            print("\n✅ No missing values found")
        
        # Create visualizations
        self._create_eda_plots(df)
        
        return df
    
    def _create_eda_plots(self, df):
        """Create exploratory data analysis plots."""
        plt.style.use('default')
        
        # Create subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Cattle Milk Yield - Exploratory Data Analysis', fontsize=16)
        
        # 1. Milk yield distribution
        axes[0, 0].hist(df[self.target_name], bins=50, alpha=0.7, color='skyblue')
        axes[0, 0].set_title('Milk Yield Distribution')
        axes[0, 0].set_xlabel('Milk Yield (Liters)')
        axes[0, 0].set_ylabel('Frequency')
        
        # 2. Milk yield by breed
        breed_yield = df.groupby('breed')[self.target_name].mean().sort_values(ascending=False)
        axes[0, 1].bar(breed_yield.index, breed_yield.values, color='lightcoral')
        axes[0, 1].set_title('Average Milk Yield by Breed')
        axes[0, 1].set_xlabel('Breed')
        axes[0, 1].set_ylabel('Average Milk Yield (L)')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 3. Milk yield by lactation stage
        lactation_yield = df.groupby('lactation_stage')[self.target_name].mean()
        axes[0, 2].bar(lactation_yield.index, lactation_yield.values, color='lightgreen')
        axes[0, 2].set_title('Average Milk Yield by Lactation Stage')
        axes[0, 2].set_xlabel('Lactation Stage')
        axes[0, 2].set_ylabel('Average Milk Yield (L)')
        
        # 4. Temperature vs Milk Yield
        axes[1, 0].scatter(df['temperature'], df[self.target_name], alpha=0.5, color='orange')
        axes[1, 0].set_title('Temperature vs Milk Yield')
        axes[1, 0].set_xlabel('Temperature (°C)')
        axes[1, 0].set_ylabel('Milk Yield (L)')
        
        # 5. Feed quantity vs Milk Yield
        axes[1, 1].scatter(df['feed_quantity_kg'], df[self.target_name], alpha=0.5, color='purple')
        axes[1, 1].set_title('Feed Quantity vs Milk Yield')
        axes[1, 1].set_xlabel('Feed Quantity (kg)')
        axes[1, 1].set_ylabel('Milk Yield (L)')
        
        # 6. Age vs Milk Yield
        axes[1, 2].scatter(df['age_months'], df[self.target_name], alpha=0.5, color='brown')
        axes[1, 2].set_title('Age vs Milk Yield')
        axes[1, 2].set_xlabel('Age (months)')
        axes[1, 2].set_ylabel('Milk Yield (L)')
        
        plt.tight_layout()
        plt.savefig('reports/eda_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✅ EDA plots saved to reports/eda_analysis.png")
    
    def prepare_features(self, df):
        """Prepare features for model training."""
        print("\n🔧 Preparing features for model training...")
        
        # Define feature categories
        categorical_features = [
            'breed', 'lactation_stage', 'feed_type', 'season', 'housing_type'
        ]
        
        numerical_features = [
            'age_months', 'weight_kg', 'lactation_day', 'parity',
            'historical_yield_7d', 'historical_yield_30d',
            'feed_quantity_kg', 'feeding_frequency',
            'walking_distance_km', 'grazing_hours', 'rumination_hours', 'resting_hours',
            'body_temperature', 'heart_rate', 'health_score',
            'temperature', 'humidity', 'ventilation_score', 'cleanliness_score',
            'day_of_year'
        ]
        
        # Ensure all features exist in the dataset
        available_categorical = [f for f in categorical_features if f in df.columns]
        available_numerical = [f for f in numerical_features if f in df.columns]
        
        print(f"Available categorical features: {len(available_categorical)}")
        print(f"Available numerical features: {len(available_numerical)}")
        
        # Create preprocessor
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), available_numerical),
                ('cat', OneHotEncoder(drop='first', sparse_output=False), available_categorical)
            ]
        )
        
        # Prepare feature matrix and target
        feature_columns = available_numerical + available_categorical
        X = df[feature_columns].copy()
        y = df[self.target_name].copy()
        
        # Store feature information
        self.feature_names = feature_columns
        
        print(f"✅ Feature preparation completed")
        print(f"Feature matrix shape: {X.shape}")
        print(f"Target vector shape: {y.shape}")
        
        return X, y
    
    def train_models(self, X, y):
        """Train multiple models and select the best one."""
        print("\n🚀 Training multiple regression models...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=None
        )
        
        print(f"Training set: {X_train.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")
        
        # Define models to train
        models_to_train = {
            'Linear Regression': Pipeline([
                ('preprocessor', self.preprocessor),
                ('regressor', LinearRegression())
            ]),
            'Ridge Regression': Pipeline([
                ('preprocessor', self.preprocessor),
                ('regressor', Ridge(alpha=1.0))
            ]),
            'Random Forest': Pipeline([
                ('preprocessor', self.preprocessor),
                ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
            ]),
            'Gradient Boosting': Pipeline([
                ('preprocessor', self.preprocessor),
                ('regressor', GradientBoostingRegressor(n_estimators=100, random_state=42))
            ]),
            'XGBoost': Pipeline([
                ('preprocessor', self.preprocessor),
                ('regressor', xgb.XGBRegressor(n_estimators=100, random_state=42))
            ]),
            'LightGBM': Pipeline([
                ('preprocessor', self.preprocessor),
                ('regressor', lgb.LGBMRegressor(n_estimators=100, random_state=42, verbose=-1))
            ])
        }
        
        # Train and evaluate models
        model_results = {}
        
        for name, model in models_to_train.items():
            print(f"\n📊 Training {name}...")
            
            # Train model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
            
            # Calculate metrics
            train_r2 = r2_score(y_train, y_pred_train)
            test_r2 = r2_score(y_test, y_pred_test)
            train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
            test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
            train_mae = mean_absolute_error(y_train, y_pred_train)
            test_mae = mean_absolute_error(y_test, y_pred_test)
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
            
            results = {
                'model': model,
                'train_r2': train_r2,
                'test_r2': test_r2,
                'train_rmse': train_rmse,
                'test_rmse': test_rmse,
                'train_mae': train_mae,
                'test_mae': test_mae,
                'cv_r2_mean': cv_scores.mean(),
                'cv_r2_std': cv_scores.std(),
                'predictions_test': y_pred_test
            }
            
            model_results[name] = results
            
            print(f"  Train R²: {train_r2:.4f}")
            print(f"  Test R²: {test_r2:.4f}")
            print(f"  Test RMSE: {test_rmse:.4f}")
            print(f"  CV R² (mean±std): {cv_scores.mean():.4f}±{cv_scores.std():.4f}")
        
        # Select best model based on test R²
        best_model_name = max(model_results.keys(), key=lambda k: model_results[k]['test_r2'])
        self.best_model = model_results[best_model_name]['model']
        
        print(f"\n🏆 Best model: {best_model_name}")
        print(f"Test R²: {model_results[best_model_name]['test_r2']:.4f}")
        
        # Store all results
        self.models = model_results
        
        # Create comparison plot
        self._create_model_comparison_plot(model_results, y_test)
        
        return X_test, y_test
    
    def _create_model_comparison_plot(self, model_results, y_test):
        """Create model comparison visualization."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Model Performance Comparison', fontsize=16)
        
        # 1. R² Score comparison
        models = list(model_results.keys())
        test_r2_scores = [model_results[m]['test_r2'] for m in models]
        
        axes[0, 0].bar(models, test_r2_scores, color='skyblue')
        axes[0, 0].set_title('Test R² Score by Model')
        axes[0, 0].set_ylabel('R² Score')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. RMSE comparison
        test_rmse_scores = [model_results[m]['test_rmse'] for m in models]
        axes[0, 1].bar(models, test_rmse_scores, color='lightcoral')
        axes[0, 1].set_title('Test RMSE by Model')
        axes[0, 1].set_ylabel('RMSE')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 3. Actual vs Predicted for best model
        best_model_name = max(models, key=lambda k: model_results[k]['test_r2'])
        y_pred_best = model_results[best_model_name]['predictions_test']
        
        axes[1, 0].scatter(y_test, y_pred_best, alpha=0.6)
        axes[1, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        axes[1, 0].set_title(f'Actual vs Predicted - {best_model_name}')
        axes[1, 0].set_xlabel('Actual Milk Yield (L)')
        axes[1, 0].set_ylabel('Predicted Milk Yield (L)')
        
        # 4. Residuals plot for best model
        residuals = y_test - y_pred_best
        axes[1, 1].scatter(y_pred_best, residuals, alpha=0.6)
        axes[1, 1].axhline(y=0, color='r', linestyle='--')
        axes[1, 1].set_title(f'Residuals Plot - {best_model_name}')
        axes[1, 1].set_xlabel('Predicted Milk Yield (L)')
        axes[1, 1].set_ylabel('Residuals')
        
        plt.tight_layout()
        plt.savefig('reports/model_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✅ Model comparison plots saved to reports/model_comparison.png")
    
    def save_model(self):
        """Save the best trained model and related artifacts."""
        print("\n💾 Saving model and artifacts...")
        
        # Save best model
        joblib.dump(self.best_model, 'models/best_milk_yield_model.pkl')
        
        # Save feature information
        joblib.dump(self.feature_names, 'models/feature_names.pkl')
        
        # Save model results summary
        results_summary = {}
        for name, results in self.models.items():
            results_summary[name] = {
                'test_r2': results['test_r2'],
                'test_rmse': results['test_rmse'],
                'test_mae': results['test_mae'],
                'cv_r2_mean': results['cv_r2_mean'],
                'cv_r2_std': results['cv_r2_std']
            }
        
        joblib.dump(results_summary, 'models/model_results.pkl')
        
        print("✅ Model artifacts saved successfully!")
        print("  - models/best_milk_yield_model.pkl")
        print("  - models/feature_names.pkl")
        print("  - models/model_results.pkl")
    
    def generate_training_report(self):
        """Generate comprehensive training report."""
        print("\n📋 Generating training report...")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = f"reports/training_report_{timestamp}.txt"
        
        with open(report_path, 'w') as f:
            f.write("AI/ML-Based Cattle Milk Yield Prediction - Training Report\n")
            f.write("=" * 60 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("MODEL PERFORMANCE SUMMARY\n")
            f.write("-" * 30 + "\n")
            
            for name, results in self.models.items():
                f.write(f"\n{name}:\n")
                f.write(f"  Test R²: {results['test_r2']:.4f}\n")
                f.write(f"  Test RMSE: {results['test_rmse']:.4f} L\n")
                f.write(f"  Test MAE: {results['test_mae']:.4f} L\n")
                f.write(f"  CV R² (mean±std): {results['cv_r2_mean']:.4f}±{results['cv_r2_std']:.4f}\n")
            
            best_model_name = max(self.models.keys(), key=lambda k: self.models[k]['test_r2'])
            f.write(f"\nBEST MODEL: {best_model_name}\n")
            f.write(f"Test R²: {self.models[best_model_name]['test_r2']:.4f}\n")
            
            f.write(f"\nFEATURES USED ({len(self.feature_names)}):\n")
            for i, feature in enumerate(self.feature_names, 1):
                f.write(f"{i:2d}. {feature}\n")
        
        print(f"✅ Training report saved to {report_path}")

def main():
    """Main training pipeline."""
    print("🐄 AI/ML-Based Cattle Milk Yield Prediction - Model 1 Training")
    print("=" * 70)
    
    # Initialize predictor
    predictor = MilkYieldPredictor()
    
    # Load or generate data
    df = predictor.load_or_generate_data(use_existing=False)  # Generate new data
    
    # Explore data
    df = predictor.explore_data(df)
    
    # Prepare features
    X, y = predictor.prepare_features(df)
    
    # Train models
    X_test, y_test = predictor.train_models(X, y)
    
    # Save model
    predictor.save_model()
    
    # Generate report
    predictor.generate_training_report()
    
    print("\n🎉 Model 1 training completed successfully!")
    print("\nNext steps:")
    print("1. Run prediction tests: python predict.py")
    print("2. Start API server: python run_fastapi.py")
    print("3. Launch dashboard: python run_streamlit.py")

if __name__ == "__main__":
    main()
