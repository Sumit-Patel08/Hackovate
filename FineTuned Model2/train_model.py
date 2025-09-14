"""
Training Pipeline for Cattle Disease Detection Model (Model 2)
Uses ensemble methods and feature engineering for disease classification
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.metrics import roc_auc_score, precision_recall_curve, roc_curve
import joblib
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from data_generator import CattleDiseaseDataGenerator, create_sample_disease_data

class CattleDiseaseClassifier:
    """Cattle Disease Detection and Classification Model."""
    
    def __init__(self):
        """Initialize the classifier."""
        self.model = None
        self.preprocessor = None
        self.label_encoder = None
        self.feature_names = None
        self.class_names = None
        self.training_report = {}
        
        # Define feature categories
        self.numerical_features = [
            'age_months', 'weight_kg', 'lactation_day', 'parity',
            'body_temperature', 'heart_rate', 'respiratory_rate',
            'white_blood_cells', 'somatic_cell_count', 'rumen_ph', 'rumen_temperature',
            'calcium_level', 'phosphorus_level', 'protein_level', 'glucose_level',
            'lameness_score', 'appetite_score', 'coat_condition',
            'feed_quantity_kg', 'feeding_frequency',
            'walking_distance_km', 'grazing_hours', 'rumination_hours', 'resting_hours',
            'temperature', 'humidity', 'milk_yield'
        ]
        
        self.categorical_features = [
            'breed', 'lactation_stage', 'feed_type', 'season'
        ]
        
        self.binary_features = ['udder_swelling']
    
    def load_or_generate_data(self, data_path: str = None) -> pd.DataFrame:
        """Load existing data or generate new dataset."""
        if data_path and os.path.exists(data_path):
            print(f"Loading existing dataset from {data_path}")
            df = pd.read_csv(data_path)
        else:
            print("Generating new cattle disease dataset...")
            generator = CattleDiseaseDataGenerator()
            df = generator.generate_dataset(1000)  # Reduced for faster training
            
            # Save the generated dataset
            os.makedirs("data", exist_ok=True)
            save_path = "data/cattle_disease_dataset.csv"
            df.to_csv(save_path, index=False)
            print(f"Dataset saved to {save_path}")
        
        return df
    
    def exploratory_data_analysis(self, df: pd.DataFrame):
        """Perform exploratory data analysis."""
        print("\n" + "="*50)
        print("EXPLORATORY DATA ANALYSIS")
        print("="*50)
        
        print(f"\nDataset shape: {df.shape}")
        print(f"\nDisease distribution:")
        disease_counts = df['disease_status'].value_counts()
        print(disease_counts)
        
        # Create visualizations
        plt.figure(figsize=(20, 15))
        
        # Disease distribution
        plt.subplot(3, 4, 1)
        disease_counts.plot(kind='bar')
        plt.title('Disease Distribution')
        plt.xticks(rotation=45)
        
        # Key health parameters by disease
        health_params = ['body_temperature', 'heart_rate', 'somatic_cell_count', 'white_blood_cells']
        
        for i, param in enumerate(health_params, 2):
            plt.subplot(3, 4, i)
            df.boxplot(column=param, by='disease_status', ax=plt.gca())
            plt.title(f'{param} by Disease')
            plt.xticks(rotation=45)
        
        # Correlation heatmap for numerical features
        plt.subplot(3, 4, 6)
        numerical_df = df[self.numerical_features[:10]]  # Top 10 for visibility
        correlation_matrix = numerical_df.corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
        plt.title('Feature Correlation Heatmap')
        
        # Milk yield vs disease
        plt.subplot(3, 4, 7)
        df.boxplot(column='milk_yield', by='disease_status', ax=plt.gca())
        plt.title('Milk Yield by Disease Status')
        plt.xticks(rotation=45)
        
        # Age distribution by disease
        plt.subplot(3, 4, 8)
        for disease in df['disease_status'].unique():
            subset = df[df['disease_status'] == disease]
            plt.hist(subset['age_months'], alpha=0.5, label=disease, bins=20)
        plt.title('Age Distribution by Disease')
        plt.xlabel('Age (months)')
        plt.legend()
        
        # Lactation stage distribution
        plt.subplot(3, 4, 9)
        lactation_disease = pd.crosstab(df['lactation_stage'], df['disease_status'])
        lactation_disease.plot(kind='bar', stacked=True, ax=plt.gca())
        plt.title('Disease by Lactation Stage')
        plt.xticks(rotation=45)
        
        # Breed distribution
        plt.subplot(3, 4, 10)
        breed_disease = pd.crosstab(df['breed'], df['disease_status'])
        breed_disease.plot(kind='bar', stacked=True, ax=plt.gca())
        plt.title('Disease by Breed')
        plt.xticks(rotation=45)
        
        # Seasonal patterns
        plt.subplot(3, 4, 11)
        season_disease = pd.crosstab(df['season'], df['disease_status'])
        season_disease.plot(kind='bar', ax=plt.gca())
        plt.title('Disease by Season')
        plt.xticks(rotation=45)
        
        # Feature importance preview (using simple correlation)
        plt.subplot(3, 4, 12)
        # Calculate correlation with disease (encoded)
        df_encoded = df.copy()
        le_temp = LabelEncoder()
        df_encoded['disease_encoded'] = le_temp.fit_transform(df['disease_status'])
        
        correlations = []
        for feature in self.numerical_features[:10]:
            if feature in df_encoded.columns:
                corr = abs(df_encoded[feature].corr(df_encoded['disease_encoded']))
                correlations.append((feature, corr))
        
        correlations.sort(key=lambda x: x[1], reverse=True)
        features, corr_values = zip(*correlations)
        
        plt.barh(range(len(features)), corr_values)
        plt.yticks(range(len(features)), features)
        plt.title('Feature Correlation with Disease')
        plt.xlabel('Absolute Correlation')
        
        plt.tight_layout()
        
        # Save plots
        os.makedirs("plots", exist_ok=True)
        plt.savefig("plots/disease_eda.png", dpi=300, bbox_inches='tight')
        print(f"\nEDA plots saved to plots/disease_eda.png")
        
        plt.close('all')  # Close all figures to free memory
        
        # Statistical summary
        print(f"\nStatistical Summary:")
        print(f"Missing values: {df.isnull().sum().sum()}")
        print(f"Duplicate rows: {df.duplicated().sum()}")
        
        return df
    
    def prepare_features(self, df: pd.DataFrame) -> tuple:
        """Prepare features for training."""
        print("\n" + "="*50)
        print("FEATURE PREPARATION")
        print("="*50)
        
        # Separate features and target
        feature_columns = self.numerical_features + self.categorical_features + self.binary_features
        
        # Remove target-related features that shouldn't be used for prediction
        if 'milk_yield' in feature_columns:
            feature_columns.remove('milk_yield')  # This might be too predictive
        
        X = df[feature_columns].copy()
        y = df['disease_status'].copy()
        
        print(f"Features shape: {X.shape}")
        print(f"Target shape: {y.shape}")
        print(f"Feature columns: {feature_columns}")
        
        # Handle missing values
        for col in self.numerical_features:
            if col in X.columns:
                X[col] = X[col].fillna(X[col].median())
        
        for col in self.categorical_features:
            if col in X.columns:
                X[col] = X[col].fillna(X[col].mode()[0])
        
        # Create preprocessor
        numerical_transformer = StandardScaler()
        categorical_transformer = OneHotEncoder(drop='first', sparse_output=False)
        
        # Update feature lists to only include existing columns
        existing_numerical = [f for f in self.numerical_features if f in X.columns and f != 'milk_yield']
        existing_categorical = [f for f in self.categorical_features if f in X.columns]
        existing_binary = [f for f in self.binary_features if f in X.columns]
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, existing_numerical),
                ('cat', categorical_transformer, existing_categorical),
                ('bin', 'passthrough', existing_binary)
            ]
        )
        
        # Encode target variable
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
        
        self.preprocessor = preprocessor
        self.label_encoder = label_encoder
        self.class_names = label_encoder.classes_
        
        print(f"Classes: {self.class_names}")
        print(f"Class distribution: {np.bincount(y_encoded)}")
        
        return X, y_encoded
    
    def train_models(self, X: pd.DataFrame, y: np.ndarray) -> dict:
        """Train multiple models and select the best one."""
        print("\n" + "="*50)
        print("MODEL TRAINING")
        print("="*50)
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"Training set size: {X_train.shape[0]}")
        print(f"Test set size: {X_test.shape[0]}")
        
        # Define models to try
        models = {
            'Random Forest': RandomForestClassifier(
                n_estimators=50,  # Reduced from 100
                max_depth=10,  # Reduced complexity
                min_samples_split=10,  # Increased for faster training
                min_samples_leaf=5,  # Increased for faster training
                random_state=42,
                n_jobs=-1,
                verbose=1  # Show progress
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=50,  # Reduced from 100
                learning_rate=0.2,  # Increased for faster convergence
                max_depth=4,  # Reduced complexity
                random_state=42,
                verbose=1  # Show progress
            ),
            'Logistic Regression': LogisticRegression(
                random_state=42,
                max_iter=1000,
                multi_class='ovr'
            ),
            'SVM': SVC(
                kernel='rbf',
                probability=True,
                random_state=42
            )
        }
        
        # Train and evaluate models
        results = {}
        trained_models = {}
        
        for name, model in models.items():
            print(f"\nTraining {name}...")
            print(f"Training set size: {X_train.shape[0]} samples")
            
            # Create pipeline
            pipeline = Pipeline([
                ('preprocessor', self.preprocessor),
                ('classifier', model)
            ])
            
            # Train model with progress indication
            import time
            start_time = time.time()
            pipeline.fit(X_train, y_train)
            end_time = time.time()
            print(f"Training completed in {end_time - start_time:.2f} seconds")
            
            # Predictions
            y_pred = pipeline.predict(X_test)
            y_pred_proba = pipeline.predict_proba(X_test)
            
            # Evaluate
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')
            
            # Cross-validation (reduced folds for speed)
            print(f"Running cross-validation for {name}...")
            cv_scores = cross_val_score(pipeline, X_train, y_train, cv=3, scoring='f1_weighted')  # Reduced from 5 to 3
            
            results[name] = {
                'accuracy': accuracy,
                'f1_score': f1,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'predictions': y_pred,
                'probabilities': y_pred_proba
            }
            
            trained_models[name] = pipeline
            
            print(f"Accuracy: {accuracy:.4f}")
            print(f"F1 Score: {f1:.4f}")
            print(f"CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        # Select best model based on F1 score
        best_model_name = max(results.keys(), key=lambda k: results[k]['f1_score'])
        best_model = trained_models[best_model_name]
        
        print(f"\nBest model: {best_model_name}")
        print(f"Best F1 Score: {results[best_model_name]['f1_score']:.4f}")
        
        # Create ensemble model
        print(f"\nCreating ensemble model...")
        ensemble = VotingClassifier(
            estimators=[
                ('rf', models['Random Forest']),
                ('gb', models['Gradient Boosting']),
                ('lr', models['Logistic Regression'])
            ],
            voting='soft'
        )
        
        ensemble_pipeline = Pipeline([
            ('preprocessor', self.preprocessor),
            ('classifier', ensemble)
        ])
        
        ensemble_pipeline.fit(X_train, y_train)
        ensemble_pred = ensemble_pipeline.predict(X_test)
        ensemble_f1 = f1_score(y_test, ensemble_pred, average='weighted')
        
        print(f"Ensemble F1 Score: {ensemble_f1:.4f}")
        
        # Choose final model (ensemble if better, otherwise best individual)
        if ensemble_f1 > results[best_model_name]['f1_score']:
            self.model = ensemble_pipeline
            final_model_name = "Ensemble"
            final_f1 = ensemble_f1
            final_pred = ensemble_pred
        else:
            self.model = best_model
            final_model_name = best_model_name
            final_f1 = results[best_model_name]['f1_score']
            final_pred = results[best_model_name]['predictions']
        
        # Store training results
        self.training_report = {
            'best_model': final_model_name,
            'best_f1_score': final_f1,
            'all_results': results,
            'test_predictions': final_pred,
            'test_actual': y_test,
            'training_date': datetime.now().isoformat()
        }
        
        # Generate detailed classification report
        print(f"\nFinal Model: {final_model_name}")
        print("\nClassification Report:")
        print(classification_report(y_test, final_pred, target_names=self.class_names))
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, final_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.class_names, yticklabels=self.class_names)
        plt.title(f'Confusion Matrix - {final_model_name}')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        
        os.makedirs("plots", exist_ok=True)
        plt.savefig("plots/confusion_matrix.png", dpi=300, bbox_inches='tight')
        plt.close('all')  # Close all figures
        
        return results
    
    def save_model(self, model_name: str = "cattle_disease_model"):
        """Save the trained model and artifacts."""
        print(f"\nSaving model artifacts...")
        
        os.makedirs("models", exist_ok=True)
        
        # Save model
        model_path = f"models/{model_name}.joblib"
        joblib.dump(self.model, model_path)
        
        # Save preprocessor and label encoder
        joblib.dump(self.preprocessor, f"models/{model_name}_preprocessor.joblib")
        joblib.dump(self.label_encoder, f"models/{model_name}_label_encoder.joblib")
        
        # Save feature names and class names
        artifacts = {
            'feature_names': self.numerical_features + self.categorical_features + self.binary_features,
            'class_names': self.class_names.tolist(),
            'training_report': self.training_report
        }
        
        joblib.dump(artifacts, f"models/{model_name}_artifacts.joblib")
        
        print(f"Model saved to: {model_path}")
        print(f"Artifacts saved to: models/{model_name}_artifacts.joblib")
        
        return model_path
    
    def generate_training_report(self):
        """Generate comprehensive training report."""
        print("\n" + "="*50)
        print("TRAINING REPORT")
        print("="*50)
        
        report = f"""
Cattle Disease Detection Model Training Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Model Performance:
- Best Model: {self.training_report['best_model']}
- Best F1 Score: {self.training_report['best_f1_score']:.4f}

All Model Results:
"""
        
        for model_name, results in self.training_report['all_results'].items():
            report += f"""
{model_name}:
  - Accuracy: {results['accuracy']:.4f}
  - F1 Score: {results['f1_score']:.4f}
  - CV Mean: {results['cv_mean']:.4f}
  - CV Std: {results['cv_std']:.4f}
"""
        
        report += f"""
Classes: {', '.join(self.class_names)}

Feature Categories:
- Numerical Features: {len([f for f in self.numerical_features if f != 'milk_yield'])}
- Categorical Features: {len(self.categorical_features)}
- Binary Features: {len(self.binary_features)}

Notes:
- Model trained for cattle disease detection and classification
- Focuses on mastitis, digestive disorders, mineral deficiency, lameness detection
- Uses ensemble methods for robust predictions
- Includes comprehensive health parameters and environmental factors
"""
        
        # Save report
        os.makedirs("reports", exist_ok=True)
        report_path = "reports/disease_training_report.txt"
        with open(report_path, 'w') as f:
            f.write(report)
        
        print(report)
        print(f"\nReport saved to: {report_path}")

def main():
    """Main training pipeline."""
    print("Starting Cattle Disease Detection Model Training...")
    
    # Initialize classifier
    classifier = CattleDiseaseClassifier()
    
    # Load or generate data
    df = classifier.load_or_generate_data("data/cattle_disease_dataset.csv")
    
    # Exploratory data analysis
    df = classifier.exploratory_data_analysis(df)
    
    # Prepare features
    X, y = classifier.prepare_features(df)
    
    # Train models
    results = classifier.train_models(X, y)
    
    # Save model
    model_path = classifier.save_model("cattle_disease_classifier")
    
    # Generate report
    classifier.generate_training_report()
    
    print(f"\n" + "="*50)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("="*50)
    print(f"Model saved to: {model_path}")
    print(f"Ready for disease prediction and diagnosis!")

if __name__ == "__main__":
    main()
