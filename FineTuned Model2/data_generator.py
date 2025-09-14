"""
Synthetic Cattle Disease Dataset Generator for Model 2
Generates realistic cattle health data with disease classifications
"""

import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import os

class CattleDiseaseDataGenerator:
    """Generate synthetic cattle disease detection dataset."""
    
    def __init__(self, seed: int = 42):
        """Initialize the generator with random seed."""
        np.random.seed(seed)
        random.seed(seed)
        
        # Disease probabilities and characteristics - BALANCED DATASET
        self.diseases = {
            'healthy': {
                'probability': 0.20,  # 20% healthy cows (balanced)
                'milk_yield_factor': (0.9, 1.1),
                'temperature_range': (38.0, 39.0),
                'heart_rate_range': (50, 70),
                'respiratory_rate_range': (18, 30),
                'rumen_ph_range': (6.2, 6.8),
                'white_blood_cells': (4000, 10000),
                'somatic_cell_count': (50000, 200000)
            },
            'mastitis': {
                'probability': 0.20,  # 20% mastitis cases
                'milk_yield_factor': (0.4, 0.8),
                'temperature_range': (39.0, 42.0),
                'heart_rate_range': (70, 120),
                'respiratory_rate_range': (30, 50),
                'rumen_ph_range': (5.8, 6.4),
                'white_blood_cells': (12000, 25000),
                'somatic_cell_count': (400000, 2000000)
            },
            'digestive_disorder': {
                'probability': 0.20,  # 20% digestive issues
                'milk_yield_factor': (0.5, 0.8),
                'temperature_range': (38.5, 41.0),
                'heart_rate_range': (65, 100),
                'respiratory_rate_range': (25, 45),
                'rumen_ph_range': (5.2, 6.0),
                'white_blood_cells': (8000, 15000),
                'somatic_cell_count': (100000, 400000)
            },
            'mineral_deficiency': {
                'probability': 0.20,  # 20% mineral deficiency
                'milk_yield_factor': (0.6, 0.9),
                'temperature_range': (37.8, 40.5),
                'heart_rate_range': (60, 95),
                'respiratory_rate_range': (22, 42),
                'rumen_ph_range': (5.9, 6.6),
                'white_blood_cells': (5000, 12000),
                'somatic_cell_count': (80000, 300000)
            },
            'lameness': {
                'probability': 0.20,  # 20% lameness cases
                'milk_yield_factor': (0.7, 0.9),
                'temperature_range': (38.2, 41.5),
                'heart_rate_range': (70, 110),
                'respiratory_rate_range': (25, 45),
                'rumen_ph_range': (6.0, 6.7),
                'white_blood_cells': (6000, 14000),
                'somatic_cell_count': (100000, 350000)
            }
        }
        
        # Breed characteristics
        self.breeds = ['Holstein', 'Jersey', 'Guernsey', 'Ayrshire', 'Brown Swiss', 'Simmental']
        self.lactation_stages = ['early', 'peak', 'mid', 'late', 'dry']
        self.feed_types = ['green_fodder', 'dry_fodder', 'concentrates', 'silage', 'mixed']
        self.seasons = ['spring', 'summer', 'autumn', 'winter']
        
    def generate_cow_data(self, cow_id: int) -> Dict:
        """Generate data for a single cow."""
        
        # Basic cow characteristics
        breed = random.choice(self.breeds)
        age_months = np.random.normal(48, 18)
        age_months = max(12, min(200, age_months))
        
        # Weight based on breed and age
        base_weights = {
            'Holstein': 550, 'Jersey': 400, 'Guernsey': 450,
            'Ayrshire': 500, 'Brown Swiss': 600, 'Simmental': 650
        }
        weight_kg = np.random.normal(base_weights[breed], 50)
        weight_kg = max(300, min(1200, weight_kg))
        
        lactation_stage = random.choice(self.lactation_stages)
        lactation_day = np.random.randint(1, 365) if lactation_stage != 'dry' else 0
        parity = np.random.randint(1, 8)
        
        # Determine disease status
        disease_probs = [self.diseases[d]['probability'] for d in self.diseases.keys()]
        disease = np.random.choice(list(self.diseases.keys()), p=disease_probs)
        
        # Generate health parameters based on disease
        disease_params = self.diseases[disease]
        
        # Vital signs
        body_temperature = np.random.uniform(*disease_params['temperature_range'])
        heart_rate = np.random.uniform(*disease_params['heart_rate_range'])
        respiratory_rate = np.random.uniform(*disease_params['respiratory_rate_range'])
        
        # Blood parameters
        white_blood_cells = np.random.uniform(*disease_params['white_blood_cells'])
        somatic_cell_count = np.random.uniform(*disease_params['somatic_cell_count'])
        
        # Rumen health
        rumen_ph = np.random.uniform(*disease_params['rumen_ph_range'])
        rumen_temperature = body_temperature + np.random.uniform(0.5, 1.5)
        
        # Feed and nutrition
        feed_type = random.choice(self.feed_types)
        feed_quantity_kg = np.random.uniform(12, 20)
        feeding_frequency = np.random.randint(2, 5)
        
        # Activity parameters (affected by disease)
        activity_factor = 1.0
        if disease in ['lameness', 'digestive_disorder']:
            activity_factor = 0.6
        elif disease == 'mastitis':
            activity_factor = 0.8
        elif disease == 'mineral_deficiency':
            activity_factor = 0.9
            
        walking_distance_km = np.random.uniform(2, 8) * activity_factor
        grazing_hours = np.random.uniform(4, 10) * activity_factor
        rumination_hours = np.random.uniform(5, 9) * activity_factor
        resting_hours = 24 - grazing_hours - rumination_hours - np.random.uniform(2, 4)
        
        # Environmental factors
        season = random.choice(self.seasons)
        temperature = np.random.uniform(15, 35)
        humidity = np.random.uniform(40, 85)
        
        # Milk yield (affected by disease)
        base_yield = 25.0  # Base yield for healthy cow
        if lactation_stage == 'early':
            base_yield *= 0.8
        elif lactation_stage == 'peak':
            base_yield *= 1.2
        elif lactation_stage == 'mid':
            base_yield *= 1.0
        elif lactation_stage == 'late':
            base_yield *= 0.7
        else:  # dry
            base_yield = 0
            
        # Apply disease effect on milk yield
        yield_factor = np.random.uniform(*disease_params['milk_yield_factor'])
        milk_yield = base_yield * yield_factor
        
        # Physical examination based on disease - MORE DISTINCT PATTERNS
        if disease == 'mastitis':
            udder_swelling = 1
            lameness_score = np.random.randint(0, 2)
            appetite_score = np.random.randint(1, 3)  # Poor appetite
            coat_condition = np.random.randint(2, 4)  # Poor coat
            # Additional mastitis indicators
            calcium_level = np.random.uniform(7.0, 9.5)  # Lower calcium
            phosphorus_level = np.random.uniform(3.0, 6.0)
            protein_level = np.random.uniform(5.5, 8.0)
            glucose_level = np.random.uniform(45, 70)
        elif disease == 'lameness':
            udder_swelling = 0
            lameness_score = np.random.randint(3, 6)  # High lameness
            appetite_score = np.random.randint(2, 4)
            coat_condition = np.random.randint(2, 4)
            # Normal blood chemistry for lameness
            calcium_level = np.random.uniform(8.5, 11.0)
            phosphorus_level = np.random.uniform(4.0, 7.0)
            protein_level = np.random.uniform(6.0, 8.5)
            glucose_level = np.random.uniform(50, 75)
        elif disease == 'digestive_disorder':
            udder_swelling = 0
            lameness_score = np.random.randint(0, 3)
            appetite_score = np.random.randint(1, 2)  # Very poor appetite
            coat_condition = np.random.randint(1, 3)  # Very poor coat
            # Digestive disorder blood chemistry
            calcium_level = np.random.uniform(8.0, 10.5)
            phosphorus_level = np.random.uniform(3.5, 6.5)
            protein_level = np.random.uniform(5.0, 7.5)
            glucose_level = np.random.uniform(35, 60)  # Lower glucose
        elif disease == 'mineral_deficiency':
            udder_swelling = 0
            lameness_score = np.random.randint(0, 3)
            appetite_score = np.random.randint(2, 4)
            coat_condition = np.random.randint(1, 2)  # Very poor coat
            # Mineral deficiency - low minerals
            calcium_level = np.random.uniform(7.0, 8.5)  # Low calcium
            phosphorus_level = np.random.uniform(2.0, 4.0)  # Low phosphorus
            protein_level = np.random.uniform(5.5, 7.5)
            glucose_level = np.random.uniform(40, 65)
        else:  # healthy
            udder_swelling = 0
            lameness_score = np.random.randint(0, 2)
            appetite_score = np.random.randint(4, 6)  # Good appetite
            coat_condition = np.random.randint(4, 6)  # Good coat
            # Healthy blood chemistry
            calcium_level = np.random.uniform(9.0, 11.5)
            phosphorus_level = np.random.uniform(4.5, 7.5)
            protein_level = np.random.uniform(6.5, 9.0)
            glucose_level = np.random.uniform(55, 80)
            
        return {
            'cow_id': cow_id,
            'breed': breed,
            'age_months': age_months,
            'weight_kg': weight_kg,
            'lactation_stage': lactation_stage,
            'lactation_day': lactation_day,
            'parity': parity,
            
            # Health parameters
            'body_temperature': body_temperature,
            'heart_rate': heart_rate,
            'respiratory_rate': respiratory_rate,
            'white_blood_cells': white_blood_cells,
            'somatic_cell_count': somatic_cell_count,
            'rumen_ph': rumen_ph,
            'rumen_temperature': rumen_temperature,
            
            # Blood chemistry
            'calcium_level': calcium_level,
            'phosphorus_level': phosphorus_level,
            'protein_level': protein_level,
            'glucose_level': glucose_level,
            
            # Physical examination
            'udder_swelling': udder_swelling,
            'lameness_score': lameness_score,
            'appetite_score': appetite_score,
            'coat_condition': coat_condition,
            
            # Feed and activity
            'feed_type': feed_type,
            'feed_quantity_kg': feed_quantity_kg,
            'feeding_frequency': feeding_frequency,
            'walking_distance_km': walking_distance_km,
            'grazing_hours': grazing_hours,
            'rumination_hours': rumination_hours,
            'resting_hours': resting_hours,
            
            # Environmental
            'temperature': temperature,
            'humidity': humidity,
            'season': season,
            
            # Outcomes
            'milk_yield': milk_yield,
            'disease_status': disease,
            'timestamp': datetime.now() - timedelta(days=np.random.randint(0, 365))
        }
    
    def generate_dataset(self, num_cows: int = 5000) -> pd.DataFrame:
        """Generate complete dataset."""
        print(f"Generating cattle disease dataset with {num_cows} samples...")
        
        data = []
        for i in range(num_cows):
            if i % 500 == 0:
                print(f"Generated {i} samples...")
            cow_data = self.generate_cow_data(i + 1)
            data.append(cow_data)
        
        df = pd.DataFrame(data)
        
        print(f"\nDataset generated successfully!")
        print(f"Total samples: {len(df)}")
        print(f"Disease distribution:")
        print(df['disease_status'].value_counts())
        
        return df
    
    def save_dataset(self, df: pd.DataFrame, filename: str = "cattle_disease_dataset.csv"):
        """Save dataset to CSV file."""
        os.makedirs("data", exist_ok=True)
        filepath = os.path.join("data", filename)
        df.to_csv(filepath, index=False)
        print(f"Dataset saved to: {filepath}")
        return filepath

def create_sample_disease_data() -> Dict:
    """Create sample data for testing."""
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
        'season': 'summer',
        'milk_yield': 28.5
    }

if __name__ == "__main__":
    # Generate dataset
    generator = CattleDiseaseDataGenerator()
    df = generator.generate_dataset(5000)
    generator.save_dataset(df)
    
    # Display sample
    print("\nSample data:")
    print(df.head())
    
    print("\nDataset info:")
    print(df.info())
