"""
Comprehensive data generator for cattle milk yield prediction.
Creates realistic synthetic data based on dairy farming research and industry standards.
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import random

class CattleDataGenerator:
    """Generate realistic cattle data for milk yield prediction."""
    
    def __init__(self, seed=42):
        """Initialize the data generator with random seed."""
        np.random.seed(seed)
        random.seed(seed)
        
        # Define realistic ranges and categories based on dairy farming research
        self.breeds = ['Holstein', 'Jersey', 'Guernsey', 'Ayrshire', 'Brown Swiss', 'Simmental']
        self.lactation_stages = ['early', 'peak', 'mid', 'late', 'dry']
        self.feed_types = ['green_fodder', 'dry_fodder', 'concentrates', 'silage', 'mixed']
        self.housing_types = ['free_stall', 'tie_stall', 'pasture', 'compost_barn']
        self.seasons = ['spring', 'summer', 'autumn', 'winter']
        
        # Breed-specific milk yield multipliers (Holstein = baseline 1.0)
        self.breed_multipliers = {
            'Holstein': 1.0,
            'Jersey': 0.7,
            'Guernsey': 0.8,
            'Ayrshire': 0.85,
            'Brown Swiss': 0.9,
            'Simmental': 0.95
        }
        
        # Lactation stage multipliers
        self.lactation_multipliers = {
            'early': 0.8,
            'peak': 1.2,
            'mid': 1.0,
            'late': 0.6,
            'dry': 0.0
        }
    
    def generate_cattle_dataset(self, n_cows=500, days_per_cow=365):
        """Generate comprehensive cattle dataset."""
        print(f"🐄 Generating dataset for {n_cows} cows over {days_per_cow} days...")
        
        all_records = []
        
        for cow_id in range(1, n_cows + 1):
            cow_records = self._generate_cow_data(cow_id, days_per_cow)
            all_records.extend(cow_records)
        
        df = pd.DataFrame(all_records)
        print(f"✅ Generated {len(df)} records")
        return df
    
    def _generate_cow_data(self, cow_id, days):
        """Generate data for a single cow over specified days."""
        # Fixed cow characteristics
        breed = np.random.choice(self.breeds)
        age_months = np.random.randint(24, 120)  # 2-10 years
        weight_kg = np.random.normal(750, 150)  # Extended weight range for larger cattle
        weight_kg = np.clip(weight_kg, 300, 1200)  # Ensure within 300-1200 kg range
        parity = max(1, int((age_months - 24) / 12))  # Calving number
        
        # Lactation cycle (305 days lactation + 60 days dry period)
        lactation_day = np.random.randint(1, 365)
        
        records = []
        base_date = datetime.now() - timedelta(days=days)
        
        for day in range(days):
            current_date = base_date + timedelta(days=day)
            
            # Determine lactation stage based on lactation day
            if lactation_day <= 60:
                lactation_stage = 'early'
            elif lactation_day <= 100:
                lactation_stage = 'peak'
            elif lactation_day <= 200:
                lactation_stage = 'mid'
            elif lactation_day <= 305:
                lactation_stage = 'late'
            else:
                lactation_stage = 'dry'
            
            # Environmental data (seasonal variations)
            season = self._get_season(current_date)
            temp_base = {'spring': 18, 'summer': 28, 'autumn': 15, 'winter': 8}[season]
            temperature = np.random.normal(temp_base, 5)
            humidity = np.random.normal(65, 15)
            
            # Feed and nutrition data
            feed_type = np.random.choice(self.feed_types)
            feed_quantity_kg = np.random.normal(12, 2)  # Daily feed intake
            feeding_frequency = np.random.randint(2, 5)  # Times per day
            
            # Activity and behavioral data
            walking_distance_km = np.random.normal(3, 1)  # Daily walking
            grazing_hours = np.random.normal(8, 2) if season in ['spring', 'summer'] else np.random.normal(4, 1)
            rumination_hours = np.random.normal(7, 1)
            resting_hours = 24 - grazing_hours - rumination_hours - 2  # 2 hours for other activities
            
            # Health data
            body_temp = np.random.normal(38.5, 0.5)  # Normal cow temperature
            heart_rate = np.random.normal(60, 10)  # Beats per minute
            health_score = np.random.uniform(0.7, 1.0)  # Health indicator (0-1)
            
            # Housing conditions
            housing_type = np.random.choice(self.housing_types)
            ventilation_score = np.random.uniform(0.6, 1.0)
            cleanliness_score = np.random.uniform(0.5, 1.0)
            
            # Historical yield (previous days average)
            historical_yield_7d = np.random.normal(25, 5)
            historical_yield_30d = np.random.normal(24, 4)
            
            # Calculate milk yield based on multiple factors
            milk_yield = self._calculate_milk_yield(
                breed, lactation_stage, age_months, weight_kg, parity,
                feed_quantity_kg, temperature, humidity, walking_distance_km,
                grazing_hours, health_score, historical_yield_7d
            )
            
            record = {
                # Animal-related data
                'cow_id': cow_id,
                'breed': breed,
                'age_months': age_months,
                'weight_kg': weight_kg,
                'lactation_stage': lactation_stage,
                'lactation_day': lactation_day,
                'parity': parity,
                'historical_yield_7d': historical_yield_7d,
                'historical_yield_30d': historical_yield_30d,
                
                # Feed and nutrition data
                'feed_type': feed_type,
                'feed_quantity_kg': feed_quantity_kg,
                'feeding_frequency': feeding_frequency,
                
                # Activity & behavioral data
                'walking_distance_km': walking_distance_km,
                'grazing_hours': grazing_hours,
                'rumination_hours': rumination_hours,
                'resting_hours': resting_hours,
                
                # Health data
                'body_temperature': body_temp,
                'heart_rate': heart_rate,
                'health_score': health_score,
                
                # Environmental data
                'temperature': temperature,
                'humidity': humidity,
                'season': season,
                'housing_type': housing_type,
                'ventilation_score': ventilation_score,
                'cleanliness_score': cleanliness_score,
                
                # Target variable
                'milk_yield_liters': milk_yield,
                
                # Metadata
                'date': current_date.strftime('%Y-%m-%d'),
                'day_of_year': current_date.timetuple().tm_yday
            }
            
            records.append(record)
            
            # Update lactation day
            lactation_day += 1
            if lactation_day > 365:  # Reset lactation cycle
                lactation_day = 1
        
        return records
    
    def _calculate_milk_yield(self, breed, lactation_stage, age_months, weight_kg,
                            parity, feed_quantity, temperature, humidity,
                            walking_distance, grazing_hours, health_score, historical_yield):
        """Calculate realistic milk yield based on multiple factors."""
        
        # Base yield for Holstein at peak lactation
        base_yield = 30.0
        
        # Apply breed multiplier
        yield_value = base_yield * self.breed_multipliers[breed]
        
        # Apply lactation stage multiplier
        yield_value *= self.lactation_multipliers[lactation_stage]
        
        # Age effect (peak at 4-6 years)
        age_years = age_months / 12
        if age_years < 3:
            age_factor = 0.8
        elif age_years < 6:
            age_factor = 1.0
        else:
            age_factor = max(0.7, 1.0 - (age_years - 6) * 0.05)
        yield_value *= age_factor
        
        # Parity effect (increases up to 3rd lactation)
        parity_factor = min(1.0, 0.7 + parity * 0.1)
        yield_value *= parity_factor
        
        # Feed quantity effect
        feed_factor = min(1.2, 0.6 + feed_quantity * 0.05)
        yield_value *= feed_factor
        
        # Temperature stress (optimal 15-25°C)
        if temperature < 10 or temperature > 30:
            temp_stress = 0.9
        elif temperature < 5 or temperature > 35:
            temp_stress = 0.8
        else:
            temp_stress = 1.0
        yield_value *= temp_stress
        
        # Activity effect (moderate activity is beneficial)
        activity_factor = 1.0 + (walking_distance - 3) * 0.02
        activity_factor = max(0.8, min(1.1, activity_factor))
        yield_value *= activity_factor
        
        # Health effect
        yield_value *= health_score
        
        # Historical yield influence (regression to mean)
        yield_value = 0.7 * yield_value + 0.3 * historical_yield
        
        # Add random variation
        yield_value += np.random.normal(0, 2)
        
        # Ensure realistic bounds
        return max(0, min(50, yield_value))
    
    def _get_season(self, date):
        """Determine season based on date."""
        month = date.month
        if month in [3, 4, 5]:
            return 'spring'
        elif month in [6, 7, 8]:
            return 'summer'
        elif month in [9, 10, 11]:
            return 'autumn'
        else:
            return 'winter'
    
    def save_dataset(self, df, filename='cattle_milk_yield_dataset.csv'):
        """Save dataset to CSV file."""
        os.makedirs('data', exist_ok=True)
        filepath = os.path.join('data', filename)
        df.to_csv(filepath, index=False)
        print(f"✅ Dataset saved to {filepath}")
        return filepath

def generate_sample_data():
    """Generate sample dataset for testing."""
    generator = CattleDataGenerator()
    
    # Generate smaller dataset for quick testing
    df = generator.generate_cattle_dataset(n_cows=50, days_per_cow=30)
    
    # Save sample data
    sample_path = generator.save_dataset(df, 'sample_cattle_data.csv')
    
    print(f"\n📊 Dataset Statistics:")
    print(f"Shape: {df.shape}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"Average milk yield: {df['milk_yield_liters'].mean():.2f} L")
    print(f"Yield range: {df['milk_yield_liters'].min():.2f} - {df['milk_yield_liters'].max():.2f} L")
    
    return df, sample_path

if __name__ == "__main__":
    # Generate sample data
    df, path = generate_sample_data()
    print(f"\n🎉 Sample data generated successfully!")
    print(f"Use this data for training: {path}")
