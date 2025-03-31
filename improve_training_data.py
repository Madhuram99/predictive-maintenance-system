import pandas as pd
import numpy as np
from faker import Faker  # Correct import - from faker package, not pandas
from datetime import datetime, timedelta
import random
import os

def enhance_data(df):
    """Add temporal and rolling features to the dataframe"""
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df['day_of_week'] = df['date'].dt.dayofweek
        df['day_of_month'] = df['date'].dt.day
        df['month'] = df['date'].dt.month
    
    metric_cols = [f'metric{i}' for i in range(1, 10)]
    for device in df['device'].unique():
        device_mask = df['device'] == device
        for col in metric_cols:
            df.loc[device_mask, f'{col}_rolling_mean_7'] = (
                df.loc[device_mask, col].rolling(window=7, min_periods=1).mean())
            df.loc[device_mask, f'{col}_rolling_std_7'] = (
                df.loc[device_mask, col].rolling(window=7, min_periods=1).std())
    
    return df.dropna()

def generate_synthetic_data(base_df, num_new_samples=1000):
    """Generate synthetic but realistic device data"""
    synthetic_data = []
    devices = base_df['device'].unique()
    failure_rates = base_df.groupby('device')['failure'].mean()
    
    for _ in range(num_new_samples):
        device = np.random.choice(devices)
        base_metrics = base_df[base_df['device'] == device].sample(1)
        
        row = {
            'device': device,
            'date': (datetime.now() - timedelta(days=random.randint(0, 365)))
                      .strftime('%m/%d/%Y'),
            'failure': np.random.binomial(1, failure_rates[device])
        }
        
        for i in range(1, 10):
            healthy_val = base_metrics[f'metric{i}'].values[0]
            if row['failure']:
                row[f'metric{i}'] = healthy_val * (1 + random.uniform(0.1, 0.5))
                if random.random() > 0.7:
                    row[f'metric{i}'] *= 1.8
            else:
                row[f'metric{i}'] = healthy_val * (1 + random.uniform(-0.1, 0.1))
        
        synthetic_data.append(row)
    
    return pd.DataFrame(synthetic_data)

def add_derived_features(df):
    """Create additional engineered features"""
    if 'date' in df.columns:
        df['days_in_service'] = (pd.to_datetime(df['date']) - 
                               pd.to_datetime(df['date']).min()).dt.days
    
    for i in range(1, 5):
        for j in range(i+1, 5):
            df[f'metric_ratio_{i}_{j}'] = df[f'metric{i}'] / (df[f'metric{j}'] + 1e-6)
    
    metric_cols = [f'metric{i}' for i in range(1, 10)]
    df['metrics_mean'] = df[metric_cols].mean(axis=1)
    df['metrics_std'] = df[metric_cols].std(axis=1)
    
    return df

def main():
    # Initialize Faker
    fake = Faker()
    
    # Set file paths
    input_path = 'maintenance_app/data/predictive_ml.csv'
    output_path = 'maintenance_app/data/improved_predictive_ml.csv'
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Load and process data
    df = pd.read_csv(input_path)
    enhanced_df = enhance_data(df)
    synthetic_df = generate_synthetic_data(df)
    improved_df = pd.concat([enhanced_df, synthetic_df]).sample(frac=1).reset_index(drop=True)
    improved_df = add_derived_features(improved_df)
    
    # Save results
    improved_df.to_csv(output_path, index=False)
    print(f"Improved dataset saved to {output_path}")
    print(f"Records: {len(improved_df)}")
    print(f"Failure rate: {improved_df['failure'].mean():.2%}")

if __name__ == "__main__":
    main()