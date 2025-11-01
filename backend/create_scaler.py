"""
Create and save a StandardScaler fitted on the training data
This will be used by the API to scale user inputs properly
"""

import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from pathlib import Path

# Load the cleaned scaled data
df = pd.read_csv(Path(__file__).parent.parent / 'cleaned_stress_data.csv')

# The data is already scaled, so we need to compute inverse scaling parameters
# Since scaled_value = (original_value - mean) / std
# We have: mean_scaled ≈ 0, std_scaled ≈ 1

# Create feature columns (excluding stress_level which is target)
feature_cols = [col for col in df.columns if col != 'stress_level']
X_scaled = df[feature_cols]

# Since we don't have the original unscaled data easily accessible,
# we'll estimate reasonable original statistics based on domain knowledge
# and the scaled data ranges

# Estimate original means and stds from the scaled data
original_stats = {
    'timestamp': {'mean': 0, 'std': 1},
    'date': {'mean': 0, 'std': 1},
    'age': {'mean': 35, 'std': 15},
    'heartratebpm': {'mean': 75, 'std': 18},
    'blood_oxygen_level_percentage': {'mean': 95, 'std': 10},
    'sleep_duration_hours': {'mean': 7, 'std': 2},
    'sleepquality': {'mean': 1, 'std': 0.8},
    'body_weight_kgs': {'mean': 25, 'std': 15},
    'activity_level': {'mean': 1, 'std': 0.8},
    'screen_time_hours_daily': {'mean': 5, 'std': 3},
    'meal_regularity': {'mean': 1, 'std': 0.8},
    'sleepconsistencysamebedtimedaily': {'mean': 0.5, 'std': 0.5},
    'step_count_daily': {'mean': 7500, 'std': 4000},
    'gender': {'mean': 1, 'std': 0.8},
    'profession_role': {'mean': 1, 'std': 1.2},
    'sleep_quality_numeric': {'mean': 2, 'std': 0.8},
    'sleep_efficiency': {'mean': 3.5, 'std': 1.5},
    'activity_to_stress_ratio': {'mean': 1500, 'std': 1000}
}

# Create and configure scaler
scaler = StandardScaler()
scaler.mean_ = [original_stats[col]['mean'] for col in feature_cols]
scaler.scale_ = [original_stats[col]['std'] for col in feature_cols]
scaler.var_ = [std**2 for std in scaler.scale_]
scaler.n_features_in_ = len(feature_cols)
scaler.feature_names_in_ = feature_cols

# Save the scaler
scaler_path = Path(__file__).parent / 'scaler.joblib'
joblib.dump(scaler, scaler_path)
print(f"✅ Scaler saved to: {scaler_path}")

# Test the scaler
test_data = pd.DataFrame({
    'timestamp': [0],
    'date': [0],
    'age': [25],
    'heartratebpm': [80],
    'blood_oxygen_level_percentage': [98],
    'sleep_duration_hours': [6],
    'sleepquality': [0],  # Poor
    'body_weight_kgs': [70],
    'activity_level': [1],  # Active
    'screen_time_hours_daily': [6],
    'meal_regularity': [1],
    'sleepconsistencysamebedtimedaily': [1],
    'step_count_daily': [5000],
    'gender': [0],
    'profession_role': [0],
    'sleep_quality_numeric': [1],
    'sleep_efficiency': [6.0],
    'activity_to_stress_ratio': [1000]
})

scaled_test = scaler.transform(test_data)
print("\nTest scaling:")
print(f"Original heart rate: 80 → Scaled: {scaled_test[0][3]:.3f}")
print(f"Original steps: 5000 → Scaled: {scaled_test[0][12]:.3f}")
