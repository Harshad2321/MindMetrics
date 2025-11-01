"""
Script to calculate actual mean and std from training data
This will create proper scaling parameters for the model
"""
import pandas as pd
import numpy as np
import json

# Load the cleaned (scaled) data
df_scaled = pd.read_csv('cleaned_stress_data.csv')

# Load the original raw data
df_raw = pd.read_csv('DPDEL-FORM (Responses) - Form responses 1.csv')

print("=" * 60)
print("ANALYZING TRAINING DATA FOR PROPER SCALING")
print("=" * 60)

# The scaled data has these columns
print("\nScaled data columns:")
print(df_scaled.columns.tolist())

# We need to reverse engineer the original values before scaling
# Scaled data has mean ≈ 0, std ≈ 1
# Original = (Scaled * std) + mean

# But we can get better stats from the raw data
print("\n" + "=" * 60)
print("RAW DATA ANALYSIS")
print("=" * 60)

# Clean column names in raw data
df_raw_clean = df_raw.copy()

# Map raw columns to cleaned names
column_mapping = {
    'Age': 'age',
    'Heart Rate(BPM) ': 'heartratebpm',
    'Blood Oxygen Level (%)--(only numbers)': 'blood_oxygen_level_percentage',
    'Sleep Duration (Hours)': 'sleep_duration_hours',
    '  Sleep Quality ': 'sleepquality',
    'Body Weight (in KGs)': 'body_weight_kgs',
    'Activity Level': 'activity_level',
    'Screen Time (Hourly)': 'screen_time_hours_daily',
    'Meal Regularity': 'meal_regularity',
    'Sleep Consistency (same bedtime daily?)  ': 'sleepconsistencysamebedtimedaily',
    'Step Count(Daily)': 'step_count_daily',
    'Stress Level (Self-Report)  (1 = Relaxed, 10 = Extremely Stressed) ': 'stress_level',
    ' Gender  ': 'gender',
    ' Profession/Role': 'profession_role'
}

# Calculate statistics for numerical columns
stats = {}

# We need to check what the actual reasonable values are
print("\nKey Statistics from Raw Data:")
print("-" * 60)

# Check a few key columns
for raw_col, clean_col in column_mapping.items():
    if raw_col in df_raw.columns:
        col_data = df_raw[raw_col]
        
        # Try to convert to numeric
        try:
            col_numeric = pd.to_numeric(col_data, errors='coerce')
            valid_data = col_numeric.dropna()
            
            if len(valid_data) > 0:
                print(f"\n{clean_col}:")
                print(f"  Mean: {valid_data.mean():.2f}")
                print(f"  Std: {valid_data.std():.2f}")
                print(f"  Min: {valid_data.min():.2f}")
                print(f"  Max: {valid_data.max():.2f}")
                print(f"  Sample values: {valid_data.head(10).tolist()}")
                
                stats[clean_col] = {
                    'mean': float(valid_data.mean()),
                    'std': float(valid_data.std())
                }
        except Exception as e:
            print(f"  Error processing {raw_col}: {e}")

print("\n" + "=" * 60)
print("SCALED DATA STATISTICS")
print("=" * 60)

# Check the scaled data statistics
print("\nScaled data stats (should be mean≈0, std≈1):")
print(df_scaled[['heartratebpm', 'sleep_duration_hours', 'step_count_daily', 'stress_level']].describe())

# Save the statistics to a JSON file
output_file = 'scaling_parameters.json'
with open(output_file, 'w') as f:
    json.dump(stats, f, indent=2)

print(f"\n✅ Scaling parameters saved to: {output_file}")

print("\n" + "=" * 60)
print("RECOMMENDATIONS")
print("=" * 60)
print("The data appears to have issues. Some observations:")
print("1. Check if columns are properly aligned in the raw CSV")
print("2. Blood oxygen values seem unrealistic (should be 90-100%)")
print("3. Sleep duration values seem very high (should be 4-10 hours)")
print("4. This suggests the CSV columns might be shifted or corrupted")
