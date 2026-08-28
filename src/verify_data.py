# ===== VERIFY THE CLEANED DATA =====
# This script checks that our three datasets (train, val, test) are good quality

import pandas as pd

print("=" * 60)
print("CHECKING THE CLEANED DATA")
print("=" * 60)
print()

# ===== LOAD THE THREE FILES =====
print("Loading the three CSV files...")
train_data = pd.read_csv("data/processed/train.csv")
val_data = pd.read_csv("data/processed/val.csv")
test_data = pd.read_csv("data/processed/test.csv")
print("✓ Files loaded successfully")
print()

# ===== CHECK TRAINING SET =====
print("-" * 60)
print("TRAINING SET (used to teach the AI)")
print("-" * 60)
print(f"Number of rows: {len(train_data)}")

# Calculate average length of findings
findings_length = train_data['findings'].str.len().mean()
print(f"Average findings length: {findings_length:.0f} characters")

# Calculate average length of impressions
impression_length = train_data['impression'].str.len().mean()
print(f"Average impression length: {impression_length:.0f} characters")

# Show one example
print("\nExample from training data:")
print(f"  Findings: {train_data['findings'].iloc[0][:100]}...")
print(f"  Impression: {train_data['impression'].iloc[0]}")
print()

# ===== CHECK VALIDATION SET =====
print("-" * 60)
print("VALIDATION SET (used to check if AI is learning)")
print("-" * 60)
print(f"Number of rows: {len(val_data)}")

# Calculate average length of findings
findings_length = val_data['findings'].str.len().mean()
print(f"Average findings length: {findings_length:.0f} characters")

# Calculate average length of impressions
impression_length = val_data['impression'].str.len().mean()
print(f"Average impression length: {impression_length:.0f} characters")
print()

# ===== CHECK TEST SET =====
print("-" * 60)
print("TEST SET (final test of how good the AI is)")
print("-" * 60)
print(f"Number of rows: {len(test_data)}")

# Calculate average length of findings
findings_length = test_data['findings'].str.len().mean()
print(f"Average findings length: {findings_length:.0f} characters")

# Calculate average length of impressions
impression_length = test_data['impression'].str.len().mean()
print(f"Average impression length: {impression_length:.0f} characters")
print()

# ===== SUMMARY =====
print("=" * 60)
print("SUMMARY")
print("=" * 60)
total_rows = len(train_data) + len(val_data) + len(test_data)
print(f"Total rows in all three sets: {total_rows}")
print(f"Training: {len(train_data)} rows ({len(train_data)/total_rows*100:.0f}%)")
print(f"Validation: {len(val_data)} rows ({len(val_data)/total_rows*100:.0f}%)")
print(f"Test: {len(test_data)} rows ({len(test_data)/total_rows*100:.0f}%)")
print()
print("✓ Data verification complete!")