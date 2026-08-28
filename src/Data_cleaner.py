# Import the libraries we need
import pandas as pd
import re
from sklearn.model_selection import train_test_split

# ===== STEP 1: LOAD THE DATA =====
print("Loading the CSV file...")
df = pd.read_csv("data/processed/findings_impressions.csv")
print(f"Total rows in original file: {len(df)}")
print()

# ===== STEP 2: REMOVE EMPTY ROWS =====
print("Removing rows with missing data...")
# dropna() removes any rows that have empty cells
df = df.dropna()
print(f"Rows after removing empty ones: {len(df)}")
print()

# ===== STEP 3: CLEAN THE TEXT =====
print("Cleaning the text (removing XXXX and extra spaces)...")

# Function to clean one piece of text
def clean_text(text):
    # Remove all XXXX (these are redacted parts)
    text = text.replace('XXXX', '')
    text = text.replace('xxxx', '')
    
    # Remove extra spaces (spaces at start, end, and multiple spaces in middle)
    text = text.strip()
    text = ' '.join(text.split())
    
    return text

# Apply the cleaning function to both columns
df['findings'] = df['findings'].apply(clean_text)
df['impression'] = df['impression'].apply(clean_text)

# Remove any rows that became empty after cleaning
df = df[(df['findings'].str.len() > 0) & (df['impression'].str.len() > 0)]
print(f"Rows after cleaning: {len(df)}")
print()

# ===== STEP 4: SPLIT DATA INTO TRAIN/VAL/TEST =====
print("Splitting data into three groups...")
print("  - Training set: 70% (the AI learns from this)")
print("  - Validation set: 15% (we check if it's learning well)")
print("  - Test set: 15% (final test of how good the model is)")
print()

# First split: 70% train, 30% temp (which will become val + test)
train, temp = train_test_split(df, test_size=0.30, random_state=42)

# Second split: split that 30% in half (15% val, 15% test)
val, test = train_test_split(temp, test_size=0.50, random_state=42)

print(f"Training set size: {len(train)} rows")
print(f"Validation set size: {len(val)} rows")
print(f"Test set size: {len(test)} rows")
print()

# ===== STEP 5: SAVE THE FILES =====
print("Saving the three files...")
train.to_csv("data/processed/train.csv", index=False)
val.to_csv("data/processed/val.csv", index=False)
test.to_csv("data/processed/test.csv", index=False)

print("✓ Done!")
print("✓ Created: train.csv, val.csv, test.csv")