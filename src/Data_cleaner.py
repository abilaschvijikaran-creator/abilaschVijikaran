import pandas as pd
import re
from sklearn.model_selection import train_test_split

#Load the data
print("Loading the CSV file...")
df = pd.read_csv("data/processed/findings_impressions.csv")
print(f"Total rows in original file: {len(df)}")
print()

#Remove empty rows
print("Removing rows with missing data...")
df = df.dropna()
print(f"Rows after removing empty ones: {len(df)}")
print()

#Clean the text
print("Cleaning the text (removing XXXX and extra spaces)...")

def clean_text(text):
    text = text.replace('XXXX', '')
    text = text.replace('xxxx', '')
    text = text.strip()
    text = ' '.join(text.split())
    
    return text

df['findings'] = df['findings'].apply(clean_text)
df['impression'] = df['impression'].apply(clean_text)
df = df[(df['findings'].str.len() > 0) & (df['impression'].str.len() > 0)]
print(f"Rows after cleaning: {len(df)}")
print()

print("Splitting 70/15/15..")

train, temp = train_test_split(df, test_size=0.30, random_state=42)
val, test = train_test_split(temp, test_size=0.50, random_state=42)

print(f"Training set size: {len(train)} rows")
print(f"Validation set size: {len(val)} rows")
print(f"Test set size: {len(test)} rows")
print()

#Save the files
print("Saving the three files..")
train.to_csv("data/processed/train.csv", index=False)
val.to_csv("data/processed/val.csv", index=False)
test.to_csv("data/processed/test.csv", index=False)

print("✓ Done!")
print("✓ Created: train.csv, val.csv, test.csv")
