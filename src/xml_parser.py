import os
import xml.etree.ElementTree as ET
import pandas as pd
from pathlib import Path

xml_folder = Path("data/raw/ecgen-radiology")
output_csv = Path("data/processed/findings_impressions.csv")
output_csv.parent.mkdir(parents=True, exist_ok=True)

data = []
xml_files = sorted(xml_folder.glob("*.xml"))
print(f"Found {len(xml_files)} XML files")

for i, xml_file in enumerate(xml_files):
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        findings = None
        impression = None
        
        for abstract_text in root.findall(".//AbstractText"):
            label = abstract_text.get("Label")
            text = abstract_text.text
            
            if label == "FINDINGS":
                findings = text
            elif label == "IMPRESSION":
                impression = text
                
        if findings and impression:
            data.append({
                "findings": findings.strip(),
                "impression": impression.strip()
            })
    
        if (i + 1) % 500 == 0:
            print(f"Processed {i + 1} / {len(xml_files)} files")
    
    except Exception as e:
        print(f"Error processing {xml_file.name}: {e}")

df = pd.DataFrame(data)
df.to_csv(output_csv, index=False)

print(f"\n Done! Saved {len(df)} records to {output_csv}")
print(f"Findings average length: {df['findings'].str.len().mean():.0f} characters")
print(f"Impression average length: {df['impression'].str.len().mean():.0f} characters")
