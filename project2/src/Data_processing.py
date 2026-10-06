"""Section 1: data cleaning.

Remove duplicate images (same MD5 hash) and images that are too small.

Run from the project2 folder (with Google Drive mounted in Colab):
    python src/data_processing.py
"""

import hashlib
import os
from pathlib import Path

import pandas as pd
from PIL import Image

from config import CLEANED_CSV, CLEANED_IMAGES_DIR, MIN_SIZE, RAW_IMAGES_DIR


def get_file_hash(filepath, chunk_size=65536):
    """Return MD5 hash of file."""
    md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            md5.update(chunk)
    return md5.hexdigest()


def clean_images(raw_dir=RAW_IMAGES_DIR, cleaned_dir=CLEANED_IMAGES_DIR,
                 cleaned_csv=CLEANED_CSV, min_size=MIN_SIZE):
    """Remove duplicates and small images, save cleaned list to CSV."""
    os.makedirs(cleaned_dir, exist_ok=True)

    # Find all images
    image_files = sorted([f for f in os.listdir(raw_dir) 
                         if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    
    print(f"Found {len(image_files)} images to check")

    # Track hashes and cleaned images
    seen_hashes = {}
    duplicates = 0
    too_small = 0
    unreadable = 0
    cleaned = []

    for i, filename in enumerate(image_files):
        filepath = f"{raw_dir}/{filename}"
        
        try:
            # Check if file is readable
            img = Image.open(filepath)
            w, h = img.size
            
            # Check size
            if w < min_size or h < min_size:
                too_small += 1
                continue
            
            # Check for duplicates
            file_hash = get_file_hash(filepath)
            if file_hash in seen_hashes:
                duplicates += 1
                continue
            
            seen_hashes[file_hash] = filename
            cleaned.append(filename)
            
        except Exception:
            unreadable += 1
            continue
        
        if (i + 1) % 5000 == 0:
            print(f"Processed {i + 1}/{len(image_files)}")

    # Save cleaned list
    df = pd.DataFrame({"filename": cleaned})
    df.to_csv(cleaned_csv, index=False)

    print(f"\nResults:")
    print(f"Duplicates removed: {duplicates}")
    print(f"Small images removed: {too_small}")
    print(f"Unreadable files: {unreadable}")
    print(f"Total removed: {duplicates + too_small + unreadable}")
    print(f"Images kept: {len(cleaned)}")
    print(f"Cleaned list saved to {cleaned_csv}")


if __name__ == "__main__":
    clean_images()
