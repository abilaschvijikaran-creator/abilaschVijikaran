"""Section 1: data cleaning.

Remove duplicate images (same MD5 hash) and images that are too small.

Run from the project2 folder (with Google Drive mounted in Colab):
    python src/data_processing.py
"""

import hashlib
import os
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
from PIL import Image

from config import CLEANED_CSV, CLEANED_IMAGES_DIR, MIN_SIZE, RAW_IMAGES_DIR


def clean_images(raw_dir=RAW_IMAGES_DIR, cleaned_dir=CLEANED_IMAGES_DIR,
                 cleaned_csv=CLEANED_CSV, min_size=MIN_SIZE):
    """Remove duplicates and small images, save cleaned list to CSV."""
    os.makedirs(cleaned_dir, exist_ok=True)

    raw_files = os.listdir(raw_dir)
    print(f"Starting with {len(raw_files)} images")

    invalid_images = []
    duplicate_hashes = {}
    cleaned_files = []

    for idx, filename in enumerate(raw_files):
        if idx % 5000 == 0:
            print(f"Processing: {idx}/{len(raw_files)}")

        filepath = os.path.join(raw_dir, filename)

        try:
            img = Image.open(filepath).convert('RGB')

            if img.size[0] < min_size or img.size[1] < min_size:
                invalid_images.append(filename)
                continue

            with open(filepath, 'rb') as f:
                file_hash = hashlib.md5(f.read()).hexdigest()

            if file_hash in duplicate_hashes:
                invalid_images.append(filename)
                continue

            duplicate_hashes[file_hash] = filename

            output_path = os.path.join(cleaned_dir, filename)
            img.save(output_path, 'JPEG', quality=95)
            cleaned_files.append(filename)

        except Exception as e:
            invalid_images.append(filename)

    print(f"\nCleaning complete:")
    print(f"Cleaned: {len(cleaned_files)}")
    print(f"Invalid: {len(invalid_images)}")
    print(f"Removed: {len(raw_files) - len(cleaned_files)}")

    df_cleaned = pd.DataFrame({'filename': cleaned_files})
    df_cleaned.to_csv(cleaned_csv, index=False)
    print(f"\nSaved {len(cleaned_files)} cleaned images")
    print(f"CSV saved: cleaned_images_v2.csv")


if __name__ == "__main__":
    clean_images()
