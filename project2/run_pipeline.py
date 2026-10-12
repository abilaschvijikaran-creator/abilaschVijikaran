"""Run the full Project 2 pipeline in order.

Run from the project2 folder:
    python run_pipeline.py                    # tests with the first cleaned catalogue image
    python run_pipeline.py path/to/photo.jpg  # tests with your own photo

The data folders are set in src/config.py (PROJECT_DIR).
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

STAGES = [
    ("1. Data cleaning", "data_processing.py"),
    ("2. Feature creation", "feature_creation.py"),
    ("3. Model and embeddings", "model.py"),
]


def run_stage(description, script, *args):
    print(f"\n{'=' * 50}\n{description} ({script})\n{'=' * 50}")
    result = subprocess.run([sys.executable, str(SRC / script), *args], cwd=ROOT)
    if result.returncode != 0:
        sys.exit(f"\n{script} failed. Pipeline stopped.")


def main():
    for description, script in STAGES:
        run_stage(description, script)

    if len(sys.argv) > 1:
        test_image = sys.argv[1]
    else:
        import pandas as pd
        from config import CLEANED_CSV, CLEANED_IMAGES_DIR

        first_file = pd.read_csv(CLEANED_CSV)["filename"].iloc[0]
        test_image = f"{CLEANED_IMAGES_DIR}/{first_file}"

    run_stage("4. Recommendations", "recommend.py", test_image)
    print(f"\n{'=' * 50}\nPipeline complete\n{'=' * 50}")


if __name__ == "__main__":
    main()