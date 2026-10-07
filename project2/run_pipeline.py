import sys
import subprocess

def run_pipeline():
    """Run the complete project2 pipeline in order."""
    scripts = [
        ("src/data_processing.py", "Data cleaning"),
        ("src/feature_creation.py", "Feature creation"),
        ("src/model.py", "Model and embeddings"),
        ("src/recommend.py", "Recommendations")
    ]
    
    for script, description in scripts:
        print(f"\n{'='*50}")
        print(f"Running: {description} ({script})")
        print(f"{'='*50}")
        result = subprocess.run([sys.executable, script], cwd="/content/drive/MyDrive/project2")
        if result.returncode != 0:
            print(f"Error in {script}. Pipeline stopped.")
            return False
    
    print(f"\n{'='*50}")
    print("Pipeline complete!")
    print(f"{'='*50}")
    return True

if __name__ == "__main__":
    run_pipeline()
