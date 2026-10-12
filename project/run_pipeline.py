import subprocess
import sys
import time

print("Medical text summarisation Full Pipeline")
print()

pipeline_stages = [
    ("Stage 1 of 4 - Data processing (parsing the raw XML reports)", "src/xml_parser.py"),
    ("Stage 2 of 4 - Data cleaning and train/validation/test split", "src/Data_cleaner.py"),
    ("Stage 3 of 4 - Model training (fine-tuning BART)", "src/train_model.py"),
    ("Stage 4 of 4 - Model evaluation (ROUGE-1 F1 score)", "src/evaluate_model.py"),]

print("This pipeline will run the following stages:")
for stage_name, script_path in pipeline_stages:
    print(f"  - {stage_name}")
print()
print("Note: stage 3 fine-tunes the model and can take several hours on a CPU.")
print()

pipeline_start_time = time.time()

# Run each stage one at a time
for stage_number, (stage_name, script_path) in enumerate(pipeline_stages, start=1):

    print(stage_name)
    print(f"Running: {script_path}")
    print()

    stage_start_time = time.time()
    result = subprocess.run([sys.executable, script_path])
    stage_minutes = (time.time() - stage_start_time) / 60

    if result.returncode != 0:
        print()
        print(f"Pipeline stopped- {stage_name} failed")
        print(f"The script {script_path} exited with code {result.returncode}")
        sys.exit(1)

    print()
    print(f"{stage_name} finished in {stage_minutes:.1f} minutes")
    print()

# This is where we know all stages have completed successfully
total_minutes = (time.time() - pipeline_start_time) / 60

print("Pipeline complete")
print()
print()
print(f"All {len(pipeline_stages)} stages finished in {total_minutes:.1f} minutes")
print()
print("The trained model has been saved to: models/medical_summarizer")
print("To serve it as a REST API, run: python src/api.py")
print()