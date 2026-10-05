# Medical Text summarisation
A fine-tuned BART model that takes written out radiology findings from a chest X-ray report and summarises them into a short clinical impression. The model is served as a rest API running on AWS EC2.

# Live API
Base address: 'http://13.63.6.159:5000'

# GET /health
Confirms the service is running.

**Response**
```json
{"status":"ok"}
```

# POST /predict
This takes radiology findings, then returns a summarised impression.

**Headers** `Content-Type: application/json`

**Request body**

```json
{"findings": "The lungs are clear. Heart size and mediastinal contours are normal. No osseous abnormalities."}
```

**Response - 200 OK**

```json 
{"impression":"No acute findings."}
```

**Response - 400 Bad Request**

The endpoint checks the request in four stages, each with its own message:

- Body is not valid JSON: `{"error": "Invalid JSON"}`
- No body sent: `{"error": "No data sent"}`
- The findings field is missing: `{"error": "findings field is required"}`
- The findings field is present but empty: `{"error": "findings cannot be empty"}`

# Repository structure
abilaschVijikaran/
|- src/
||- api.py
||- train_model.py
||- Data_cleaner.py
||- evaluate_model.py
||- xml_parser.py
||- verify_data.py
|- data/
||- raw/
|||- ecgen-radiology/
||||- (raw XML files from Indiana University Chest X-ray collection)
|- processed/
||- findings_impressions.csv
||- train.csv
||- test.csv
||- val.csv
|- models/
||- medical_summarizer
|- requirements.txt
|- Dockerfile


# Setup

## Obtaining the Raw Dataset

The raw radiology dataset must be downloaded separately before running the data processing pipeline:

1. Download the Indiana University Chest X-ray collection from the [NIH Open-i archive](https://openi.nlm.nih.gov/faq?download=true)
2. Extract the XML files to the `data/raw/ecgen-radiology/` folder in your project directory
3. Ensure the folder structure matches the example above

Once the raw dataset is in place, you can proceed with cloning the repository and installing dependencies.

## Cloning and Installation

Clone the repository. Git LFS is required as the fine-tuned model weights are stored using it:

```powershell
git lfs install
git clone https://github.com/abilaschvijikaran-creator/abilaschVijikaran.git
cd abilaschVijikaran
```

Create and activate a virtual environment:

```powershell
python -m venv venv
.\\venv\\Scripts\\Activate
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

# Pipeline overview

The project follows this sequence:

1. **Data processing** (`xml_parser.py`) - Parses raw XML reports from `data/raw/ecgen-radiology/` and extracts findings/impression pairs
2. **Data cleaning** (`Data_cleaner.py`) - Cleans and validates the text, then splits it into train, validation and test sets
3. **Model training** (`train_model.py`) - Fine-tunes `facebook/bart-base` on the training split, with validation loss tracked during training
4. **Evaluation** (`evaluate_model.py`) - Creates predictions on the test set and calculates ROUGE scores
5. **Deployment** (`api.py`) - Serves the fine-tuned model as a REST API on AWS EC2

The trained model is saved to `models/medical_summarizer/` and deployed on the live API.

# Dataset

The Indiana University Chest X-ray collection, available from the [NIH Open-i archive](https://openi.nlm.nih.gov/faq?download=true):

- 3955 raw XML report files
- 3419 usable pairs after extracting the different 'Findings' and 'Impression' fields and dropping records where either was missing or empty
- All other fields in the XML are ignored
- Split into 2393 training, 513 validation, and 513 test examples (70/15/15 split)

# Model

Base checkpoint: `facebook/bart-base`
Task: Abstractive summarisation 
Usable pairs: 3419
Training pairs: 2393
Validation pairs: 513
Test pairs: 513
Train/validation/test split: 70/15/15
Epochs: 3
Batch size: 2 
Learning rate: 5e-5
Validation loss: Tracked during training to monitor overfitting

The validation dataset is used during training to evaluate model performance on unseen data after each epoch. This helps prevent overfitting and ensures the model generalises well to new examples.

# Results

**ROUGE-1: 0.6060, ROUGE-2: 0.5224, ROUGE-L: 0.6028**, measured on 100 held-out test examples. The brief requires 0.3 or higher.

The three sample findings for the project brief run against the live API:

1. input findings: 'The trachea is midline. The cardiomediastinal silhouette is normal. The lungs are clear, without evidence of acute infiltrate or effusion. There is no pneumothorax. The visualized bony structures reveal no acute abnormalities.'
Model impression: 'No acute cardiopulmonary abnormalities.'

2. input findings: 'The lungs are clear. Heart size and mediastinal contours are normal. No osseous abnormalities. '
Model impression: 'No acute findings.'

3. input findings: 'AP and lateral views were obtained. Bibasilar atelectasis and small left-sided pleural effusion. Stable cardiomegaly. No pneumothorax. Mild pulmonary vascular congestion.'
Model impression: 'Bibasilar atelectasis with left pleural effusion and cardiomegaly.'

Output 1 and 2 correspond correctly to their inputs, which describe normal studies. Output 3 now correctly identifies the key findings: bibasilar atelectasis, left pleural effusion, and cardiomegaly. With three epochs of training and validation loss monitoring, the model has learned to better balance normal and abnormal cases.

# Deployment

The API runs on an AWS EC2 't3.micro' instance behind an Elastic IP ('13.63.6.159'), so the address stays fixed across restarts. The app is containerised with Docker and served by Gunicorn with a single worker, as each worker loads its own copy of the model into memory.

Three constraints shaped the deployment:

1. Model size - The fine-tuned BART model is 532MB, which exceeded the storage limits of free hosting platforms like Render and Railway. Int8 dynamic quantisation was applied to reduce the model to 286MB as a way around this, but moving to EC2 removed the constraint altogether, so the deployed model is the full-precision version - the same one the ROUGE score above was measured on. The quantisation scripts (`quantize_model.py` and `evaluate_quantized.py`) were used during development to test the approach but are not included in the final repo.
2. Training on limited hardware - During model training on a laptop with 8GB RAM and no GPU, loading the entire dataset at once caused crashes. The solution was to load and process the data in batches (batch size 2), which is reflected in `train_model.py`. This allows the model to train without running out of memory.
3. Other platforms used - Railway and Render had storage or memory limitations. Fly.io had Docker registry authentication failures. AWS EC2 provided the flexibility needed to work around these constraints.

# Tech stack
- python 3
- pytorch
- transformers
- rouge-score
- pandas
- flask
- gunicorn
- docker
- AWS EC2
- git LFS







