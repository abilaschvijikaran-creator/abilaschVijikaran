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
project1/
|- src/
|  |- api.py
|  |- train_model.py
|  |- Data_cleaner.py
|  |- evaluate_model.py
|  |- xml_parser.py
|  |- verify_data.py
|- data/
|  |- raw/
|  |- processed/
|- models/
|  |- medical_summarizer/
|- requirements.txt
|- Dockerfile
|- .dockerignore
|- .gitattributes
|- .gitignore


# Setup
Python version: this project was built and tested with Python 3.11, which matches the python:3.11-slim base image used in the Dockerfile. Use Python 3.11 before installing the requirements below

Clone the repository. Git LFS is required as the fine-tuned model weights are stored using it:

```powershell
git lfs install
git clone https://github.com/abilaschvijikaran-creator/abilaschVijikaran.git
cd abilaschVijikaran\project1
```

create and activate a virtual environment:

```powershell
python -m venv venv
.\venv\Scripts\Activate
```

Install dependencies:

```powershell
pip install -r requirements.txt
```
# Dataset setup

The raw data isn't included in this repository, so it needs to be downloaded before the pipeline can be run

1. Download the reports
From the [NIH Open-i archive](https://openi.nlm.nih.gov/faq?download=true), select "Where can I get the Chest X-ray images in Open-i?", then "Reports:Link". Alternatively, use the supplied NLMCXR_reports.tgz.

2. Extract the XML files to data/raw/ecgen-radiology/
The folder doesn't exist in the repository so create it. The expected structure is:
```text
project1/
|- data/
||- raw/
|||- ecgen-radiology/
||||- 1.xml
||||- 2.xml
||||- ...
||- processed/
```

3. Check if it worked

```powershell
Get-ChildItem data\raw\ecgen-radiology\*.xml | Measure-Object
```
This should report 3955 files.

4. Run the pipeline
Start with the parser, this reads the XML files and creates data/processed/findings_impressions.csv:
```powershell
python src\xml_parser.py
```
Then follow through the Pipeline overview section below for the rest of the steps.

# Pipeline overview

The project follows this sequence:

1. Data processing ('xml_parser.py') - Parses raw XML reports and extracts findings/impression pairs into a CSV
2. Data cleaning ('Data_cleaner.py') - Cleans and validates the text, then splits into train, validation and test sets
3. Model training ('train_model.py') - Fine-tunes 'facebook/bart-base' on the training split (2393 examples)
4. Evaluation ('evaluate_model.py') - Creates predictions on the test set and calculates ROUGE-1 F1 score
5. Deployment ('api.py') - Serves the fine-tuned model as a REST API on AWS EC2

The trained model is saved to 'models/medical_summarizer/' and deployed on the live API.

# Dataset

The Indiana University Chest X-ray collection, available from the [NIH Open-i archive](https://openi.nlm.nih.gov/faq?download=true):

- 3955 raw XML report files
- 3419 usable pairs after extracting the different 'Findings' and 'Impression' fields and dropping records where either was missing or empty
- All other fields in the XML are ignored

# Validation
Training uses 'train.csv' to update the model weights and val.csv to measure performance during training. After each epoch the model is run over the full validation set and the average validation loss is printed with the training loss.

The validation data is never used to update the weights. The evaluation runs inside model.eval() and torch.no_grad() and neither loss.backward() or optimizer.step() is called on it so no gradients are generated and no weights are changed.

# Model

Base checkpoint: 'facebook/bart-base'
Task: Abstractive summarisation
Usable pairs: 3419
Training pairs: 2393
Train/validation/test split: 70/15/15 (2393 train, 513 validation, 513 test)
Epochs: 1
Batch size: 2
Learning rate: 5e-5

# Results

**ROUGE-1 F1: 0.5154**, measured on 100 reports from the held-out test set. The brief requires 0.3 or higher.

The three sample findings for the project brief, tested against the live API:

1. Input findings: 'The trachea is midline. The cardiomediastinal silhouette is normal. The lungs are clear, without evidence of acute infiltrate or effusion. There is no pneumothorax. The visualized bony structures reveal no acute abnormalities.'
   Model impression: 'No acute cardiopulmonary abnormalities.'

2. Input findings: 'The lungs are clear. Heart size and mediastinal contours are normal. No osseous abnormalities.'
   Model impression: 'No acute process.'

3. Input findings: 'AP and lateral views were obtained. Bibasilar atelectasis and small left-sided pleural effusion. Stable cardiomegaly. No pneumothorax. Mild pulmonary vascular congestion.'
   Model impression: 'No acute or active cardiac, pulmonary or pleural disease.'

Output 1 and 2 correspond correctly to their inputs, which describe normal studies. Output 3 doesn't, the input describes bibasilar atelectasis, and the model returned a normal impression. This is majority class bias, normal studies dominate the training data, so the model has learned to favor a "nothing acute" impression regardless of input. See the limitations section of the project documentation for a fuller discussion and proposed fixes.

# Deployment

The API runs on an AWS EC2 't3.micro' instance behind an Elastic IP ('13.63.6.159'), so the address stays fixed across restarts. The app is containerised with Docker and served by Gunicorn with a single worker, as each worker loads its own copy of the model into memory.

The model is the full-precision fine-tuned BART, the same one used in evaluation. It runs on CPU only, which is fine for a single worker on a t3.micro.

Three constraints shaped the deployment:

1. Model size - The fine-tuned BART model is 532MB, which exceeded the storage limits of free hosting platforms like Render and Railway. Int8 dynamic quantisation was applied to reduce the model to 286MB as a way around this, but moving to EC2 removed the constraint altogether, so the deployed model is the full-precision version - the same one the ROUGE score above was measured on. The quantisation scripts ('quantize_model.py' and 'evaluate_quantized.py') were used during development to test the approach but are not included in the final repo.
2. Training on limited hardware - During model training on a laptop with 8GB RAM and no GPU, loading the entire dataset at once caused crashes. The solution was to load and process the data in batches (batch size 2), which is reflected in 'train_model.py'. This allows the model to train without running out of memory.
3. Other platforms used - Railway and Render had storage or memory limitations. Fly.io had Docker registry authentication failures. AWS EC2 provided the flexibility needed to work around these constraints.

# Tech stack
- python 3.11
- pytorch
- transformers
- rouge-score
- pandas
- flask
- gunicorn
- docker
- AWS EC2
- git LFS
