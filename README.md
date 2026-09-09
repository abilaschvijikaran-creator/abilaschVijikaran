# Medical Text summarisation
A fine-tuned BART model that takes written out radiology findings from a chest X-ray report and summaries them into a short clinical impression. The model is served as a rest API running on AWS EC2.

# Live API
Base address: 'http://13.63.6.159:5000'

# Get /health
Confirms the service is running.

**response**
```json
{"status":"ok"}
```

# POST /predict
This takes radiology findings, then returns a summarised impression.

**Headers** 'Content-Type: application/json'

**Request body**

```json
{"findings": "The lungs are clear. Heart size and mediastinal contours are normal. No osseous abnormalities."}
```

**Response -200 OK**

```json 
{"impression":"No acute findings."}
```
**Response - 400 Bad Request** (This is when finding field is missing)

```json
{"error": "Invalid JSON"}
```

# Repository stucture
project1/
|- src/
||- api.py
||- train_model.py
||- Data_cleaner.py
||- evaluate_model.py
||- xml_parser.py
||- verify_data.py
|- data/
|- processed/
||- findings_impressions.csv
||- train.csv
||- test.csv
||- val.csv
|- models/
|- medical_summarizer_m18
|- requirements.txt
|- Dockerfile


# Setup

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

# Pipeline overview

The project follows this sequence:
1. Data processing ('xml_parser.py') - Parses raw XML reports and extracts findings/impression pairs
2. Data cleaning ('Data_cleaner.py') -  Cleans, validates and tokenises the data
3. Model training ('train_model.py') - Fine-tunes 'facebook/bart-base' on the training split (2393)
4. Evaluation ('evaluate_model.py') - Creates predictions on the test set and calculates ROUGE-1 F1 score
5. Deployment ('api.py') - Serves the fine-tuned model as a rest API on AWS EC2

The trained model is saved to 'models/medical_summarizer_m18/' and deployed on the live API.

# Dataset

The Indiana University Chest X-ray collection,available from the [NIH Open-i archive](https://openi.nlm.nih.gov/faq?download=true) :

- 3955 raw XML report files
- 3419 usable pairs after extracting the different 'Findings' and 'Impression' fields and dropping records where either was missing or empty
- All other fields in the XML are ignored

# Model

Base checkpoint : 'facebook/bart-base'
Task: Abstractive summarisation 
Training pairs: 3419
Train/test split : 70/15/15 (2393 train, 513 validation, 513 test)
Epochs: 1 
Batch size: 2 
Learning rate: 5e-5

# Results

**ROUGE-1 F1 on the test set: 0.5154**

The three sample findings for the project brief run against the live API:

1. input findings: 'The trachea is midline. The cardiomediastinal silhouette is normal. The lungs are clear, without evidence of acute infiltrate or effusion. There is no pneumothorax. The visualized bony structures reveal no acute abnormalities.'
Model impression: 'No acute cardiopulmonary abnormalities.'
2. input findings: 'The lungs are clear. Heart size and mediastinal contours are normal. No osseous abnormalities. '
Model impression: 'No acute findings.'
3. input findings: 'AP and lateral views were obtained. Bibasilar atelectasis and small left-sided pleural effusion. Stable cardiomegaly. No pneumothorax. Mild pulmonary vascular congestion.'
Model impression: 'No acute cardiopulmonary process.'

Output 1 and 2 correspond correctly to their inputs, which describe normal studies. Output 3 doesn't, the input describes bibasilar atelectasis, and the model returned a normal impression. This is majority class bias, normal studies dominate the training data, so the model has learned to favor a "nothing acute" impression regardless of input. See the limitations section of the project documentation for a fuller discussion and proposed fixes.

# Deployment

The API runs on an AWS EC2 't3.micro' instance behind an Elastic IP ('13.63.6.159'), so the address stays fixed across restarts.

Two constraints shaped the deployment:

1. Model size - The fine-tuned BART model was originally 532MB, which exceeded the storage limits of free hosting platforms like Render and Railway. To work around this, Int8 dynamic quantisation was applied to reduce the model to 286MB which was a trade-off of roughly 0.014 ROUGE-1 F1 for significantly smaller file size. This quantised version is what is deployed on EC2. The quantisation scripts ('quantize_model.py' and 'evaluate_quantized.py') were used during development to test this approach but are not included in the final repo, since the quantised model weights are already saved in 'models/medical_summarizer_m18/'.
2. Training on limited hardware - During model training on a laptop with 8GB RAM, loading the entire dataset at once caused crashes. The solution was to load and process the data in batches (batch size 2), which is reflected in 'train_model.py'. This allows the model to train without running out of memory.
3. Other platforms used - Railway and Render had storage or memory limitations. Fly.io had Docker registry authentication failures. AWS EC2 provided the flexibility needed to work around these constraints.

# Tech stack
- python 3
- pytorch
- rouge-score
- pandas
- flask
- AWS EC2
- git LFS
