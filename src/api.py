# This is the REST API for the medical text summariser
# It takes medical findings as input and returns a summary

import torch
from transformers import BartForConditionalGeneration, BartTokenizer
from flask import Flask, request, jsonify
import os

# Create the Flask app
app = Flask(__name__)

print("Starting API...")
print()

# Check if we have a GPU
print("Checking hardware...")
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("GPU found!")
else:
    device = torch.device("cpu")
    print("No GPU found - using CPU")

print()

# Load the model
print("Loading the trained model...")

model_path = "models/medical_summarizer"

if not os.path.exists(model_path):
    print("ERROR: Model not found at", model_path)
    exit(1)

tokenizer = BartTokenizer.from_pretrained(model_path)
model = BartForConditionalGeneration.from_pretrained(model_path)
model = model.to(device)
model.eval()

print("Model loaded!")
print()


# The /predict endpoint
@app.route('/predict', methods=['POST'])
def predict():
    
    # Get the input from the user
    try:
        input_data = request.get_json()
    except:
        return jsonify({"error": "Invalid JSON"}), 400
    
    # Check that findings field exists
    if input_data is None:
        return jsonify({"error": "No data sent"}), 400
    
    if 'findings' not in input_data:
        return jsonify({"error": "findings field is required"}), 400
    
    findings = input_data['findings']
    
    # Check that findings is not empty
    if findings == "" or findings is None:
        return jsonify({"error": "findings cannot be empty"}), 400
    
    # Convert findings to numbers (tokens)
    try:
        inputs = tokenizer(
            findings,
            max_length=256,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        
        inputs = {key: value.to(device) for key, value in inputs.items()}
    
    except Exception as e:
        return jsonify({"error": "Error with tokenizer: " + str(e)}), 500
    
    # Generate the impression
    try:
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_length=64,
                min_length=5,
                num_beams=4,
                no_repeat_ngram_size=3,
                early_stopping=True
            )
        
        impression = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    except Exception as e:
        return jsonify({"error": "Error generating: " + str(e)}), 500
    
    # Send back the impression
    return jsonify({"impression": impression.strip()}), 200


# Health check endpoint
@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok"}), 200


# Run the server
if __name__ == '__main__':
    print("API is ready!")
    print()
    print("Server running at: http://localhost:5000")
    print()
    print("Press Ctrl+C to stop")
    print()
    
    # debug off - gunicorn runs this in production
    app.run(host='127.0.0.1', port=5000)
