import torch
from flask import Flask, request, jsonify
import os
from transformers import BartForConditionalGeneration, AutoTokenizer, BartConfig

# Create the Flask app
app = Flask(__name__)

print("Starting API...")
print()

# Check if our quantised model is running on cpu
print("Checking hardware...")
device = torch.device("cpu")
print("Quantised model - running on CPU")
print()

# Load the model
print("Loading the trained model...")

src_folder = os.path.dirname(os.path.abspath(__file__))
project_folder = os.path.dirname(src_folder)
model_path = os.path.join(project_folder, "models", "medical_summarizer_int8")

print("Looking for the model at:", model_path)

if not os.path.exists(model_path):
    print("ERROR: Model not found at", model_path)
    print("Files in the project folder:", os.listdir(project_folder))
    exit(1)

tokenizer = AutoTokenizer.from_pretrained(model_path)
config = BartConfig.from_pretrained(model_path)
model = BartForConditionalGeneration(config)
model = torch.quantization.quantize_dynamic(
    model,
    {torch.nn.Linear},
    dtype=torch.qint8)
weights_path = os.path.join(model_path, "quantized_weights.pth")
model.load_state_dict(torch.load(weights_path, map_location="cpu"))
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
