import torch
from transformers import BartForConditionalGeneration, AutoTokenizer
import os


print("MAKING THE MODEL SMALLER (QUANTISATION)")
print()


src_folder = os.path.dirname(os.path.abspath(__file__))
project_folder = os.path.dirname(src_folder)

original_model_folder = os.path.join(project_folder, "models", "medical_summarizer")
smaller_model_folder = os.path.join(project_folder, "models", "medical_summarizer_int8")

print("  Reading the trained model from:", original_model_folder)
print("  Saving the smaller model to:", smaller_model_folder)
print()


#Load the trained model
print("Loading the trained model..")

tokenizer = AutoTokenizer.from_pretrained(original_model_folder)
model = BartForConditionalGeneration.from_pretrained(original_model_folder)
model.eval()

print("  Model loaded")
print()



# Checking how big it is now 
print("STEP 3: Measuring the original size...")

original_weights_file = os.path.join(original_model_folder, "model.safetensors")
original_size_mb = os.path.getsize(original_weights_file) / (1024 * 1024)

print(f"  Original weights file: {original_size_mb:.0f} MB")
print()


#Do the quantisation
print("Converting the numbers from float32 to int8..")

smaller_model = torch.quantization.quantize_dynamic(
    model,
    {torch.nn.Linear},
    dtype=torch.qint8
)

print("  Conversion done")
print()


# Save the smaller model
print("Saving the smaller model..")

os.makedirs(smaller_model_folder, exist_ok=True)

torch.save(
    smaller_model.state_dict(),
    os.path.join(smaller_model_folder, "quantized_weights.pth")
)

tokenizer.save_pretrained(smaller_model_folder)
model.config.save_pretrained(smaller_model_folder)

print("  Saved to:", smaller_model_folder)
print()


# Check the new size
print("Measuring the new size..")

new_weights_file = os.path.join(smaller_model_folder, "quantized_weights.pth")
new_size_mb = os.path.getsize(new_weights_file) / (1024 * 1024)

print(f"  Original: {original_size_mb:.0f} MB")
print(f"  Smaller:  {new_size_mb:.0f} MB")
print(f"  Reduced by {(1 - new_size_mb / original_size_mb) * 100:.0f}%")
print()

# Test the model on one example
print("Testing the smaller model on one example..")

test_findings = ("The lungs are clear. Heart size and mediastinal contours "
                 "are normal. No osseous abnormalities.")

inputs = tokenizer(
    test_findings,
    max_length=256,
    truncation=True,
    return_tensors="pt"
)

with torch.no_grad():
    outputs = smaller_model.generate(
        input_ids=inputs['input_ids'],
        attention_mask=inputs['attention_mask'],
        max_length=64,
        min_length=5,
        num_beams=4,
        no_repeat_ngram_size=3,
        early_stopping=True
    )

result = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("  Input findings:", test_findings)
print("  Model output:  ", result.strip())
print()


print("DONE")
print()
print("Next step: run evaluate_model.py against the smaller model")
print("to check the ROUGE score is still above 0.3")