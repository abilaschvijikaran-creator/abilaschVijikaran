import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import BartForConditionalGeneration, BartTokenizer
import time
import os
from datetime import datetime

print("Training the medical summarisation model")
print()

# Check avaliable hardware
print("Checking hardware..")

if torch.cuda.is_available():
    device = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(0)
    gpu_props = torch.cuda.get_device_properties(0)
    gpu_memory_bytes = gpu_props.total_memory
    gpu_memory = gpu_memory_bytes / (1024 ** 3) 
    print(f"Found GPU: {gpu_name}")
    print(f"GPU memory: {gpu_memory:.1f} GB")
else:
    device = torch.device("cpu")
    print("No GPU found so im using the CPU instead")

print(f"Training will run on: {device}")
print()


# Load the training data
print("Loading the training data...")

training_data = pd.read_csv("data/processed/train.csv")

# Remove rows with missing values
training_data = training_data.dropna()

print(f"Loaded {len(training_data)} training examples")
print(f"Example finding: {training_data['findings'].iloc[0][:70]}...")
print(f"Example impression: {training_data['impression'].iloc[0]}")
print()

# Loading the validation data
print("Loading the validation data...") 
validation_data = pd.read_csv("data/processed/val.csv")
validation_data = validation_data.dropna()

print(f"Loaded {len(validation_data)} validation examples")
print()

# Load the pre-trained BART model
print("Loading the pre-trained BART model..")
print()

model_name = "facebook/bart-base"

tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)

model = model.to(device)
print(f"Model loaded: {model_name}")
print()

# tokenize the findings and impressions
print("Converting the text into numbers...")
max_input_length = 256  
max_output_length = 64    

# tokenize the findings
model_inputs = tokenizer(
    list(training_data['findings']),
    max_length=max_input_length,
    padding='max_length',      
    truncation=True,           
    return_tensors="pt")

# Tokenise the impression 
model_targets = tokenizer(
    list(training_data['impression']),
    max_length=max_output_length,
    padding='max_length',
    truncation=True,
    return_tensors="pt")

labels = model_targets['input_ids'].clone()
labels[labels == tokenizer.pad_token_id] = -100

print(f"Input shape: {model_inputs['input_ids'].shape}")
print(f"Target shape: {labels.shape}")
print()

#tokenize the validation data
print("Tokenizing the validation data...")
validation_inputs = tokenizer(
    list(validation_data['findings']),
    max_length=max_input_length,
    padding='max_length',
    truncation=True,
    return_tensors="pt")

validation_targets = tokenizer(
    list(validation_data['impression']),
    max_length=max_output_length,
    padding='max_length',
    truncation=True,
    return_tensors="pt")

validation_labels = validation_targets['input_ids'].clone()
validation_labels[validation_labels == tokenizer.pad_token_id] = -100

print(f"Validation input shape: {validation_inputs['input_ids'].shape}")
print(f"Validation target shape: {validation_labels.shape}")
print()

# Create training batches
print("Preparing the data batches...")
batch_size = 2
dataset = TensorDataset(
    model_inputs['input_ids'],
    model_inputs['attention_mask'],
    labels)

train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
print(f"Batch size: {batch_size} examples at a time")
print(f"Total batches per epoch: {len(train_loader)}")
print()

#create validation batches
validation_dataset = TensorDataset(
    validation_inputs['input_ids'],
    validation_inputs['attention_mask'],
    validation_labels)

validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)
print(f"Validation batches: {len(validation_loader)}")
print()


# Set up training hyperparameters
print("Setting up the training hyperparameters...")
learning_rate = 5e-5
num_epochs = 1         
save_every = 300
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
save_folder = "models/medical_summarizer"
checkpoint_folder = "models/checkpoint"

os.makedirs(save_folder, exist_ok=True)
os.makedirs(checkpoint_folder, exist_ok=True)

print(f"Learning rate: {learning_rate}")
print(f"Number of epochs: {num_epochs}")
print(f"Saving a backup every {save_every} batches")
print()


print("Model starting..")
print()
model.train()
start_time = time.time()

# Loop through each epoch
for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1} of {num_epochs}")
    print()

    running_loss = 0.0
    batches_done = 0

    for batch_number, (input_ids, attention_mask, target_labels) in enumerate(train_loader):

        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        target_labels = target_labels.to(device)
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=target_labels)
        loss = outputs.loss
        running_loss += loss.item()
        batches_done += 1
        optimizer.zero_grad()
        loss.backward()   
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()        

        if (batch_number + 1) % 100 == 0:
            average_loss = running_loss / batches_done
            minutes_so_far = (time.time() - start_time) / 60
            batches_left = len(train_loader) - (batch_number + 1)
            minutes_per_batch = minutes_so_far / (batch_number + 1)
            minutes_left = batches_left * minutes_per_batch

            print(f"Batch {batch_number + 1} of {len(train_loader)} | Average loss: {average_loss:.4f} | {minutes_so_far:.1f} min so far | ~{minutes_left:.0f} min left")

        if (batch_number + 1) % save_every == 0:
            model.save_pretrained(checkpoint_folder)
            tokenizer.save_pretrained(checkpoint_folder)
            print(f"[backup saved at batch {batch_number + 1}]")

    average_loss = running_loss / batches_done

    # Checking the performance on the validation set after this epoch
    print("Checking the performance on the validation set...")
    model.eval()
    with torch.no_grad():
        total_val_loss = 0
        num_val_batches = 0
        for val_batch in validation_loader:
            val_inputs, val_attention_mask, val_labels = val_batch
            val_inputs = val_inputs.to(device)
            val_attention_mask = val_attention_mask.to(device)
            val_labels = val_labels.to(device)
            val_outputs = model(
                input_ids=val_inputs,
                attention_mask=val_attention_mask,
                labels=val_labels)
            val_loss = val_outputs.loss
            total_val_loss += val_loss.item()
            num_val_batches += 1

    average_val_loss = total_val_loss / num_val_batches

    # Now both numbers exist, so print them together
    print(f"Epoch {epoch + 1} training loss:   {average_loss:.4f}")
    print(f"Epoch {epoch + 1} validation loss: {average_val_loss:.4f}")
    print()

    # Switch back to training mode ready for the next epoch
    model.train()

# Save the trained model
print("Training complete..")
print()

total_minutes = (time.time() - start_time) / 60
print(f"Training took {total_minutes:.1f} minutes")
print()

print("Saving the trained model...")
model.save_pretrained(save_folder)
tokenizer.save_pretrained(save_folder)

print(f"Model saved to: {save_folder}")
print()