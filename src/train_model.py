import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import BartForConditionalGeneration, BartTokenizer
import time
import os

print("=" * 70)
print("TRAINING THE MEDICAL SUMMARISATION MODEL")
print("=" * 70)
print()


# Check what hardware we can use 
print("Checking hardware..")

if torch.cuda.is_available():
    device = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    print(f"  Found GPU: {gpu_name}")
    print(f"  GPU memory: {gpu_memory:.1f} GB")
else:
    device = torch.device("cpu")
    print("  No GPU found - using the CPU instead (this will be slower)")

print(f"  Training will run on: {device}")
print()


# Load the training data
print("Loading the training data...")

train_data = pd.read_csv("data/processed/train.csv")

# makes sure there are no empty cells
train_data = train_data.dropna()

print(f"  Loaded {len(train_data)} training examples")
print(f"  Example finding:    {train_data['findings'].iloc[0][:70]}...")
print(f"  Example impression: {train_data['impression'].iloc[0]}")
print()



# Loading the pre-trained model
print("Loading the pre-trained BART model...")
print("  (First run downloads the model - this can take a few minutes)")

model_name = "facebook/bart-base"

tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)

# Move the model onto the GPU (or leave it on the CPU)
model = model.to(device)

print(f"  Model loaded: {model_name}")
print()


# Turning text into numbers (TOKENISATION)
print("Converting the text into numbers...")

max_input_length = 256    
max_output_length = 64    

# Tokenise the findings
model_inputs = tokenizer(
    list(train_data['findings']),
    max_length=max_input_length,
    padding='max_length',      
    truncation=True,           
    return_tensors="pt"        
)

# Tokenise the impressions 
model_targets = tokenizer(
    list(train_data['impression']),
    max_length=max_output_length,
    padding='max_length',
    truncation=True,
    return_tensors="pt"
)

labels = model_targets['input_ids'].clone()
labels[labels == tokenizer.pad_token_id] = -100

print(f"  Input shape:  {model_inputs['input_ids'].shape}  (rows, tokens per row)")
print(f"  Target shape: {labels.shape}")
print()


# Put the data into batches
print("Preparing the data batches...")

batch_size = 2   

dataset = TensorDataset(
    model_inputs['input_ids'],
    model_inputs['attention_mask'],
    labels
)

train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

print(f"  Batch size: {batch_size} examples at a time")
print(f"  Total batches per epoch: {len(train_loader)}")
print()


# Set up training settings

print("Setting up the training settings..")

learning_rate = 5e-5   
num_epochs = 1         
save_every = 300       

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

save_folder = "models/medical_summarizer"
checkpoint_folder = "models/checkpoint"

os.makedirs(save_folder, exist_ok=True)
os.makedirs(checkpoint_folder, exist_ok=True)

print(f"  Learning rate: {learning_rate}")
print(f"  Number of epochs: {num_epochs}")
print(f"  Saving a backup every {save_every} batches")
print()


# Train the model

print("=" * 70)
print("STARTING TRAINING")
print("=" * 70)
print()

model.train()   
start_time = time.time()

for epoch in range(num_epochs):
    print(f"EPOCH {epoch + 1} of {num_epochs}")
    print("-" * 70)

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

            print(f"  Batch {batch_number + 1} of {len(train_loader)}"
                  f"  |  Average loss: {average_loss:.4f}"
                  f"  |  {minutes_so_far:.1f} min elapsed"
                  f"  |  ~{minutes_left:.0f} min left")

        if (batch_number + 1) % save_every == 0:
            model.save_pretrained(checkpoint_folder)
            tokenizer.save_pretrained(checkpoint_folder)
            print(f"    [backup saved at batch {batch_number + 1}]")

    average_loss = running_loss / batches_done
    print(f"  Epoch {epoch + 1} finished. Average loss: {average_loss:.4f}")
    print()



# Save the trained model

print("=" * 70)
print("TRAINING COMPLETE")
print("=" * 70)
print()

total_minutes = (time.time() - start_time) / 60
print(f"Training took {total_minutes:.1f} minutes")
print()

print("Saving the trained model...")
model.save_pretrained(save_folder)
tokenizer.save_pretrained(save_folder)

print(f"  Model saved to: {save_folder}")
print()
print("Next step: run evaluate_model.py to check the ROUGE score.")
