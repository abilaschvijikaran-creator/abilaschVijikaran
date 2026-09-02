# ============================================================================
# TRAIN THE MEDICAL TEXT SUMMARISATION MODEL
# ============================================================================
# What this script does:
#   1. Loads the cleaned training data we made in Week 2
#   2. Loads a pre-trained BART model from HuggingFace
#   3. Fine-tunes (re-trains) that model on our medical data
#   4. Saves a checkpoint every so often so a crash does not lose our work
#   5. Saves the finished model so we can use it in our API later
#
# Why we fine-tune: BART already knows English and knows how to summarise,
# but it has never seen a radiology report. Fine-tuning teaches it the
# specific pattern "long FINDINGS text  ->  short IMPRESSION text".
# ============================================================================

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


# ============================================================================
# STEP 1: CHECK WHAT HARDWARE WE CAN USE
# ============================================================================
# A GPU (graphics card) trains much faster than a CPU.
# torch.cuda.is_available() returns True if an NVIDIA GPU is usable.

print("STEP 1: Checking hardware...")

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


# ============================================================================
# STEP 2: LOAD THE TRAINING DATA
# ============================================================================
# This is the train.csv file we created in Week 2.
# It has two columns: 'findings' (the input) and 'impression' (the target).

print("STEP 2: Loading the training data...")

train_data = pd.read_csv("data/processed/train.csv")

# Safety check - make sure there are no empty cells, which would crash training
train_data = train_data.dropna()

print(f"  Loaded {len(train_data)} training examples")
print(f"  Example finding:    {train_data['findings'].iloc[0][:70]}...")
print(f"  Example impression: {train_data['impression'].iloc[0]}")
print()


# ============================================================================
# STEP 3: LOAD THE PRE-TRAINED MODEL
# ============================================================================
# 'facebook/bart-base' is the smaller version of BART.
# The first time this runs it downloads about 500 MB from the internet.
# After that it is cached on the computer and loads instantly.

print("STEP 3: Loading the pre-trained BART model...")
print("  (First run downloads the model - this can take a few minutes)")

model_name = "facebook/bart-base"

tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)

# Move the model onto the GPU (or leave it on the CPU)
model = model.to(device)

print(f"  Model loaded: {model_name}")
print()


# ============================================================================
# STEP 4: TURN THE TEXT INTO NUMBERS (TOKENISATION)
# ============================================================================
# A neural network cannot read words - it only understands numbers.
# The tokenizer converts each piece of text into a list of numbers (tokens).
#
# max_input_length  = how many tokens of FINDINGS we keep (longer = more memory)
# max_output_length = how many tokens of IMPRESSION we keep

print("STEP 4: Converting the text into numbers...")

max_input_length = 256    # findings are ~220 characters, so 256 tokens is plenty
max_output_length = 64    # impressions are short, ~60 characters

# Tokenise the findings (this is the INPUT the model reads)
model_inputs = tokenizer(
    list(train_data['findings']),
    max_length=max_input_length,
    padding='max_length',      # pad short texts so every row is the same length
    truncation=True,           # cut off anything longer than max_length
    return_tensors="pt"        # "pt" means return PyTorch tensors
)

# Tokenise the impressions (this is the TARGET the model tries to produce)
model_targets = tokenizer(
    list(train_data['impression']),
    max_length=max_output_length,
    padding='max_length',
    truncation=True,
    return_tensors="pt"
)

# --- Important step for quality ---
# The padding we added is just filler. We do NOT want the model to be marked
# wrong for failing to predict filler. Replacing the pad token id with -100
# tells PyTorch "ignore these positions when calculating the error".
labels = model_targets['input_ids'].clone()
labels[labels == tokenizer.pad_token_id] = -100

print(f"  Input shape:  {model_inputs['input_ids'].shape}  (rows, tokens per row)")
print(f"  Target shape: {labels.shape}")
print()


# ============================================================================
# STEP 5: PUT THE DATA INTO BATCHES
# ============================================================================
# We cannot feed all 2,393 examples to the model at once - it would run out
# of memory. Instead we feed it a few at a time. Each group is called a batch.
# Smaller batch_size = less memory used = less chance of crashing.

print("STEP 5: Preparing the data batches...")

batch_size = 2   # kept low deliberately because this laptop has 8 GB of RAM

dataset = TensorDataset(
    model_inputs['input_ids'],
    model_inputs['attention_mask'],
    labels
)

train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

print(f"  Batch size: {batch_size} examples at a time")
print(f"  Total batches per epoch: {len(train_loader)}")
print()


# ============================================================================
# STEP 6: SET UP THE TRAINING SETTINGS
# ============================================================================
# learning_rate = how big a step the model takes when correcting itself.
#                 Too big and it never settles, too small and it learns nothing.
# num_epochs    = how many times we show the model the whole dataset.
# save_every    = how often (in batches) we save a backup copy of the model.

print("STEP 6: Setting up the training settings...")

learning_rate = 5e-5   # 0.00005 - the standard starting point for fine-tuning
num_epochs = 1         # one pass through the data is enough for this dataset
save_every = 300       # save a backup every 300 batches

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# Where the finished model and the backups will be written
save_folder = "models/medical_summarizer"
checkpoint_folder = "models/checkpoint"

os.makedirs(save_folder, exist_ok=True)
os.makedirs(checkpoint_folder, exist_ok=True)

print(f"  Learning rate: {learning_rate}")
print(f"  Number of epochs: {num_epochs}")
print(f"  Saving a backup every {save_every} batches")
print()


# ============================================================================
# STEP 7: TRAIN THE MODEL
# ============================================================================
# This is the actual learning loop. For every batch we:
#   a) give the model the findings and the correct impression
#   b) the model makes a guess and we measure how wrong it was (the "loss")
#   c) we nudge the model's settings slightly to make it less wrong next time
#
# The loss number should generally go DOWN over time. That means it's learning.
#
# IMPORTANT: this takes roughly 50 minutes on a CPU. That is normal.
# Leave the laptop alone while it runs - close the browser and other apps.

print("=" * 70)
print("STARTING TRAINING")
print("=" * 70)
print()

model.train()   # tells the model it is in training mode (not prediction mode)

start_time = time.time()

for epoch in range(num_epochs):
    print(f"EPOCH {epoch + 1} of {num_epochs}")
    print("-" * 70)

    running_loss = 0.0
    batches_done = 0

    for batch_number, (input_ids, attention_mask, target_labels) in enumerate(train_loader):

        # Move this batch onto the GPU (or CPU)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        target_labels = target_labels.to(device)

        # a) The model reads the findings and compares its guess to the target
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=target_labels
        )

        # b) How wrong was it?
        loss = outputs.loss
        running_loss += loss.item()
        batches_done += 1

        # c) Nudge the model to be less wrong next time
        optimizer.zero_grad()   # clear the previous batch's corrections
        loss.backward()         # work out which way to nudge each setting

        # Gradient clipping: stops any single batch from making a huge, wild
        # correction that wrecks what the model has already learned.
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()        # actually apply the nudge

        # Print a progress update every 100 batches so we know it's alive
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

        # ----- SAVE A BACKUP -----
        # If the computer crashes or we stop the script, we can come back to
        # this saved copy instead of starting again from zero.
        if (batch_number + 1) % save_every == 0:
            model.save_pretrained(checkpoint_folder)
            tokenizer.save_pretrained(checkpoint_folder)
            print(f"    [backup saved at batch {batch_number + 1}]")

    average_loss = running_loss / batches_done
    print(f"  Epoch {epoch + 1} finished. Average loss: {average_loss:.4f}")
    print()


# ============================================================================
# STEP 8: SAVE THE TRAINED MODEL
# ============================================================================
# We save both the model and the tokenizer. We need both to make predictions
# later, because the tokenizer is what turns text into numbers and back again.

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