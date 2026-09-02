# ============================================================================
# EVALUATE THE MEDICAL TEXT SUMMARISATION MODEL
# ============================================================================
# What this script does:
#   1. Loads the model we fine-tuned in train_model.py
#   2. Loads the test data (which the model has NEVER seen before)
#   3. Asks the model to write an IMPRESSION for each set of FINDINGS
#   4. Compares the model's answer to the real doctor's answer using ROUGE
#   5. Prints the scores and tells us whether we passed the 0.3 threshold
#
# What is ROUGE?
#   ROUGE measures how much overlap there is between the summary the model
#   wrote and the summary the doctor actually wrote.
#     ROUGE-1 = overlap of single words
#     ROUGE-2 = overlap of two-word phrases (harder)
#     ROUGE-L = overlap of the longest matching sequence of words
#   Each one gives a Precision, a Recall and an F1 score.
#   F1 combines precision and recall into one number. That is the one the
#   assessment brief cares about, and it needs to be 0.3 or higher.
#
# BEFORE RUNNING THIS: install the scoring library with
#   pip install rouge-score
# ============================================================================

import pandas as pd
import torch
from transformers import BartForConditionalGeneration, BartTokenizer
from rouge_score import rouge_scorer
import time

print("=" * 70)
print("EVALUATING THE MEDICAL SUMMARISATION MODEL")
print("=" * 70)
print()


# ============================================================================
# STEP 1: SETTINGS YOU CAN CHANGE
# ============================================================================
# how_many_to_test:
#   Generating text is slow on a CPU - roughly 1 to 2 seconds per report.
#   Start with 100 to get a quick reliable estimate (takes a few minutes).
#   Once you are happy, set this to None to run the full test set of ~513.

how_many_to_test = 100      # set to None to use every row in test.csv

model_folder = "models/medical_summarizer"

print("STEP 1: Settings")
print(f"  Model folder: {model_folder}")
print(f"  Number of test reports: {how_many_to_test if how_many_to_test else 'ALL'}")
print()


# ============================================================================
# STEP 2: CHECK WHAT HARDWARE WE CAN USE
# ============================================================================

print("STEP 2: Checking hardware...")

if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"  Found GPU: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("  No GPU found - using the CPU instead (this will be slower)")

print(f"  Evaluation will run on: {device}")
print()


# ============================================================================
# STEP 3: LOAD OUR TRAINED MODEL
# ============================================================================
# Note we load from our own folder, NOT from HuggingFace. This is the model
# that has already learned the radiology pattern from our training data.

print("STEP 3: Loading our trained model...")

tokenizer = BartTokenizer.from_pretrained(model_folder)
model = BartForConditionalGeneration.from_pretrained(model_folder)

model = model.to(device)

# eval() switches the model into "prediction mode".
# This turns off training-only behaviour like dropout, which would otherwise
# make the model's answers random and slightly worse.
model.eval()

print(f"  Model loaded from: {model_folder}")
print()


# ============================================================================
# STEP 4: LOAD THE TEST DATA
# ============================================================================
# This is the 15% of the data we deliberately held back in Week 2.
# The model has never seen these reports, so this is a fair test.

print("STEP 4: Loading the test data...")

test_data = pd.read_csv("data/processed/test.csv")
test_data = test_data.dropna()

# Cut it down if we only want a quick check
if how_many_to_test is not None:
    test_data = test_data.head(how_many_to_test)

findings_list = list(test_data['findings'])
real_impressions = list(test_data['impression'])

print(f"  Loaded {len(findings_list)} test reports")
print()


# ============================================================================
# STEP 5: ASK THE MODEL TO WRITE THE IMPRESSIONS
# ============================================================================
# We feed the findings in small groups and let the model generate its answer.
#
# The generation settings matter a lot for the score:
#   num_beams=4          the model explores 4 possible sentences and keeps the
#                        best one, instead of just grabbing the first word that
#                        looks good. This alone lifts ROUGE noticeably.
#   max_length=64        do not let it ramble - real impressions are short
#   min_length=5         do not let it output an empty or one-word answer
#   no_repeat_ngram_size=3   stops it repeating the same phrase over and over,
#                            which is a common failure of small summarisers
#   early_stopping=True  stop as soon as all 4 beams have finished a sentence

print("STEP 5: Generating summaries...")
print("  (this is the slow part - please wait)")
print()

generated_impressions = []

generation_batch_size = 4   # small, to stay safe on memory

start_time = time.time()

# torch.no_grad() tells PyTorch "we are not learning right now, so do not
# bother remembering how to correct the model". This saves a lot of memory
# and makes generation noticeably faster.
with torch.no_grad():

    for start_index in range(0, len(findings_list), generation_batch_size):

        # Take the next small group of findings
        batch_of_findings = findings_list[start_index:start_index + generation_batch_size]

        # Turn the text into numbers
        inputs = tokenizer(
            batch_of_findings,
            max_length=256,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )

        inputs = {key: value.to(device) for key, value in inputs.items()}

        # The model writes its answer
        outputs = model.generate(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_length=64,
            min_length=5,
            num_beams=4,
            no_repeat_ngram_size=3,
            early_stopping=True
        )

        # Turn the numbers back into readable text
        for output in outputs:
            text = tokenizer.decode(output, skip_special_tokens=True)
            generated_impressions.append(text.strip())

        # Progress update every 20 reports
        done = len(generated_impressions)
        if done % 20 == 0 or done == len(findings_list):
            minutes_so_far = (time.time() - start_time) / 60
            print(f"  Generated {done} of {len(findings_list)}"
                  f"  |  {minutes_so_far:.1f} min elapsed")

print()
print(f"  Finished generating in {(time.time() - start_time) / 60:.1f} minutes")
print()


# ============================================================================
# STEP 6: SCORE THE ANSWERS WITH ROUGE
# ============================================================================
# use_stemmer=True means "running" and "run" count as a match. This is
# standard practice for ROUGE and gives a fairer score.

print("STEP 6: Scoring the answers with ROUGE...")

scorer = rouge_scorer.RougeScorer(
    ['rouge1', 'rouge2', 'rougeL'],
    use_stemmer=True
)

# We will collect every individual score, then average them at the end
rouge1_scores = []
rouge2_scores = []
rougeL_scores = []

for generated, real in zip(generated_impressions, real_impressions):

    scores = scorer.score(real, generated)   # (reference first, then prediction)

    # .fmeasure is the F1 score - the number the brief asks about
    rouge1_scores.append(scores['rouge1'].fmeasure)
    rouge2_scores.append(scores['rouge2'].fmeasure)
    rougeL_scores.append(scores['rougeL'].fmeasure)

# Work out the average of each list
average_rouge1 = sum(rouge1_scores) / len(rouge1_scores)
average_rouge2 = sum(rouge2_scores) / len(rouge2_scores)
average_rougeL = sum(rougeL_scores) / len(rougeL_scores)

print("  Scoring complete")
print()


# ============================================================================
# STEP 7: SHOW SOME EXAMPLES
# ============================================================================
# Numbers alone do not tell you whether the output actually reads sensibly.
# Always eyeball a few real examples.

print("=" * 70)
print("EXAMPLE OUTPUTS")
print("=" * 70)
print()

for i in range(min(3, len(generated_impressions))):
    print(f"--- Example {i + 1} ---")
    print(f"FINDINGS (input):     {findings_list[i][:150]}...")
    print(f"REAL impression:      {real_impressions[i]}")
    print(f"MODEL's impression:   {generated_impressions[i]}")
    print(f"This one's ROUGE-1 F1: {rouge1_scores[i]:.4f}")
    print()


# ============================================================================
# STEP 8: THE FINAL RESULTS
# ============================================================================

print("=" * 70)
print("FINAL ROUGE SCORES (F1)")
print("=" * 70)
print()
print(f"  Reports tested: {len(generated_impressions)}")
print()
print(f"  ROUGE-1 F1:  {average_rouge1:.4f}   (single word overlap)")
print(f"  ROUGE-2 F1:  {average_rouge2:.4f}   (two word phrase overlap)")
print(f"  ROUGE-L F1:  {average_rougeL:.4f}   (longest matching sequence)")
print()

# The assessment brief requires a ROUGE F1 of 0.3 or higher
threshold = 0.3

print("-" * 70)
print(f"The assessment requires a ROUGE F1 of {threshold} or higher.")
print("-" * 70)

if average_rouge1 >= threshold:
    print(f"  PASS - ROUGE-1 F1 of {average_rouge1:.4f} is above {threshold}")
else:
    print(f"  BELOW THRESHOLD - ROUGE-1 F1 of {average_rouge1:.4f} is under {threshold}")
    print()
    print("  Things to try, easiest first:")
    print("    1. Train for 2 epochs instead of 1 (change num_epochs in train_model.py)")
    print("    2. Switch model_name to 'facebook/bart-large-cnn', which already")
    print("       knows how to summarise before we even start fine-tuning")
    print("    3. Adjust max_length / min_length in the generate() settings above")

print()
print("Evaluation finished.")