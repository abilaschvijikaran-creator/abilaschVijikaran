# Evaluate the medical summarisation model
import pandas as pd
import torch
from transformers import BartForConditionalGeneration, BartTokenizer
from rouge_score import rouge_scorer
import time

print("=" * 70)
print("EVALUATING THE MEDICAL SUMMARISATION MODEL")
print("=" * 70)
print()

# Settings you can change
how_many_to_test = 100
model_folder = "models/medical_summarizer"
print("STEP 1: Settings")
print(f"  Model folder: {model_folder}")
print(f"  Number of test reports: {how_many_to_test if how_many_to_test else 'ALL'}")
print()


# Hardware we can use
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
print("STEP 3: Loading our trained model...")

tokenizer = BartTokenizer.from_pretrained(model_folder)
model = BartForConditionalGeneration.from_pretrained(model_folder)

model = model.to(device)
model.eval()

print(f"  Model loaded from: {model_folder}")
print()


# ============================================================================
# STEP 4: LOAD THE TEST DATA
# ============================================================================
print("STEP 4: Loading the test data...")

test_data = pd.read_csv("data/processed/test.csv")
test_data = test_data.dropna()

if how_many_to_test is not None:
    test_data = test_data.head(how_many_to_test)

findings_list = list(test_data['findings'])
real_impressions = list(test_data['impression'])

print(f"  Loaded {len(findings_list)} test reports")
print()


# ============================================================================
# STEP 5: ASK THE MODEL TO WRITE THE IMPRESSIONS
# ===========================================================================
print("STEP 5: Generating summaries...")
print("  (this is the slow part - please wait)")
print()

generated_impressions = []
generation_batch_size = 4   
start_time = time.time()
with torch.no_grad():

    for start_index in range(0, len(findings_list), generation_batch_size):
        batch_of_findings = findings_list[start_index:start_index + generation_batch_size]
        inputs = tokenizer(
            batch_of_findings,
            max_length=256,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        inputs = {key: value.to(device) for key, value in inputs.items()}
        outputs = model.generate(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_length=64,
            min_length=5,
            num_beams=4,
            no_repeat_ngram_size=3,
            early_stopping=True
        )
        for output in outputs:
            text = tokenizer.decode(output, skip_special_tokens=True)
            generated_impressions.append(text.strip())
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
print("STEP 6: Scoring the answers with ROUGE...")
scorer = rouge_scorer.RougeScorer(
    ['rouge1', 'rouge2', 'rougeL'],
    use_stemmer=True
)

rouge1_scores = []
rouge2_scores = []
rougeL_scores = []

for generated, real in zip(generated_impressions, real_impressions):

    scores = scorer.score(real, generated)
    rouge1_scores.append(scores['rouge1'].fmeasure)
    rouge2_scores.append(scores['rouge2'].fmeasure)
    rougeL_scores.append(scores['rougeL'].fmeasure)

average_rouge1 = sum(rouge1_scores) / len(rouge1_scores)
average_rouge2 = sum(rouge2_scores) / len(rouge2_scores)
average_rougeL = sum(rougeL_scores) / len(rougeL_scores)

print("  Scoring complete")
print()


# ============================================================================
# STEP 7: SHOW SOME EXAMPLES
# ============================================================================
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

# the 3 sample findings the brief asks us to test
sample_findings = [
    "The trachea is midline. The cardiomediastinal silhouette is normal. The lungs are clear, without evidence of acute infiltrate or effusion. There is no pneumothorax. The visualized bony structures reveal no acute abnormalities.",
    "The lungs are clear. Heart size and mediastinal contours are normal. No osseous abnormalities.",
    "AP and lateral views were obtained. Bibasilar atelectasis and small left-sided pleural effusion. Stable cardiomegaly. No pneumothorax. Mild pulmonary vascular congestion."
]

print("PREDICTIONS ON THE 3 SAMPLE FINDINGS FROM THE BRIEF")
print()

for i, text in enumerate(sample_findings):
    inputs = tokenizer(text, max_length=256, truncation=True, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model.generate(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'],
                             max_length=64, min_length=5, num_beams=4, no_repeat_ngram_size=3, early_stopping=True)
    result = tokenizer.decode(out[0], skip_special_tokens=True)
    print(f"Sample {i + 1}: {text[:90]}...")
    print(f"  Impression: {result.strip()}\n")
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
threshold = 0.3

print("-" * 70)
print(f"The assessment requires a ROUGE F1 of {threshold} or higher.")
print("-" * 70)

if average_rouge1 >= threshold:
    print(f"  PASS - ROUGE-1 F1 of {average_rouge1:.4f} is above {threshold}")
else:
    print(f"  BELOW THRESHOLD - {average_rouge1:.4f} is under {threshold}")

print()
print("Evaluation finished.")
