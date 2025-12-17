import torch
import os
import sys
import re
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from peft import PeftModel
from tqdm import tqdm
import time

# --- FAST EVALUATION CONFIGURATION ---
NUM_EXAMPLES_PER_TASK = 100  # Reduce from 100 to 25 for 4x speedup
MAX_NEW_TOKENS = 1024         # Enough for CoT reasoning
BATCH_SIZE = 4               # Process multiple examples at once
QUICK_MODE = False            # If True, test only 3 tasks instead of 6

# --- Configuration ---
DATASET_PATH = "/users/mmeciani/scratch/apertus-project/huggingface_cache/datasets--cais--mmlu/snapshots/c30699e8356da336a370243923dbaf21066bb9fe"
BASE_MODEL_PATH = "/users/mmeciani/scratch/apertus-project/huggingface_cache/models--swiss-ai--Apertus-8B-Instruct-2509/snapshots/cdb3e4f4ad41e0cc394bb92c302ac2eed57e9586"
LORA_ADAPTER_PATH = "/users/mmeciani/scratch/apertus-project/output/checkpoint-60"

# MMLU Medical Subsets (SAME AS BASELINE!)
if QUICK_MODE:
    TASKS = [
        "clinical_knowledge",
        "professional_medicine",
        "anatomy"
    ]
    print("🚀 QUICK MODE: Testing only 3 tasks for fast iteration")
else:
    TASKS = [
        "clinical_knowledge",
        "professional_medicine",
        "anatomy",
        "medical_genetics",
        "college_medicine",
        "college_biology"
    ]

print(f"--- Fast Fine-tuned Model MMLU Evaluation ---")
print(f"Settings: {NUM_EXAMPLES_PER_TASK} examples/task, {MAX_NEW_TOKENS} max tokens, batch_size={BATCH_SIZE}")
print(f"Transformers version: {transformers.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")

# Load Tokenizer
print(f"Loading tokenizer from {BASE_MODEL_PATH}...")
tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL_PATH,
    local_files_only=True,
    trust_remote_code=True
)
tokenizer.padding_side = "left"  # Important for batch generation
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load Base Model
print(f"Loading base model from {BASE_MODEL_PATH}...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_PATH,
    device_map="auto",
    dtype=torch.bfloat16,
    local_files_only=True,
    trust_remote_code=True
)

# Check output directory
print(f"\nChecking LoRA adapter at {LORA_ADAPTER_PATH}...", flush=True)
if not os.path.exists(LORA_ADAPTER_PATH):
    print(f"ERROR: Output directory does not exist: {LORA_ADAPTER_PATH}")
    sys.exit(1)

# Load and Merge Adapter
print("Loading and Merging LoRA Adapter...", flush=True)
try:
    model = PeftModel.from_pretrained(base_model, LORA_ADAPTER_PATH)
    print("Merging adapter weights into base model...")
    model = model.merge_and_unload()

    torch.cuda.empty_cache()
    model.eval()
    print(f"Fine-tuned model ready! Device: {model.device}, Dtype: {model.dtype}\n")

except Exception as e:
    print(f"Error loading PEFT adapter: {e}")
    sys.exit(1)

# Helper: Format MMLU Example (SAME AS BASELINE!)
def format_mmlu_prompt(example):
    """Format to match training format exactly."""
    options = example['choices']
    labels = ["A", "B", "C", "D"]

    prompt_text = f"Question: {example['question']}\n\nOptions:\n"
    for label, opt in zip(labels, options):
        prompt_text += f"{label}) {opt}\n"

    prompt_text += "\nProvide your step-by-step analysis, then state your answer as 'Answer: X' where X is the option letter."
    return prompt_text

def extract_answer(response):
    """Extract answer letter from response"""
    match = re.search(r"Answer:\s*([A-D])", response, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    # Fallback: Check if the very last word is A/B/C/D
    last_word = response.split()[-1].strip(".").upper()
    if last_word in ["A", "B", "C", "D"]:
        return last_word
    return None

# Batch Evaluation Function
def evaluate_batch(examples, ground_truths):
    """Evaluate a batch of examples at once"""
    # Prepare all prompts
    all_messages = []
    for ex in examples:
        prompt = format_mmlu_prompt(ex)
        messages = [
            {"role": "system", "content": "You are a medical expert. Analyze medical questions step-by-step and select the best option."},
            {"role": "user", "content": prompt}
        ]
        all_messages.append(messages)

    # Apply chat template to all
    texts = [
        tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        for msgs in all_messages
    ]

    # Tokenize as batch
    model_inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        add_special_tokens=False
    ).to(model.device)

    # Generate for batch
    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.pad_token_id
        )

    # Decode and evaluate
    correct = 0
    parse_errors = []

    for i, (gen_ids, input_len) in enumerate(zip(generated_ids, model_inputs.input_ids)):
        output_ids = gen_ids[len(input_len):]
        response = tokenizer.decode(output_ids, skip_special_tokens=True).strip()

        model_answer = extract_answer(response)
        gt_letter = ["A", "B", "C", "D"][ground_truths[i]]

        if model_answer is None:
            # Parse error - save the question and response
            parse_errors.append({
                'question': examples[i]['question'],
                'response': response,
                'ground_truth': gt_letter
            })
        elif model_answer == gt_letter:
            correct += 1

    return correct, parse_errors

# Main Evaluation Loop
results = {}
all_parse_errors = []
total_time = 0
total_correct = 0
total_wrong = 0
total_parse_errors = 0

print("="*60)
for task in TASKS:
    print(f"Processing Task: {task}...", flush=True)

    try:
        ds = load_dataset(DATASET_PATH, task, split="test", trust_remote_code=True)
    except Exception as e:
        print(f"⚠️ Failed to load {task}: {e}", flush=True)
        continue

    correct = 0
    total = 0
    task_parse_errors = []

    # Limit to NUM_EXAMPLES_PER_TASK
    eval_set = ds.select(range(min(NUM_EXAMPLES_PER_TASK, len(ds))))

    # Process in batches with progress bar
    start_time = time.time()

    for i in tqdm(range(0, len(eval_set), BATCH_SIZE), desc=task):
        batch_end = min(i + BATCH_SIZE, len(eval_set))
        batch_examples = [eval_set[j] for j in range(i, batch_end)]
        batch_gts = [ex['answer'] for ex in batch_examples]

        batch_correct, batch_parse_errors = evaluate_batch(batch_examples, batch_gts)
        correct += batch_correct
        total += len(batch_examples)

        # Add task name to parse errors
        for err in batch_parse_errors:
            err['task'] = task
            task_parse_errors.append(err)
            all_parse_errors.append(err)

    task_time = time.time() - start_time
    total_time += task_time

    wrong = total - correct - len(task_parse_errors)
    acc = correct / total if total > 0 else 0
    results[task] = {
        'accuracy': acc,
        'correct': correct,
        'wrong': wrong,
        'parse_errors': len(task_parse_errors),
        'total': total
    }

    total_correct += correct
    total_wrong += wrong
    total_parse_errors += len(task_parse_errors)

    print(f"--> {task}: {acc:.2%} ({correct} correct, {wrong} wrong, {len(task_parse_errors)} parse errors) in {task_time:.1f}s", flush=True)

# Final Report
print("\n" + "="*60)
print("FINE-TUNED MODEL - MMLU MEDICAL RESULTS")
print("="*60)
avg_acc = 0
for task, task_results in results.items():
    acc = task_results['accuracy']
    correct = task_results['correct']
    wrong = task_results['wrong']
    parse_errors = task_results['parse_errors']
    total = task_results['total']
    print(f"{task:<25} : {acc:.2%} ({correct}✓ {wrong}✗ {parse_errors}⚠️  / {total})", flush=True)
    avg_acc += acc

if len(results) > 0:
    grand_total = total_correct + total_wrong + total_parse_errors
    print("-" * 60)
    print(f"AVERAGE ACCURACY          : {avg_acc / len(results):.2%}", flush=True)
    print(f"Total Correct             : {total_correct}/{grand_total} ({total_correct/grand_total:.2%})", flush=True)
    print(f"Total Wrong               : {total_wrong}/{grand_total} ({total_wrong/grand_total:.2%})", flush=True)
    print(f"Total Parse Errors        : {total_parse_errors}/{grand_total} ({total_parse_errors/grand_total:.2%})", flush=True)
    print("-" * 60)
    print(f"Total Evaluation Time     : {total_time:.1f}s", flush=True)
    print(f"Throughput                : {sum([NUM_EXAMPLES_PER_TASK]*len(results))/total_time:.1f} examples/sec", flush=True)
print("="*60)

# Print Parse Errors in Detail
if len(all_parse_errors) > 0:
    print(f"\n{'='*60}")
    print(f"PARSE ERRORS DETAILS ({len(all_parse_errors)} total)")
    print("="*60)
    for idx, err in enumerate(all_parse_errors, 1):
        print(f"\n[Parse Error {idx}] Task: {err['task']}")
        print(f"Question: {err['question'][:100]}..." if len(err['question']) > 100 else f"Question: {err['question']}")
        print(f"Expected Answer: {err['ground_truth']}")
        print(f"Model Response:")
        print("-" * 60)
        # Print response with truncation if too long
        if len(err['response']) > 500:
            print(f"{err['response'][:500]}...")
            print(f"[... truncated, total length: {len(err['response'])} chars]")
        else:
            print(err['response'])
        print("=" * 60)
else:
    print("\n✅ No parse errors! All responses were correctly formatted.")

if NUM_EXAMPLES_PER_TASK < 100:
    print(f"\n💡 Running with {NUM_EXAMPLES_PER_TASK} examples per task (quick mode)")
    print(f"   For final results, set NUM_EXAMPLES_PER_TASK=100 and QUICK_MODE=False")
