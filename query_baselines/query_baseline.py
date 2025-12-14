import torch
import os
import re
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import sys

# --- ARGUMENT PARSING ---
parser = argparse.ArgumentParser(description="Evaluate Llama 8B Instruct Models")
parser.add_argument("--model_path", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct", help="Hugging Face model ID or local path")
parser.add_argument("--dataset_path", type=str, default="/users/sgoel/scratch/apertus-project/huggingface_cache/datasets--mamachang--medical-reasoning/snapshots/3e784b9fee85b9d8b6974449b3dfe0737ac9ecba", help="Hugging Face dataset ID or local path")
parser.add_argument("--device", type=str, default="auto", help="Device to use (cuda, cpu, or auto)")
args = parser.parse_args()

# --- CONFIGURATION ---
MODEL_PATH = args.model_path
DATASET_PATH = args.dataset_path

# System Instruction strictly following the requested format
INSTRUCTION_TEXT = (
    "Please answer with one of the option in the bracket. "
    "Write reasoning in between <analysis></analysis>. "
    "Write answer in between <answer></answer>."
)

print(f"--- Starting Evaluation ---")
print(f"Model: {MODEL_PATH}")
print(f"Dataset: {DATASET_PATH}")
print(f"CUDA Available: {torch.cuda.is_available()}")

# --- SETUP ---

# 1. Load Tokenizer
print(f"Loading tokenizer...")
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
except Exception as e:
    print(f"Error loading tokenizer: {e}")
    exit(1)

# CRITICAL FOR LLAMA GENERATION:
# 1. Padding must be on the left for causal generation
tokenizer.padding_side = "left"

# 2. Fix pad_token issues common in Llama 3
if tokenizer.pad_token is None:
    if tokenizer.eos_token:
        tokenizer.pad_token = tokenizer.eos_token
        print(f"Info: Pad token was None, set to EOS token: {tokenizer.eos_token}")
    else:
        # Fallback if even EOS is missing (rare)
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# 2. Load Model
print(f"Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    device_map=args.device,
    torch_dtype=torch.bfloat16, # Efficient and standard for Llama 3
    trust_remote_code=True,
    # attn_implementation="flash_attention_2" # Uncomment if you have Ampere GPU (A100/A10/3090/4090) for speed
)
model.eval()

print(f"Model loaded on device: {model.device}")

# --- HELPER FUNCTIONS ---

def get_answer_from_text(text):
    """
    Robustly extracts the option letter (A-E).
    Strategy 1: Look for <answer> [Letter]
    Strategy 2: Look for </analysis> ... [Letter] (Fallback if model skips tag)
    """
    # 1. Standard Case: Look for <answer> followed by A-E
    # Matches: <answer>A, <answer> A, <answer>\nC
    match = re.search(r"<answer>\s*([A-E])", text, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    
    # 2. Fallback Case: Look for </analysis> followed by A-E
    # Matches: </analysis>\nC:, </analysis> \n A
    match = re.search(r"</analysis>\s*([A-E])", text, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    
    #Fallback: Sometimes models write "Answer: A" inside the tag
    match_loose = re.search(r"<answer>.*?([A-E]).*?</answer>", text, re.IGNORECASE | re.DOTALL)
    if match_loose:
        return match_loose.group(1).upper()

    return None

def get_ground_truth(text):
    """
    Extracts ground truth from the dataset output field.
    """
    if not text: return None
    # Dataset often has explicit tags, try to parse them first
    parsed = get_answer_from_text(text)
    if parsed: return parsed

    # If dataset is just a single letter "A", return it
    clean_text = text.strip()
    if len(clean_text) == 1 and clean_text in "ABCDE":
        return clean_text
    
    return None

# --- EVALUATION LOOP ---

print(f"Loading dataset...")
try:
    # Try loading as a path first (if it exists locally), else treat as hub ID
    if os.path.exists(DATASET_PATH):
        ds = load_dataset(DATASET_PATH, split="train")
    else:
        ds = load_dataset(DATASET_PATH, split="train")
except Exception as e:
    print(f"Dataset load failed: {e}")
    exit(1)

# Select range (first 100 for test)
eval_subset = ds.shuffle(seed=42).select(range(100)) 

correct = 0
total = 0

print(f">>> Starting Evaluation Loop...")
sys.stdout.flush()
for i, ex in enumerate(eval_subset):
    total += 1
    print("-" * 50)
    print(f"Example {i+1}/{len(eval_subset)}")

    # 1. Format Prompt
    # Llama 3 Instruct expects specific chat template formatting
    messages = [
        {"role": "system", "content": INSTRUCTION_TEXT},
        {"role": "user", "content": ex["input"]}
    ]
    
    # apply_chat_template handles <|begin_of_text|>, <|start_header_id|>, etc.
    try:
        prompt_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    except Exception as e:
        print(f"Template Error: {e}. Ensure model has a chat_template in tokenizer_config.json")
        continue
    
    # 2. Tokenize
    inputs = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False).to(model.device)

    # 3. Generate
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=512,     # Lowered from 4096 for speed (reasoning usually fits in 512-1024)
            do_sample=False,        # Greedy decoding for reproducibility
            temperature=None,
            top_p=None,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            stop_strings=["</answer>", "</s>", "<|eot_id|>"], # Stop immediately after answer or turn end
            tokenizer=tokenizer # Required for stop_strings
        )

    # 4. Decode
    # Slice input tokens to get only the generated response
    output_ids = generated_ids[0][len(inputs.input_ids[0]):]
    response = tokenizer.decode(output_ids, skip_special_tokens=True)

    # 5. Extract & Compare
    model_answer = get_answer_from_text(response)
    ground_truth = get_ground_truth(ex["output"])

    # Logging
    print(f"Output: {response.strip()}") # Uncomment to debug text generation
    print(f"Model: {model_answer} | Truth: {ground_truth}", end=" ")

    if model_answer and ground_truth:
        if model_answer == ground_truth:
            correct += 1
            print(">> CORRECT")
        else:
            print(">> WRONG")
    else:
        print(">> PARSE ERROR")

    # Periodic Status
    if (i + 1) % 10 == 0:
        print(f"--- Current Accuracy: {correct/total:.2%} ({correct}/{total}) ---")

    sys.stdout.flush()
# --- FINAL RESULTS ---
print("="*50)
print(f"Final Result for {MODEL_PATH}")
print(f"Correct: {correct}")
print(f"Total: {total}")
print(f"Accuracy: {correct/total:.4f}")
print("="*50)
sys.stdout.flush()