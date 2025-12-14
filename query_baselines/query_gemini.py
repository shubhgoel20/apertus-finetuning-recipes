import os
import re
import sys
import time
import google.generativeai as genai
from datasets import load_dataset
from google.api_core import retry

import google.generativeai as genai
import os

# api_key = os.environ.get("GOOGLE_API_KEY")
# genai.configure(api_key=api_key)

# print("Checking available models for your API key...")
# try:
#     for m in genai.list_models():
#         if 'generateContent' in m.supported_generation_methods:
#             print(f"- {m.name}")
# except Exception as e:
#     print(f"Error listing models: {e}")

# --- CONFIGURATION ---
# 1. Set your API Key
# You can hardcode it here or export it in your terminal: export GOOGLE_API_KEY="your_key"
API_KEY = os.environ.get("GOOGLE_API_KEY") 

if not API_KEY:
    print("Error: GOOGLE_API_KEY environment variable not found.")
    print("Please set it via: export GOOGLE_API_KEY='your_actual_key'")
    sys.exit(1)

# 2. Configure Gemini
genai.configure(api_key=API_KEY)

# 3. Model Configuration
# Note: "gemini-1.5-pro" has a lower rate limit (2 RPM) on the free tier compared to "gemini-1.5-flash" (15 RPM).
# If this script runs too slowly, consider switching strictly to "gemini-1.5-flash".
MODEL_NAME = "gemini-2.5-flash" 
SLEEP_TIME = 4

INSTRUCTION_TEXT = (
    "Please answer with one of the option in the bracket. "
    "Write reasoning in between <analysis></analysis>. "
    "Write answer in between <answer></answer>."
)

generation_config = {
    "temperature": 0.0,        # Greedy decoding for evaluation
    "top_p": 1.0,
    "max_output_tokens": 4096,
}

print(f"--- Starting Inference using {MODEL_NAME} ---")

# --- DATASET LOADING ---
# Keeping your original dataset path logic
dataset_path = "/users/sgoel/scratch/apertus-project/huggingface_cache/datasets--mamachang--medical-reasoning/snapshots/3e784b9fee85b9d8b6974449b3dfe0737ac9ecba"
 # Switched to hub path for portability, change back to local path if needed
print(f"Loading dataset ...")
try:
    # If using local path, keep your original string. If using HF Hub, use the string above.
    # ds = load_dataset("/users/sgoel/scratch/...", split="train") 
    ds = load_dataset(dataset_path, split="train") 
except Exception as e:
    print(f"Error loading dataset: {e}")
    sys.exit(1)

print("Dataset loaded successfully.")

# --- HELPER FUNCTIONS ---

def get_answer_from_text(text):
    """
    Robustly extracts the option letter (A-E).
    Strategy 1: Look for <answer> [Letter]
    Strategy 2: Look for </analysis> ... [Letter] (Fallback if model skips tag)
    """
    if not text: return None
    
    # 1. Standard Case: Look for <answer> followed by A-E
    match = re.search(r"<answer>\s*([A-E])", text, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    
    # 2. Fallback Case: Look for </analysis> followed by A-E
    match = re.search(r"</analysis>\s*([A-E])", text, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    return None

# --- INITIALIZE MODEL ---
model = genai.GenerativeModel(
    model_name=MODEL_NAME,
    system_instruction=INSTRUCTION_TEXT,
    generation_config=generation_config
)

# --- EVALUATION LOOP ---
correct = 0
total = 0
error_count = 0

print(f">>> Starting Evaluation...", flush=True)

# Select range (first 100 examples for test)
eval_subset = ds.shuffle(seed=42).select(range(100)) 

for i, ex in enumerate(eval_subset):
    total += 1
    print("="*50, flush=True)
    print(f"Example {i}", flush=True)

    prompt = ex["input"]

    try:
        # Generate Content
        # We add a retry policy for transient errors, but manual rate limiting is below
        response = model.generate_content(prompt)
        
        # Check if response was blocked (Safety filters)
        if response.prompt_feedback.block_reason:
            print(f"Response Blocked: {response.prompt_feedback.block_reason}", flush=True)
            response_text = ""
        else:
            response_text = response.text

    except Exception as e:
        print(f"API Error: {e}", flush=True)
        response_text = ""
        error_count += 1
        # If we hit a 429 (Quota), wait a bit longer
        if "429" in str(e):
            print(f"Usage limit exceeded. Sleeping for {120}s...", flush=True)
            time.sleep(120)

    # Extract Answers
    model_answer_letter = get_answer_from_text(response_text)
    
    # Handle Ground Truth
    ground_truth_letter = get_answer_from_text(ex["output"])
    if not ground_truth_letter:
        try:
            ground_truth_letter = ex["output"].split("<answer>")[-1].strip()[0]
        except:
            ground_truth_letter = None

    # Print Debug Info
    print(f"Full Response:\n{response_text}", flush=True) # Uncomment to see full reasoning
    print(f"Model Letter: {model_answer_letter}", flush=True)
    print(f"Ground Truth: {ground_truth_letter}", flush=True)

    # Calculate Accuracy
    if model_answer_letter and ground_truth_letter:
        if model_answer_letter == ground_truth_letter:
            correct += 1
            print(">> RESULT: CORRECT", flush=True)
        else:
            print(">> RESULT: WRONG", flush=True)
    else:
        print(">> RESULT: PARSE ERROR / BLOCKED", flush=True)

    # --- RATE LIMIT HANDLING ---
    # Free tier Gemini 1.5 Pro allows ~2 requests per minute (RPM).
    # We must sleep to avoid 429 errors.
    # If using paid tier, you can reduce this sleep significantly.
    print(f"Sleeping {SLEEP_TIME}s for rate limit compliance (Free Tier)...", flush=True) 
    time.sleep(SLEEP_TIME) 

    # Periodic logging
    if (i + 1) % 5 == 0:
        print(f"\n--- Progress: {i+1}/{len(eval_subset)} ---")
        print(f"Current Accuracy: {correct/total:.2%}")
        sys.stdout.flush()

print("="*50)
print(f"Final Correct: {correct}")
print(f"Final Total: {total}")
print(f"Final Accuracy: {correct/total:.4f}")
print(f"Total API Errors: {error_count}")