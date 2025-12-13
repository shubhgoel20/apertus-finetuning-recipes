import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import json
import sys
import re

dataset_path = "/users/sgoel/scratch/apertus-project/huggingface_cache/datasets--mamachang--medical-reasoning/snapshots/3e784b9fee85b9d8b6974449b3dfe0737ac9ecba"
ds = load_dataset(dataset_path, split="train")


BASE_MODEL_PATH = "/users/sgoel/scratch/apertus-project/huggingface_cache/models--swiss-ai--Apertus-8B-Instruct-2509/snapshots/cdb3e4f4ad41e0cc394bb92c302ac2eed57e9586"

print(f"--- Starting Inference for {BASE_MODEL_PATH} ---")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA Device Name: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Device Count: {torch.cuda.device_count()}")

# Confirm HF_HOME and offline status (still good to see for debugging)
hf_home_env = os.environ.get("HF_HOME", "Not Set")
hf_offline_env = os.environ.get("HF_HUB_OFFLINE", "0")
print(f"HF_HOME environment variable: {hf_home_env}")
print(f"HF_HUB_OFFLINE environment variable: {hf_offline_env}")

# Check transformers version for compatibility
import transformers
print(f"Transformers version: {transformers.__version__}")

# Load tokenizer
print(f"Loading tokenizer from {BASE_MODEL_PATH}...")
tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL_PATH,
    local_files_only=True,
    trust_remote_code=True
)
tokenizer.pad_token = tokenizer.eos_token
print(" Tokenizer loaded")

# Load base model
print(f"Loading model from {BASE_MODEL_PATH} (device_map='auto', torch_dtype=bfloat16)...") # Now using local path
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_PATH,
    device_map="auto",
    dtype=torch.bfloat16,
    local_files_only=True,
    trust_remote_code=True
)

print("check")
print(f"Model loaded on device: {base_model.device}", flush=True)
print(f"Model dtype: {base_model.dtype}", flush=True)

import glob
import json
LORA_ADAPTER_PATH = "/users/sgoel/scratch/apertus-project/output"
# Step 1: Check output directory structure
print("\n[1] Checking output directory structure...", flush=True)
print(f"Output directory: {LORA_ADAPTER_PATH}", flush=True)

if not os.path.exists(LORA_ADAPTER_PATH):
    print(f"ERROR: Output directory does not exist: {LORA_ADAPTER_PATH}", flush=True)
    exit(1)

required_files = [
    "adapter_config.json",
    "adapter_model.safetensors",  # or adapter_model.bin
    "README.md"
]

print("\nChecking for required files:", flush=True)
for file in required_files:
    file_path = os.path.join(LORA_ADAPTER_PATH, file)
    # Check both safetensors and bin for model file
    if file == "adapter_model.safetensors" and not os.path.exists(file_path):
        file_path = os.path.join(LORA_ADAPTER_PATH, "adapter_model.bin")

    if os.path.exists(file_path):
        size = os.path.getsize(file_path) / (1024 * 1024)  # MB
        print(f"{file}: {size:.2f} MB", flush=True)
    else:
        print(f"{file}: NOT FOUND", flush=True)

# Check for trainer_state.json (contains training metrics)
checkpoint_pattern = os.path.join(LORA_ADAPTER_PATH, "checkpoint-*")
checkpoints = sorted(glob.glob(checkpoint_pattern))

trainer_state_path = None
for ckpt in checkpoints:
    candidate = os.path.join(ckpt, "trainer_state.json")
    if os.path.isfile(candidate):
        trainer_state_path = candidate
        break

if os.path.exists(trainer_state_path):
    print(f"trainer_state.json: Found", flush=True)

    # Step 2: Read training metrics
    print("\n[2] Training Metrics:", flush=True)
    with open(trainer_state_path, 'r') as f:
        trainer_state = json.load(f)

    if "log_history" in trainer_state and len(trainer_state["log_history"]) > 0:
        print("\nTraining Loss History:", flush=True)
        for entry in trainer_state["log_history"]:
            if "loss" in entry:
                step = entry.get("step", "N/A")
                loss = entry.get("loss", "N/A")
                epoch = entry.get("epoch", "N/A")
                print(f"  Step {step} | Epoch {epoch:.2f} | Loss: {loss:.4f}", flush=True)

        # Show final metrics
        final_entry = trainer_state["log_history"][-1]
        print(f"\n Final Training Metrics:", flush=True)
        for key, value in final_entry.items():
            if key not in ["step", "epoch"]:
                print(f"  {key}: {value}", flush=True)

    print(f"\ntotal training steps: {trainer_state.get('global_step', 'N/A')}", flush=True)
    print(f"Best metric: {trainer_state.get('best_metric', 'N/A')}", flush=True)
else:
    print(f"  ⚠️  trainer_state.json: NOT FOUND (training metrics unavailable)", flush=True)


# Step 3: Load and test the model
print("\n[3] Loading Fine-Tuned Model...", flush=True)
print(f"Base model: {BASE_MODEL_PATH}", flush=True)
print(f"LoRA adapter: {LORA_ADAPTER_PATH}", flush=True)

sys.stdout.flush()
try:
    
    from peft import PeftModel
    
    # 1. CRITICAL FIX: Set Padding Side to LEFT for Generation
    # Llama/Mistral generation fails or slows down with Right Padding
    tokenizer.padding_side = "left" 
    
    # Load LoRA adapter
    print("\nLoading LoRA adapter...")
    model = PeftModel.from_pretrained(base_model, LORA_ADAPTER_PATH)
    
    # 2. CRITICAL FIX: Merge Adapter into Base Model
    # This removes the runtime overhead of LoRA
    print("Merging adapter weights into base model...")
    model = model.merge_and_unload()
    print("Model merged! Inference speed will now match base model.")

    # 3. Switch to eval mode
    model.eval()

    # Get model info
    print("\nModel Information:")
    print(f"  Device: {model.device}")
    print(f"  Dtype: {model.dtype}")

    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable %: {100 * trainable_params / total_params:.2f}%")

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

        return None

    # 1. Define the exact instruction used in training
    instruction_text = "Please answer with one of the option in the bracket. Write reasoning in between <analysis></analysis>. Write answer in between <answer></answer>."

    correct = 0
    total = 0

    # Ensure pad token is set for generation
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print(f">>> Starting Evaluation...", flush=True)

    # Select range (e.g., first 50 examples)
    eval_subset = ds.shuffle(seed=42).select(range(10))
    
    for i, ex in enumerate(eval_subset):
         # Debug Printing
        print("="*50, flush=True)
        print(f"Example {i}", flush=True)
        total += 1
    
        # 2. Construct Messages exactly like the training script
        messages = [
            {"role": "system", "content": instruction_text},
            {"role": "user", "content": ex["input"]}
        ]

        # Apply chat template
        prompt_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        model_inputs = tokenizer([prompt_text], return_tensors="pt", add_special_tokens=False).to(model.device)

        # 3. Generate with Stopping Criteria
        # stop_strings requires tokenizer to be passed to generate.
        # It will stop generation once "</answer>" is produced.
        generated_ids = model.generate(
            **model_inputs,
            use_cache=True,
            max_new_tokens=4096,
            do_sample=False,       # Greedy decoding is usually better for strict evaluation
            temperature=0.0,       # Turn off randomness for reproducibility
            tokenizer=tokenizer,
            stop_strings=["</answer>", "```"] # Stop at closing tag
        )

        # Decode response
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]) :]
        response = tokenizer.decode(output_ids, skip_special_tokens=True)

        # print(f"Generated Response :\n{response}")

        # 4. Extract Answers using the new function
        model_answer_letter = get_answer_from_text(response)
        
        # Handle the Ground Truth format
        # Your ground truth format is: ... <answer>\nE: ...
        # We can use the same function on ground truth to be safe
        ground_truth_letter = get_answer_from_text(ex["output"])
        
        # If function fails on ground_truth, fall back to simple split
        if not ground_truth_letter:
            ground_truth_letter = ex["output"].split("<answer>")[-1].strip()[0]

       
        print(f"Full Response:\n{response}", flush=True) # Uncomment to see full reasoning
        print(f"Model Letter: {model_answer_letter}", flush=True)
        print(f"Ground Truth: {ground_truth_letter}", flush=True)

        # 5. Calculate Accuracy
        if model_answer_letter and ground_truth_letter:
            if model_answer_letter == ground_truth_letter:
                correct += 1
                print(">> RESULT: CORRECT", flush=True)
            else:
                print(">> RESULT: WRONG", flush=True)
        else:
            print(">> RESULT: PARSE ERROR (Model failed to generate valid format)", flush=True)

        # Periodic logging
        if (i + 1) % 5 == 0:
            print(f"\n--- Progress: {i+1}/{len(eval_subset)} ---")
            print(f"Current Accuracy: {correct/total:.2%}")
            sys.stdout.flush()

    print("="*50)
    print(f"Final Correct: {correct}")
    print(f"Final Total: {total}")
    print(f"Final Accuracy: {correct/total:.4f}")

    print("\n" + "=" * 80)
    print("✅ VERIFICATION COMPLETE - MODEL IS WORKING!")
    print("=" * 80)
    print("\nYour fine-tuned model has been successfully verified and tested.")
    print(f"You can load it using:")
    print(f"  Base model: {BASE_MODEL_PATH}")
    print(f"  LoRA adapter: {LORA_ADAPTER_PATH}")

    sys.stdout.flush()
    
except Exception as e:
    print(f"\n❌ ERROR during model loading or inference:")
    print(f"  {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    exit(1)
