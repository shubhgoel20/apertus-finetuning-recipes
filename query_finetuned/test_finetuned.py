import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import json
from peft import PeftModel

dataset_path = "/users/mmeciani/scratch/apertus-project/huggingface_cache/datasets--mamachang--medical-reasoning/snapshots/3e784b9fee85b9d8b6974449b3dfe0737ac9ecba"
ds = load_dataset(dataset_path, split="train")

# Base model path (same as before)
base_model_path = "/users/mmeciani/scratch/apertus-project/huggingface_cache/models--swiss-ai--Apertus-8B-Instruct-2509/snapshots/cdb3e4f4ad41e0cc394bb92c302ac2eed57e9586"

# Fine-tuned LoRA adapter path (from your training output)
finetuned_model_path = "/users/mmeciani/scratch/apertus-project/output"

print(f"--- Starting Inference for {base_model_path} ---")
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

# 2. Load Tokenizer
print(f"Loading tokenizer from {base_model_path}...") # Now using local path
tokenizer = AutoTokenizer.from_pretrained(
    base_model_path, # Use the direct local path
    local_files_only=True, # Keep this, it reinforces offline mode
    trust_remote_code=True
)
tokenizer.pad_token = tokenizer.eos_token

# 3. Load Model
print(f"Loading model from {base_model_path} (device_map='auto', torch_dtype=bfloat16)...") # Now using local path
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_path, # Use the direct local path
    device_map="auto",
    dtype=torch.bfloat16,
    local_files_only=True, # Keep this
    trust_remote_code=True
)
print(f"Model loaded on device: {base_model.device}")
print(f"Model dtype: {base_model.dtype}")

print(f"Loading LoRA adapter from {finetuned_model_path}...")
model = PeftModel.from_pretrained(base_model, finetuned_model_path)
print("✅ Fine-tuned model loaded successfully!")

# 4. Prepare Input
correct = 0
total = 0
instruction = " Instruction: Please answer with one of the option in the bracket. Write your answer as follows: Answer: <your answer>"
for i, ex in enumerate(ds.shuffle(seed=42).select(range(100))):
    # print("=== example", i, "===")
    total+=1
    prompt = ex["input"]+instruction
    
    messages = [
        {"role": "user", "content": prompt}
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    # print(f"Input prompt prepared:\n{text}")

    model_inputs = tokenizer([text], return_tensors="pt", add_special_tokens=False).to(model.device)

    # 5. Generate
    # print("Generating response...")
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=1024,
        do_sample=True,
        temperature=0.5,
        top_p=0.9
    )

    # 6. Decode and Print
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]) :]
    response = tokenizer.decode(output_ids, skip_special_tokens=True)

    model_answer = response.split(" ")
    if len(model_answer) < 2 :
        continue
    model_answer = model_answer[1]
    if len(model_answer) < 1 :
        continue

    model_answer = model_answer[0]
    ground_truth = ex["output"].split("\n<answer>\n")[-1][0]
    if model_answer == ground_truth:
        correct+=1
    
    # print(f"model answer: {model_answer}, ground truth: {ground_truth}")
    # print("-" * 30)
    # print(f"Generated Output:\n{response}")
    # print("-" * 30)
    # print("--- Inference Complete ---")
    if i%100 == 0:
        print("=== example", i, "===")
        print(f"correct: {correct}")
        print(f"total: {total}")
        print(f"accuracy: {correct/total}")

print(f"correct: {correct}")
print(f"total: {total}")
print(f"accuracy: {correct/total}")