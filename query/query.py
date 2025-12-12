import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "/users/mmeciani/scratch/apertus-project/huggingface_cache/models--swiss-ai--Apertus-8B-Instruct-2509/snapshots/cdb3e4f4ad41e0cc394bb92c302ac2eed57e9586"


print(f"--- Starting Inference for {model_path} ---")
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
print(f"Loading tokenizer from {model_path}...") # Now using local path
tokenizer = AutoTokenizer.from_pretrained(
    model_path, # Use the direct local path
    local_files_only=True, # Keep this, it reinforces offline mode
    trust_remote_code=True
)
tokenizer.pad_token = tokenizer.eos_token

# 3. Load Model
print(f"Loading model from {model_path} (device_map='auto', torch_dtype=bfloat16)...") # Now using local path
model = AutoModelForCausalLM.from_pretrained(
    model_path, # Use the direct local path
    device_map="auto",
    dtype=torch.bfloat16,
    local_files_only=True, # Keep this
    trust_remote_code=True
)
print(f"Model loaded on device: {model.device}")
print(f"Model dtype: {model.dtype}")

# 4. Prepare Input
prompt = "What is suggested by high PCWP worsening after IVF and arrhythmias? Provide a precise, concise answer of maximum 2 sentences"
messages = [
    {"role": "user", "content": prompt}
]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)
print(f"Input prompt prepared:\n{text}")

model_inputs = tokenizer([text], return_tensors="pt", add_special_tokens=False).to(model.device)

# 5. Generate
print("Generating response...")
generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=1024,
    do_sample=True,
    temperature=0.7,
    top_p=0.9
)

# 6. Decode and Print
output_ids = generated_ids[0][len(model_inputs.input_ids[0]) :]
response = tokenizer.decode(output_ids, skip_special_tokens=True)

print("-" * 30)
print(f"Generated Output:\n{response}")
print("-" * 30)
print("--- Inference Complete ---")
