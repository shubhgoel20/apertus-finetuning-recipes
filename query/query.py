import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "/users/sgoel/scratch/apertus-project/huggingface_cache/models--swiss-ai--Apertus-8B-Instruct-2509/snapshots/cdb3e4f4ad41e0cc394bb92c302ac2eed57e9586"

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
prompt = "Q:An 8-year-old boy is brought to the pediatrician by his mother with nausea, vomiting, and decreased frequency of urination. He has acute lymphoblastic leukemia for which he received the 1st dose of chemotherapy 5 days ago. His leukocyte count was 60,000/mm3 before starting chemotherapy. The vital signs include: pulse 110/min, temperature 37.0°C (98.6°F), and blood pressure 100/70 mm Hg. The physical examination shows bilateral pedal edema. Which of the following serum studies and urinalysis findings will be helpful in confirming the diagnosis of this condition?\
{'A': 'Hyperkalemia, hyperphosphatemia, hypocalcemia, and extremely elevated creatine kinase (MM)', 'B': 'Hyperkalemia, hyperphosphatemia, hypocalcemia, hyperuricemia, urine supernatant pink, and positive for heme', 'C': 'Hyperuricemia, hyperkalemia, hyperphosphatemia, lactic acidosis, and urate crystals in the urine', 'D': 'Hyperuricemia, hyperkalemia, hyperphosphatemia, and urinary monoclonal spike', 'E': 'Hyperuricemia, hyperkalemia, hyperphosphatemia, lactic acidosis, and oxalate crystals'}, Instruction: Please answer with one of the option in the bracket. Write your answer as follows: Answer: <your answer>"

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
    temperature=0.5,
    top_p=0.9
)

# 6. Decode and Print
output_ids = generated_ids[0][len(model_inputs.input_ids[0]) :]
response = tokenizer.decode(output_ids, skip_special_tokens=True)

print("-" * 30)
print(f"Generated Output:\n{response}")
print("-" * 30)
print("--- Inference Complete ---")