import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 1. Setup
model_name = "swiss-ai/Apertus-8B-Instruct-2509"

print(f"Loading {model_name}...")

# 2. Load Tokenizer
# Added 'local_files_only=True' to prevent the OfflineModeIsEnabled error
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    local_files_only=True
)

# 3. Load Model
# - device_map="auto": Automatically places model on GPU
# - torch_dtype=torch.bfloat16: Uses half the memory of default float32 (crucial for 8B models)
# - local_files_only=True: Respects your offline environment
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    local_files_only=True
)

# 4. Prepare Input
prompt = "Answer this question truthfully and in a coincise manner: What is suggested by high PCWP worsening after IVF and arrhythmias?"
messages = [
    {"role": "user", "content": prompt}
]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)

# Move inputs to the same device as the model
model_inputs = tokenizer([text], return_tensors="pt", add_special_tokens=False).to(model.device)

# 5. Generate
print("Generating response...")
generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=1024,  # 32k is the limit, but 1024 is safer for a quick test
    do_sample=True,       # 'True' adds variety (chat-like), 'False' is deterministic
    temperature=0.7,       # Controls creativity (only works if do_sample=True)
    top_p=0.9             # Nucleus sampling for diversity (only works if do_sample=True)  
)

# 6. Decode and Print
# Slice the output to remove the input prompt from the result
output_ids = generated_ids[0][len(model_inputs.input_ids[0]) :]
response = tokenizer.decode(output_ids, skip_special_tokens=True)

print("-" * 30)
print(f"Output:\n{response}")
print("-" * 30)