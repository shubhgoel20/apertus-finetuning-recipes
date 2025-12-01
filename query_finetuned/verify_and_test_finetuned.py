import os
import glob
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Paths - adjust if needed
BASE_MODEL_PATH = "/path/to/model/config.json"
LORA_ADAPTER_PATH = "/path/to/output"

print("=" * 80)
print("FINE-TUNED MODEL VERIFICATION SCRIPT")
print("=" * 80)

# Step 1: Check output directory structure
print("\n[1] Checking output directory structure...")
print(f"Output directory: {LORA_ADAPTER_PATH}")

if not os.path.exists(LORA_ADAPTER_PATH):
    print(f"❌ ERROR: Output directory does not exist: {LORA_ADAPTER_PATH}")
    exit(1)

required_files = [
    "adapter_config.json",
    "adapter_model.safetensors",  # or adapter_model.bin
    "README.md"
]

print("\nChecking for required files:")
for file in required_files:
    file_path = os.path.join(LORA_ADAPTER_PATH, file)
    # Check both safetensors and bin for model file
    if file == "adapter_model.safetensors" and not os.path.exists(file_path):
        file_path = os.path.join(LORA_ADAPTER_PATH, "adapter_model.bin")

    if os.path.exists(file_path):
        size = os.path.getsize(file_path) / (1024 * 1024)  # MB
        print(f"  ✅ {file}: {size:.2f} MB")
    else:
        print(f"  ❌ {file}: NOT FOUND")

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
    print(f"  ✅ trainer_state.json: Found")

    # Step 2: Read training metrics
    print("\n[2] Training Metrics:")
    with open(trainer_state_path, 'r') as f:
        trainer_state = json.load(f)

    if "log_history" in trainer_state and len(trainer_state["log_history"]) > 0:
        print("\nTraining Loss History:")
        for entry in trainer_state["log_history"]:
            if "loss" in entry:
                step = entry.get("step", "N/A")
                loss = entry.get("loss", "N/A")
                epoch = entry.get("epoch", "N/A")
                print(f"  Step {step} | Epoch {epoch:.2f} | Loss: {loss:.4f}")

        # Show final metrics
        final_entry = trainer_state["log_history"][-1]
        print(f"\n📊 Final Training Metrics:")
        for key, value in final_entry.items():
            if key not in ["step", "epoch"]:
                print(f"  {key}: {value}")

    print(f"\n✅ Total training steps: {trainer_state.get('global_step', 'N/A')}")
    print(f"✅ Best metric: {trainer_state.get('best_metric', 'N/A')}")
else:
    print(f"  ⚠️  trainer_state.json: NOT FOUND (training metrics unavailable)")


# Step 3: Load and test the model
print("\n[3] Loading Fine-Tuned Model...")
print(f"Base model: {BASE_MODEL_PATH}")
print(f"LoRA adapter: {LORA_ADAPTER_PATH}")

try:
    # Check CUDA availability
    print(f"\nCUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")

    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL_PATH,
        local_files_only=True,
        trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    print("✅ Tokenizer loaded")

    # Load base model
    print("\nLoading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        local_files_only=True,
        trust_remote_code=True
    )
    print("✅ Base model loaded")

    # Load LoRA adapter
    print("\nLoading LoRA adapter...")
    model = PeftModel.from_pretrained(base_model, LORA_ADAPTER_PATH)
    print("✅ LoRA adapter loaded successfully!")

    # Get model info
    print("\n📊 Model Information:")
    print(f"  Device: {model.device}")
    print(f"  Dtype: {model.dtype}")

    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable %: {100 * trainable_params / total_params:.2f}%")

    
    # Step 4: Run test inference
    print("\n[4] Running Test Inference...")

    test_prompts = [
        "What is the capital of Switzerland?",
        "Explain quantum computing in simple terms.",
        "Write a short poem about mountains."
    ]

    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n--- Test {i} ---")
        print(f"Prompt: {prompt}")

        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        model_inputs = tokenizer([text], return_tensors="pt", add_special_tokens=False).to(model.device)

        print("Generating response...")
        with torch.no_grad():
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=256,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id
            )

        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):]
        response = tokenizer.decode(output_ids, skip_special_tokens=True)

        print(f"Response:\n{response}")
        print("-" * 40)

    print("\n" + "=" * 80)
    print("✅ VERIFICATION COMPLETE - MODEL IS WORKING!")
    print("=" * 80)
    print("\nYour fine-tuned model has been successfully verified and tested.")
    print(f"You can load it using:")
    print(f"  Base model: {BASE_MODEL_PATH}")
    print(f"  LoRA adapter: {LORA_ADAPTER_PATH}")

    
except Exception as e:
    print(f"\n❌ ERROR during model loading or inference:")
    print(f"  {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    exit(1)
