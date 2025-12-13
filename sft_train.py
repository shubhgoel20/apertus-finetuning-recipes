# Copyright 2020-2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
accelerate launch \
    --config_file configs/zero3.yaml \
    sft_train.py \
    --config configs/sft_lora.yaml \
    --model_name_or_path swiss-ai/Apertus-8B-Instruct-2509 \
"""

import os

os.environ["HF_HOME"] = "/users/sgoel/scratch/apertus-project/huggingface_cache"


from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import (
    ModelConfig,
    ScriptArguments,
    SFTConfig,
    SFTTrainer,
    TrlParser,
    get_peft_config,
)

def convert_to_messages(example):       # Specific version to work with medalpaca
    """
    Maps 'instruction', 'input', 'output' to the standard messages format.
    Handles cases where instruction might be empty.
    """
    # 1. Extract content safely
    instruction = example.get("instruction", "")
    user_content = example.get("input", "")
    assistant_content = example.get("output", "")
    
    # 2. Build the messages list
    messages = []
    
    # Add System Prompt (Critical for setting behavior)
    # Even if it's generic, it tells the model "This is an instruction-following task"
    if instruction:
        messages.append({"role": "system", "content": instruction})
    
    # Add User/Assistant turns
    messages.append({"role": "user", "content": user_content})
    messages.append({"role": "assistant", "content": assistant_content})

    return {"messages": messages}


def main(script_args, training_args, model_args):
    print(">>> DEBUG: Process started. Setting up...", flush=True)
    # ------------------------
    # Load model & tokenizer
    # ------------------------
    #Set base directory to store model
    store_base_dir = "./" #os.getenv("STORE")

    print(f">>> DEBUG: Loading Model from {model_args.model_name_or_path}...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        dtype=model_args.dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        attn_implementation=model_args.attn_implementation,
        local_files_only=True, 
        trust_remote_code=True,    
    )
    print(">>> DEBUG: Model loaded successfully.", flush=True)

    print(">>> DEBUG: Loading Tokenizer...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        local_files_only=True,
        trust_remote_code=True,
    )
    tokenizer.pad_token = tokenizer.eos_token
    print(">>> DEBUG: Tokenizer loaded.", flush=True)
    # --------------
    # Load dataset
    # --------------
    print(f">>> DEBUG: Loading Dataset {script_args.dataset_name}...", flush=True)
    dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)
    if(script_args.dataset_name != "HuggingFaceH4/Multilingual-Thinking"):
        column_names = dataset[script_args.dataset_train_split].column_names    
        dataset = dataset.map(
            convert_to_messages,
            remove_columns=column_names, # IMPORTANT: Remove old columns
            desc="Formatting dataset"
        )
        print(f">>> DEBUG: Format complete. Columns are now: {dataset[script_args.dataset_train_split].column_names}", flush=True)
        print(f"\n>>> DEBUG: Verify Data Format (First Example):", flush=True)
        print(dataset[script_args.dataset_train_split][0]["messages"], flush=True)
        print("-" * 50, flush=True)

    print(">>> DEBUG: Dataset loaded.", flush=True)

    # -------------
    # Train model
    # -------------
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split]
        if training_args.eval_strategy != "no"
        else None,
        processing_class=tokenizer,
        peft_config=get_peft_config(model_args),
    )

    print(">>> DEBUG: Starting Trainer...", flush=True)
    trainer.train()
    trainer.save_model(os.path.join(store_base_dir, training_args.output_dir))
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)


if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, SFTConfig, ModelConfig))
    script_args, training_args, model_args, _ = parser.parse_args_and_config(
        return_remaining_strings=True
    )
    main(script_args, training_args, model_args)
