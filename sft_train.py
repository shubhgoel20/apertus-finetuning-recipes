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

import os
import sys
from datasets import load_dataset
from transformers import AutoTokenizer
from trl import (
    ModelConfig,
    ScriptArguments,
    SFTConfig,
    SFTTrainer,
    TrlParser,
    get_peft_config,
)

# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------
os.environ["HF_HOME"] = "/users/ddixit/scratch/apertus-project/huggingface_cache"


def convert_to_messages(example):
    """
    Fallback: Maps 'instruction', 'input', 'output' to the standard messages format.
    Only used if the data isn't pre-formatted.
    """
    instruction = example.get("instruction", "")
    user_content = example.get("input", "")
    assistant_content = example.get("output", "")
    
    messages = []
    
    if instruction:
        messages.append({"role": "system", "content": instruction})
    
    messages.append({"role": "user", "content": user_content})
    messages.append({"role": "assistant", "content": assistant_content})

    return {"messages": messages}


def main(script_args, training_args, model_args):
    print(">>> DEBUG: Process started. Setting up...", flush=True)
    
    # ------------------------
    # 1. Load Tokenizer
    # ------------------------
    print(">>> DEBUG: Loading Tokenizer...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        local_files_only=True,
        trust_remote_code=True,
    )
    tokenizer.pad_token = tokenizer.eos_token
    print(">>> DEBUG: Tokenizer loaded.", flush=True)

    # ------------------------
    # 2. Load Dataset (Local or Hub)
    # ------------------------
    print(f">>> DEBUG: Loading Dataset target: {script_args.dataset_name}...", flush=True)

    # LOGIC FIX: Check if it's a local file (ending in .json or .jsonl)
    if script_args.dataset_name.endswith(".json") or script_args.dataset_name.endswith(".jsonl"):
        print(">>> DEBUG: Detected local file. Loading via 'json' loader...", flush=True)
        # Load directly as the training split
        dataset = load_dataset("json", data_files=script_args.dataset_name, split="train")

        # Check if validation file exists (e.g., my_train.jsonl -> my_val.jsonl)
        val_file = script_args.dataset_name.replace("_train.jsonl", "_val.jsonl").replace("train.jsonl", "val.jsonl")
        if os.path.exists(val_file) and training_args.eval_strategy != "no":
            print(f">>> DEBUG: Found validation file: {val_file}", flush=True)
            val_dataset = load_dataset("json", data_files=val_file, split="train")
        else:
            val_dataset = None
    else:
        print(">>> DEBUG: Detected Hub dataset. Loading...", flush=True)
        dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config, split=script_args.dataset_train_split)
        val_dataset = None

    print(">>> DEBUG: Dataset loaded.", flush=True)

    # ------------------------
    # 3. Smart Format Check
    # ------------------------
    # We check if the data is *already* in the right format (from prepare_data.py)
    if "messages" in dataset.column_names:
        print(">>> DEBUG: Dataset already has 'messages' column. SKIPPING CONVERSION.", flush=True)
        # No mapping needed, data is ready.
    else:
        print(">>> DEBUG: 'messages' column missing. Running conversion...", flush=True)
        dataset = dataset.map(
            convert_to_messages,
            remove_columns=dataset.column_names,
            desc="Formatting dataset"
        )

    # Apply same format check to validation dataset if it exists
    if val_dataset is not None:
        if "messages" in val_dataset.column_names:
            print(">>> DEBUG: Validation dataset already has 'messages' column. SKIPPING CONVERSION.", flush=True)
        else:
            print(">>> DEBUG: 'messages' column missing in validation. Running conversion...", flush=True)
            val_dataset = val_dataset.map(
                convert_to_messages,
                remove_columns=val_dataset.column_names,
                desc="Formatting validation dataset"
            )

    print(f">>> DEBUG: Format confirmed. Columns: {dataset.column_names}", flush=True)

    # Optional: Verify first example structure
    try:
        print(f"\n>>> DEBUG: First Training Example:\n{dataset[0]['messages'][:2]}...", flush=True)
    except:
        pass
    print("-" * 50, flush=True)

    # ------------------------
    # 4. Initialize Trainer
    # ------------------------
    # Use the separate validation file if it exists, otherwise split the training data
    if training_args.eval_strategy != "no" and training_args.eval_strategy is not None:
        if val_dataset is not None:
            print(f">>> DEBUG: Using separate validation file with {len(val_dataset)} examples.", flush=True)
            train_dataset = dataset
            eval_dataset = val_dataset
        elif "test" not in dataset:
            print(">>> DEBUG: No validation file found. Creating validation split (10%)...", flush=True)
            dataset = dataset.train_test_split(test_size=0.1)
            train_dataset = dataset["train"]
            eval_dataset = dataset["test"]
        else:
            train_dataset = dataset
            eval_dataset = None
    else:
        train_dataset = dataset
        eval_dataset = None

    print(f">>> DEBUG: Training on {len(train_dataset)} examples.", flush=True)
    if eval_dataset is not None:
        print(f">>> DEBUG: Validating on {len(eval_dataset)} examples.", flush=True)

    trainer = SFTTrainer(
        model=model_args.model_name_or_path,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=get_peft_config(model_args),
    )

    # ------------------------
    # 5. Train & Save
    # ------------------------
    print(">>> DEBUG: Starting Trainer...", flush=True)
    trainer.train()
    
    save_path = training_args.output_dir
    trainer.save_model(save_path)
    print(f">>> DEBUG: Model saved to {save_path}", flush=True)

    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)


if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, SFTConfig, ModelConfig))
    script_args, training_args, model_args, _ = parser.parse_args_and_config(
        return_remaining_strings=True
    )
    main(script_args, training_args, model_args)