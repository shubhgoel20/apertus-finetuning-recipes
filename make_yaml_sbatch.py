import os
import yaml
from pathlib import Path
from itertools import product

def write_training_yaml(
    learning_rate: float,
    num_train_epochs: float,
    lora_rank: int,
    dropout: float,
    warmup_ratio: float,
    config_dir: str,
    id: int
):
    print(f"Generating config for id={id} lr={learning_rate}, epochs={num_train_epochs}, r={lora_rank}, do={dropout}, wu={warmup_ratio}")

    lora_alpha = 2 * lora_rank
    config = {
        # Model
        "model_name_or_path": "/users/ddixit/scratch/apertus-project/huggingface_cache/models--swiss-ai--Apertus-8B-Instruct-2509/snapshots/cdb3e4f4ad41e0cc394bb92c302ac2eed57e9586",
        "attn_implementation": "flash_attention_2",
        "dtype": "bfloat16",

        # Dataset
        "dataset_name": "/users/ddixit/scratch/apertus-project/huggingface_cache/json_datasets/my_train.jsonl",
        "dataset_num_proc": 12,

        # Hyperparameters
        "learning_rate": learning_rate,
        "gradient_checkpointing": True,
        "num_train_epochs": num_train_epochs,
        "logging_steps": 1,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 16,
        "max_grad_norm": 1.0,

        # Evaluation & Early Stopping
        "eval_strategy": "steps",
        "eval_steps": 15,
        "per_device_eval_batch_size": 1,
        "save_strategy": "steps",
        "save_steps": 15,
        "save_total_limit": 5,
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,

        # LoRA / PEFT
        "use_peft": True,
        "lora_r": lora_rank,
        "lora_alpha": lora_alpha,
        "lora_dropout": dropout,
        "lora_target_modules": "all-linear",

        # Seq length & scheduler
        "max_length": 4096,
        "warmup_ratio": warmup_ratio,
        "lr_scheduler_type": "cosine_with_min_lr",
        "lr_scheduler_kwargs": {
            "min_lr_rate": 0.1
        },

        # Output & logging
        "output_dir": f"/users/ddixit/scratch/apertus-project/output_{id}",
        "report_to": "none",
        "seed": 42,
    }

    filename = (f"sft_lora_{id}.yaml")

    filepath = os.path.join(config_dir, filename)

    with open(filepath, "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)

    return filepath



SBATCH_BASE = """#!/bin/bash
#SBATCH --job-name=apertus_finetune_{id}
#SBATCH --account=large-sc-2
#SBATCH --time=02:30:00
#SBATCH --partition=normal
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=1       # One manager process per node
#SBATCH --gpus-per-node=4         # 4 GPUs per node
#SBATCH --cpus-per-task=72
#SBATCH --output=apertus_finetune_{id}.log
#SBATCH --error=apertus_finetune_{id}.err

# --- 1. Environment & Paths ---
PROJECT_DIR="/users/ddixit/scratch/apertus-project"
export SLURM_CPU_BIND="none"

# Cache locations
export HF_HOME="$PROJECT_DIR/huggingface_cache"
export TRITON_CACHE_DIR="$PROJECT_DIR/triton_cache"
export TORCH_EXTENSIONS_DIR="$PROJECT_DIR/torch_extensions"
mkdir -p "$TRITON_CACHE_DIR" "$TORCH_EXTENSIONS_DIR"

echo "Clearing stale cache locks..."
find "$TRITON_CACHE_DIR" -name "*.lock" -mmin +60 -delete 2>/dev/null || true
find "$TORCH_EXTENSIONS_DIR" -name "*.lock" -mmin +60 -delete 2>/dev/null || true
find "$HF_HOME" -name "*.lock" -mmin +60 -delete 2>/dev/null || true

cleanup() {
    echo "Cleaning up processes..."
    pkill -P $$ 2>/dev/null || true
    srun --overlap bash -c "pkill -u $USER python; pkill -u $USER accelerate" 2>/dev/null || true
}
trap cleanup EXIT SIGTERM SIGINT

export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export NCCL_TIMEOUT=1800
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_DISTRIBUTED_DEBUG=DETAIL

export NCCL_SOCKET_IFNAME=hsn,ib
export NCCL_IB_DISABLE=0

MASTER_NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_ADDR=$(srun --nodes=1 --ntasks=1 -w "$MASTER_NODE" hostname --ip-address)
MASTER_PORT=29500

CONFIG_FILE="$PROJECT_DIR/apertus-finetuning-recipes/configs/zero3_multinode.yaml"

GPUS_PER_NODE=4
NUM_PROCESSES=$((SLURM_NNODES * GPUS_PER_NODE))

CMD="source $PROJECT_DIR/.venv/bin/activate && \
    export PYTHONPATH=$PROJECT_DIR/.venv/lib/python3.13/site-packages:\\$PYTHONPATH && \
    accelerate launch \
    --config_file $CONFIG_FILE \
    --num_processes $NUM_PROCESSES \
    --num_machines $SLURM_NNODES \
    --machine_rank \\$SLURM_PROCID \
    --main_process_ip $MASTER_ADDR \
    --main_process_port $MASTER_PORT \
    --mixed_precision bf16 \
    --dynamo_backend no \
    /users/ddixit/scratch/apertus-project/apertus-finetuning-recipes/sft_train.py \
    --config /users/ddixit/scratch/apertus-project/apertus-finetuning-recipes/configs/sft_lora_{id}.yaml \
    --ddp_timeout 5400"

srun bash -c "$CMD"
"""

def write_sbatch_script(script_path: str, id: int):
    content = SBATCH_BASE.replace("{id}", str(id))
    path = f"{script_path}/apertus_finetune_{id}.sbatch"
    with open(path, "w") as f:
        f.write(content)






import sys

if __name__ == "__main__":


    learning_rates = [5e-5]
    num_train_epochs_list = [2.0, 3.0]
    lora_ranks = [8, 16, 32]
    dropouts = [0.1]
    warmup_ratios = [0.1, 0.2]

    config_dir = "/users/ddixit/scratch/apertus-project/apertus-finetuning-recipes/configs/"
    script_dir = "/users/ddixit/scratch/apertus-project/apertus-finetuning-recipes/"


    id = 0
    for lr, epochs, r, d, w in product(
        learning_rates,
        num_train_epochs_list,
        lora_ranks,
        dropouts,
        warmup_ratios,
    ):
        path = write_training_yaml(
            learning_rate=lr,
            num_train_epochs=epochs,
            lora_rank=r,
            dropout=d,
            warmup_ratio=w,
            config_dir=config_dir,
            id=id
        )
        print(f"gen {path} ")
        write_sbatch_script(script_path=script_dir, id=id)
        print(f'gen sbatch script for id={id}')
        id+=1

    sys.exit(id)