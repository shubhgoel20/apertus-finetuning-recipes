#!/bin/bash
#SBATCH --job-name=apertus_finetune_35
#SBATCH --account=large-sc-2
#SBATCH --time=02:30:00
#SBATCH --partition=normal
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=1       # One manager process per node
#SBATCH --gpus-per-node=4         # 4 GPUs per node
#SBATCH --cpus-per-task=72
#SBATCH --output=apertus_finetune_35.log
#SBATCH --error=apertus_finetune_35.err

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

CMD="source $PROJECT_DIR/.venv/bin/activate &&     export PYTHONPATH=$PROJECT_DIR/.venv/lib/python3.13/site-packages:\$PYTHONPATH &&     accelerate launch     --config_file $CONFIG_FILE     --num_processes $NUM_PROCESSES     --num_machines $SLURM_NNODES     --machine_rank \$SLURM_PROCID     --main_process_ip $MASTER_ADDR     --main_process_port $MASTER_PORT     --mixed_precision bf16     --dynamo_backend no     /users/ddixit/scratch/apertus-project/apertus-finetuning-recipes/sft_train.py     --config /users/ddixit/scratch/apertus-project/apertus-finetuning-recipes/configs/sft_lora_35.yaml     --ddp_timeout 5400"

srun bash -c "$CMD"
