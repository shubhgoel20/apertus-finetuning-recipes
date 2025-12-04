#!/bin/bash
#SBATCH --job-name=apertus_finetune
#SBATCH --account=large-sc-2
#SBATCH --time=02:30:00
#SBATCH --partition=normal
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --output=apertus_finetune_%j.log
#SBATCH --error=apertus_finetune_%j.err
#SBATCH --uenv=pytorch/v2.6.0:v1
#SBATCH --view=default

# Set project directory
PROJECT_DIR="/users/sgoel/scratch/apertus-project"

# Disable Slurm CPU binding to prevent "outside of job step allocation" errors
export SLURM_CPU_BIND="none"

# Set cache locations
export HF_HOME="$PROJECT_DIR/huggingface_cache"
export TRITON_CACHE_DIR="$PROJECT_DIR/triton_cache"
mkdir -p "$TRITON_CACHE_DIR"   

export TORCH_EXTENSIONS_DIR="$PROJECT_DIR/torch_extensions"
mkdir -p "$TORCH_EXTENSIONS_DIR"
export TOKENIZERS_PARALLELISM=false

# Set offline usage for Hugging Face to avoid internet access during training (CHECK IF NEEDED)
# export HF_DATASETS_OFFLINE=1
# export HF_HUB_OFFLINE=1

# Set master address and port for distributed training
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=29500

echo "Master node: $MASTER_ADDR"
echo "Master port: $MASTER_PORT"


# Configuration file: use zero3.yaml or zero3_multinode.yaml for multi-node training
CONFIG_FILE="$PROJECT_DIR/apertus-finetuning-recipes/configs/zero3_multinode.yaml"       

# Calculate total processes (3 nodes * 4 GPUs = 12 processes)
GPUS_PER_NODE=4
NUM_PROCESSES=$((SLURM_NNODES * GPUS_PER_NODE))

echo "Launching $NUM_PROCESSES processes across $SLURM_NNODES nodes..."

# Launch training
CMD="source $PROJECT_DIR/.venv/bin/activate && \
    export PYTHONPATH=$PROJECT_DIR/.venv/lib/python3.13/site-packages:\$PYTHONPATH && \
    echo 'Node \${SLURM_PROCID} launched' && \
    accelerate launch \
    --config_file $CONFIG_FILE \
    --num_processes $NUM_PROCESSES \
    --num_machines $SLURM_NNODES \
    --machine_rank \$SLURM_PROCID \
    --main_process_ip $MASTER_ADDR \
    --main_process_port $MASTER_PORT \
    --mixed_precision bf16 \
    --dynamo_backend no \
    sft_train.py \
    --config configs/sft_lora.yaml \
    --ddp_timeout 1800"

srun bash -c "$CMD"
