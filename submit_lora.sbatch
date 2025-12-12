#!/bin/bash
#SBATCH --job-name=apertus_finetune
#SBATCH --account=large-sc-2
#SBATCH --time=02:30:00
#SBATCH --partition=normal
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=1       # One manager process per node
#SBATCH --gpus-per-node=4         # 4 GPUs per node
#SBATCH --cpus-per-task=72
#SBATCH --output=apertus_finetune_%j.log
#SBATCH --error=apertus_finetune_%j.err

# --- 1. Environment & Paths ---
PROJECT_DIR="/users/mmeciani/scratch/apertus-project"
export SLURM_CPU_BIND="none"

# Cache locations
export HF_HOME="$PROJECT_DIR/huggingface_cache"
export TRITON_CACHE_DIR="$PROJECT_DIR/triton_cache"
export TORCH_EXTENSIONS_DIR="$PROJECT_DIR/torch_extensions"
mkdir -p "$TRITON_CACHE_DIR" "$TORCH_EXTENSIONS_DIR"

# Force offline mode
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1

# --- 2. CRITICAL: Network & Debugging Configuration ---
# 1. Enable logging so we see WHERE it hangs.
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL

# 2. Force NCCL to use the High-Speed Network.
# On CSCS/Cray machines, this is usually 'hsn' or 'ib'. 
# We exclude loopback (lo) and docker bridges to prevent confusion.
export NCCL_SOCKET_IFNAME=hsn,ib,eth
export NCCL_IB_DISABLE=0

# 3. Distributed Setup
# Get the first node name
MASTER_NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
# Resolve that name to an IP address (safer than using the name directly)
MASTER_ADDR=$(srun --nodes=1 --ntasks=1 -w "$MASTER_NODE" hostname --ip-address)
MASTER_PORT=29500

echo "Master Node: $MASTER_NODE"
echo "Master IP:   $MASTER_ADDR"
echo "Master Port: $MASTER_PORT"

# --- 3. Launch Configuration ---
CONFIG_FILE="$PROJECT_DIR/apertus-finetuning-recipes/configs/zero3_multinode.yaml"       

# Total GPUs = Nodes * GPUs per Node
GPUS_PER_NODE=4
NUM_PROCESSES=$((SLURM_NNODES * GPUS_PER_NODE))

echo "Launching $NUM_PROCESSES processes across $SLURM_NNODES nodes..."

# --- 4. The Launch Command ---
# We use 'srun' to start one copy of 'accelerate launch' on every node.
# 'accelerate launch' then spawns the 4 local GPU processes.
CMD="source $PROJECT_DIR/.venv/bin/activate && \
    export PYTHONPATH=$PROJECT_DIR/.venv/lib/python3.13/site-packages:\$PYTHONPATH && \
    echo 'Node \${SLURM_PROCID} initializing...' && \
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
    --ddp_timeout 5400" # Increased timeout for initial heavy loading

# Launch
srun bash -c "$CMD"