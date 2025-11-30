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

# Disable Slurm CPU binding to prevent "outside of job step allocation" errors
export SLURM_CPU_BIND="none"

# Set HF Cache to scratch (optional but good practice)
export HF_HOME="/users/mmeciani/scratch/apertus/huggingface_cache"
export TRITON_CACHE_DIR="/users/mmeciani/scratch/apertus/triton_cache"

# ---------------------------------------------------------
# 2. Network Setup for Distributed Training
# ---------------------------------------------------------
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=29500

echo "Master node: $MASTER_ADDR"
echo "Master port: $MASTER_PORT"

# ---------------------------------------------------------
# 3. Execution
# ---------------------------------------------------------
# Explicitly set the config path
CONFIG_FILE="configs/zero3.yaml" # Use zero3.yaml (or zero3_multinode.yaml if you have created it)
DATASET_NAME="medalpaca_formatted" # Use the local dataset we fixed earlier

# Calculate total processes (3 nodes * 4 GPUs = 12 processes)
GPUS_PER_NODE=4
NUM_PROCESSES=$((SLURM_NNODES * GPUS_PER_NODE))

echo "Launching $NUM_PROCESSES processes across $SLURM_NNODES nodes..."

# We use 'srun' to launch the training script on every node.
# Note: We activate the venv INSIDE the srun command to ensure it exists on worker nodes.


# Launch training
CMD="source .venv/bin/activate && \
    export PYTHONPATH=$PWD/.venv/lib/python3.13/site-packages:\$PYTHONPATH && \
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