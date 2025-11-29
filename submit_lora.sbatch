#!/bin/bash
#SBATCH --job-name=apertus_finetune
#SBATCH --account=large-sc-2
#SBATCH --time=02:30:00
#SBATCH --partition=normal
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --output=apertus_finetune_%j.log
#SBATCH --error=apertus_finetune_%j.err

# HF cache
export HF_HOME="/users/$USER/scratch/apertus/huggingface_cache"

# Conda environment
source /users/$USER/miniconda3/etc/profile.d/conda.sh
conda activate apertus_env

# Multi-node setup
ACCEL_PROCS=$(( SLURM_NNODES * SLURM_GPUS_PER_NODE ))
MAIN_ADDR=$(scontrol show hostnames $SLURM_NODELIST | head -n1)
MAIN_PORT=12802

# Launch training
CMD="accelerate launch \
    --config_file /users/$USER/scratch/apertus/configs/zero3_multinode.yaml \
    --num_machines=$SLURM_NNODES \
    --num_processes=$((SLURM_NNODES * SLURM_GPUS_PER_NODE)) \
    --machine_rank=$SLURM_PROCID \
    --main_process_ip=$MAIN_ADDR \
    --main_process_port=$MAIN_PORT \
    /users/$USER/scratch/apertus/sft_train.py \
    --config /users/$USER/scratch/apertus/configs/sft_lora.yaml \
    --dataset_name medalpaca/medical_meadow_medical_flashcards"

srun bash -c "$CMD"