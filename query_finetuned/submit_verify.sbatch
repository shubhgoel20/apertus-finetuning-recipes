#!/bin/bash
#SBATCH --job-name=verify_finetuned
#SBATCH --account=large-sc-2
#SBATCH --time=00:30:00
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=18
#SBATCH --output=verify_finetuned_%j.log
#SBATCH --error=verify_finetuned_%j.err
#SBATCH --uenv=pytorch/v2.6.0:v1
#SBATCH --view=default

# Set project directory
PROJECT_DIR="/users/mmeciani/scratch/apertus-project"

# Set cache locations
export HF_HOME="$PROJECT_DIR/huggingface_cache"
export TOKENIZERS_PARALLELISM=false

# Activate virtual environment
source $PROJECT_DIR/.venv/bin/activate

# Run verification script
python verify_and_test_finetuned.py
