#!/bin/bash
#SBATCH --job-name=apertus_inference
#SBATCH --account=large-sc-2
#SBATCH --time=00:20:00             # adjust based on expected runtime
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=12           # adjust based on cluster
#SBATCH --mem=64G                   # total memory
#SBATCH --output=apertus_inference_%j.log
#SBATCH --error=apertus_inference_%j.err

export HF_HOME="/users/$USER/scratch/apertus/huggingface_cache"
# export HF_HUB_OFFLINE=1

# Activate your virtual environment
source /users/$USER/miniconda3/etc/profile.d/conda.sh
conda activate apertus_env

# Run the inference script
python /users/$USER/scratch/apertus/query.py
