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
export PYTHONPATH="$PROJECT_DIR/.venv/lib/python3.13/site-packages:$PYTHONPATH"

echo "Job started on $(hostname) at $(date)"
echo "Working directory: $PWD"
echo "HF_HOME set to: $HF_HOME"
echo "Python interpreter: $(which python)"
echo "Running Python script..."

# Run the inference script
python "$PROJECT_DIR/apertus-finetuning-recipes/query_finetuned/verify_and_test_finetuned.py"

echo "Python script finished."
echo "Job ended at $(date)"
