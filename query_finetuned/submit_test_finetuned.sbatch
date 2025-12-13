#!/bin/bash
#SBATCH --job-name=apertus_eval_ft
#SBATCH --account=large-sc-2
#SBATCH --time=00:30:00             # 30 minutes should be enough for 100 examples
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4           # Only need 1 GPU for inference
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --output=apertus_eval_finetuned_%j.log
#SBATCH --error=apertus_eval_finetuned_%j.err
#SBATCH --uenv=pytorch/v2.6.0:v1

# Define project directory
PROJECT_DIR="/users/mmeciani/scratch/apertus-project"

# Set environment variables for Hugging Face cache 
export HF_HOME="$PROJECT_DIR/huggingface_cache"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONPATH="$PROJECT_DIR/.venv/lib/python3.13/site-packages:$PYTHONPATH"

# Activate the virtual environment
source "$PROJECT_DIR/.venv/bin/activate"

echo "Job started on $(hostname) at $(date)"
echo "Working directory: $PWD"
echo "HF_HOME set to: $HF_HOME"
echo "Python interpreter: $(which python)"
echo "Running fine-tuned model evaluation..."

# Run the evaluation script
python "$PROJECT_DIR/apertus-finetuning-recipes/query_finetuned/test_finetuned.py"

echo "Evaluation finished."
echo "Job ended at $(date)"
