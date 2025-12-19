#!/bin/bash
#SBATCH --job-name=verify_finetuned_32
#SBATCH --account=large-sc-2
#SBATCH --time=02:30:00
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=12          # adjust based on cluster
#SBATCH --mem=64G                   # total memory
#SBATCH --output=finetuned_mixed_long_32.log
#SBATCH --error=finetuned_mixed_long_32.err
#SBATCH --uenv=pytorch/v2.6.0:v1     # This loads your base PyTorch environment

# Set project directory
PROJECT_DIR="/users/ddixit/scratch/apertus-project"

# Set environment variables for Hugging Face cache 
export HF_HOME="$PROJECT_DIR/huggingface_cache"
export HF_HUB_OFFLINE=1
export PYTHONPATH="$PROJECT_DIR/.venv/lib/python3.13/site-packages:$PYTHONPATH"

# Debugging
export NCCL_DEBUG=INFO
export TORCH_DISTRIBUTED_DEBUG=DETAIL

# Activate the virtual environment
source "$PROJECT_DIR/.venv/bin/activate"


echo "Job started on $(hostname) at $(date)"
echo "Working directory: $PWD"
echo "HF_HOME set to: $HF_HOME"
echo "Python interpreter: $(which python)"
echo "Running Python script..."

# Run the inference script
python "$PROJECT_DIR/apertus-finetuning-recipes/query_finetuned/test_finetuned_32.py"

echo "Python script finished."
echo "Job ended at $(date)"
