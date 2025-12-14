#!/bin/bash
#SBATCH --job-name=verify_finetuned
#SBATCH --account=large-sc-2
#SBATCH --time=07:30:00
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=12           # adjust based on cluster
#SBATCH --mem=64G                   # total memory
#SBATCH --output=baseline_%j.log
#SBATCH --error=baseline_%j.err
#SBATCH --uenv=pytorch/v2.6.0:v1     # This loads your base PyTorch environment

# Set project directory
PROJECT_DIR="/users/sgoel/scratch/apertus-project"

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
python "$PROJECT_DIR/apertus-finetuning-recipes/query_baselines/query_baseline.py" --model_path "/users/sgoel/scratch/apertus-project/huggingface_cache/models--meta-llama--Meta-Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659"

echo "Python script finished."
echo "Job ended at $(date)"
