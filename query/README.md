# Apertus Query on Clariden

This directory contains scripts for running inference queries on the Apertus-8B-Instruct model on the Clariden cluster.

## Overview

- `query.py`: Python script that loads the Apertus-8B-Instruct model and runs inference
- `submit_query.sbatch.sbatch`: SLURM batch script for submitting the query job
- `requirements.txt`: Python dependencies needed for running queries

## Prerequisites

- Access to the Clariden cluster
- Conda/Miniconda installed in your home directory
- Sufficient storage in your scratch directory (~16GB for model cache)

## Setup Instructions

### 1. Connect to Clariden

```bash
ssh <your-username>@clariden.ethz.ch
```

### 2. Create Directory Structure

```bash
# Create the scratch directory for the project
mkdir -p ~/scratch/apertus/query
cd ~/scratch/apertus/query
```

### 3. Set Up Virtual Environment

Create a new conda environment for running queries:

```bash
# Load conda
source ~/miniconda3/etc/profile.d/conda.sh

# Create environment with Python 3.10+
conda create -n apertus_env python=3.10 -y

# Activate the environment
conda activate apertus_env
```

### 4. Install Required Packages

```bash
# Install PyTorch with CUDA support
conda install pytorch pytorch-cuda=12.1 -c pytorch -c nvidia -y

# Install other dependencies
pip install -r requirements.txt
```

Alternatively, if you prefer to use pip for everything:

```bash
pip install torch transformers>=4.55.0 accelerate>=0.20.0
```

### 5. Download the Model to HuggingFace Cache

Before running the query in offline mode, you need to download the model to your HuggingFace cache:

```bash
# Set the HuggingFace cache directory
export HF_HOME="$HOME/scratch/apertus/huggingface_cache"
mkdir -p $HF_HOME

# Run a Python script to download the model
python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = 'swiss-ai/Apertus-8B-Instruct-2509'
print(f'Downloading {model_name}...')

# Download tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
print('Tokenizer downloaded.')

# Download model
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True
)
print('Model downloaded successfully!')
print(f'Model cached in: {HF_HOME}')
"
```

**Note**: This download step requires internet access and may take 10-15 minutes depending on network speed. The model is approximately 16GB.

### 6. Copy Query Files to Scratch

Copy the query script and batch file to your scratch directory:

```bash
# From your local machine or from the repository
cp query.py ~/scratch/apertus/query/
cp submit_query.sbatch.sbatch ~/scratch/apertus/
```

### 7. Update the Batch Script Paths

Edit `submit_query.sbatch.sbatch` to ensure paths match your setup:

```bash
nano ~/scratch/apertus/submit_query.sbatch.sbatch
```

Verify these lines match your environment:
- Line 14: `export HF_HOME="/users/$USER/scratch/apertus/huggingface_cache"`
- Line 18: `source /users/$USER/miniconda3/etc/profile.d/conda.sh`
- Line 19: `conda activate apertus_env`
- Line 22: `python /users/$USER/scratch/apertus/query/query.py`

## Running the Query

### Submit the Job

```bash
cd ~/scratch/apertus
sbatch submit_query.sbatch.sbatch
```

### Monitor Job Status

```bash
# Check job queue
squeue -u $USER

# View output log (replace <job_id> with your actual job ID)
tail -f apertus_inference_<job_id>.log

# View error log
tail -f apertus_inference_<job_id>.err
```

### Check Results

Once the job completes, check the output log:

```bash
cat apertus_inference_<job_id>.log
```

The response from the model will be printed between the separator lines (`------------------------------`).

## Customizing the Query

To modify the prompt or generation parameters, edit `query.py`:

- **Line 28**: Change the `prompt` variable to your desired question
- **Line 46**: Adjust `max_new_tokens` (max: 32768, but 1024 is recommended for speed)
- **Line 48**: Adjust `temperature` (0.1-1.0, higher = more creative)
- **Line 49**: Adjust `top_p` for nucleus sampling

## Resource Allocation

The default SLURM configuration allocates:
- 1 GPU
- 12 CPUs
- 64GB RAM
- 20 minutes runtime

Adjust these in `submit_query.sbatch.sbatch` if needed for your use case.

## Troubleshooting

### Model Not Found Error
If you see `OSError: swiss-ai/Apertus-8B-Instruct-2509 does not appear to be the name of a model`, ensure:
1. The model was downloaded successfully (step 5)
2. `HF_HOME` is set correctly in the batch script
3. The cache directory contains the model files

### Out of Memory Error
If the job fails with OOM:
- Increase `--mem` in the batch script (try 96G or 128G)
- Reduce `max_new_tokens` in query.py

### GPU Not Available
Verify GPU allocation:
```bash
scontrol show job <job_id>
```

## Additional Resources

- [Clariden User Guide](https://scicomp.ethz.ch/wiki/Clariden)
- [SLURM Documentation](https://slurm.schedmd.com/documentation.html)
- [Apertus Model Card](https://huggingface.co/swiss-ai/Apertus-8B-Instruct-2509)
