# Project Setup and Usage

This guide outlines the steps to set up the environment, install dependencies, prepare datasets, and run experiments (Zero-shot, LoRA, and Fine-tuning).

## 1. Environment Setup

Initialize the PyTorch environment and create a virtual environment with system site packages enabled.

```bash
# Start the specific uenv environment
uenv start pytorch/v2.6.0:v1 --view=default

# Create virtual environment
python -m venv .venv --system-site-packages

# Activate the environment
source .venv/bin/activate
```

## 2. Package installation

Install the required packages and xIELU

```bash
# Install XIELU without build isolation
pip install --no-build-isolation git+[https://github.com/nickjbrowning/XIELU](https://github.com/nickjbrowning/XIELU)

# Install remaining requirements
pip install -r requirements.txt
```

## 3. Dataset setup

Install the dataset in the HF cache and format it with ```format_data.py```

```bash
# Set cache directory
export HF_HOME="/users/ddixit/scratch/apertus-project/huggingface_cache"

# Download MMLU
huggingface-cli download cais/mmlu \
    --repo-type dataset \
    --cache-dir $HF_HOME

# Download MedMCQA
huggingface-cli download openlifescienceai/medmcqa \
    --repo-type dataset \
    --cache-dir $HF_HOME

# Format data
python prepare_data.py

#
```
## 4. Running experiment

Submit the sbatch files in the /query /query_finetuned and main folder
