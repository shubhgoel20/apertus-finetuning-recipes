# Apertus Fine-Tuning on Clariden (GH200 / uenv)

This guide documents the specific steps required to run the Apertus fine-tuning recipes on the CSCS Clariden cluster using the uenv software stack (NVIDIA GH200).

## 1. Initial Setup (One-Time)

### A. Start an Interactive Session

Do not run installation steps on the login node. Always acquire a compute node first.

```bash
srun --nodes=1 --account=large-sc-2 --time=01:00:00 --pty bash
```

### B. Pull the PyTorch Image

Download the specific image optimized for GH200 (version v2.6.0:v1 as of April 2025). *Only needed once.*

```bash
uenv image pull pytorch/v2.6.0:v1
```

### C. Start the Environment

Start the container with the default view.

```bash
uenv start pytorch/v2.6.0:v1 --view=default
```

### D. Create Virtual Environment

Create a venv that inherits system packages (PyTorch, drivers) but allows local installations.

```bash
// Create venv (only once)
python -m venv .venv --system-site-packages

// Activate venv (every session)
source .venv/bin/activate
```

## 2. Dependency Installation

Run the following commands in order:

```bash
# 1. Install modified requirements
pip install -r requirements.txt

# 2. Force install transformers to local venv
pip install --ignore-installed transformers
pip install deprecated
```

## 3. Configuration Fixes

The default config uses an experimental attention kernel that fails on this cluster. Switch to standard Flash Attention 2.

- Open your config file (e.g., `configs/sft_lora.yaml` or `configs/sft_full.yaml`).
- Find the `attn_implementation` line.
- Change it to:

```yaml
attn_implementation: "flash_attention_2"
```

## 4. Running Training

Python may prioritize the system `transformers` over your local installation. Export `PYTHONPATH` before every run.

### Option A: Interactive Run

```bash
source .venv/bin/activate
export PYTHONPATH=$PWD/.venv/lib/python3.13/site-packages:$PYTHONPATH
python sft_train.py --config configs/sft_lora.yaml
```

Submit the job:

```bash
sbatch submit_clariden.slurm
```

