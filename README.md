# Apertus Project Setup Guide

## Directory Structure to recreate inside the cluster
```
apertus-project/
├─ .venv/
├─ apertus-finetuning-recipes/
│  ├─ configs/
│  └─ query/
│  └─ query_finetuned/
├─ output/
├─ huggingface_cache/
└─ triton-cache/
```

## Steps to Set-Up

### 1. Clone and Import Finetuning Recipes
- Place `apertus-finetuning-recipes` inside `apertus-project`.
- Replace all paths placeholders with real paths: submit_lora.sbatch, sft_train.py, .yaml file, ...

### 2. Configure Output Directory
- Create `output/` in `apertus-project`.
- Update YAML config paths inside `apertus-finetuning-recipes/configs` to point to this directory.

### 3. Environment Setup on Login Node
- Pull PyTorch uenv image and start session.

```bash
uenv image pull pytorch/v2.6.0:v1
uenv start pytorch/v2.6.0:v1 --view=default
```

- Create and activate a virtual environment:

```python
// Create venv (only once)
python -m venv .venv --system-site-packages

// Activate venv (every session)
source .venv/bin/activate
```

### 4. Install Required Packages
```bash
pip install --upgrade pip
pip install --no-build-isolation git+https://github.com/nickjbrowning/XIELU
pip install -r requirements.txt
```

### 5. Configure Hugging Face Cache
```bash
export HF_HOME="/users/mmeciani/scratch/apertus-project/huggingface_cache"
```

### 6. Download Apertus Model (within .venv)
```bash
huggingface-cli download swiss-ai/Apertus-8B-Instruct-2509 --cache-dir $HF_HOME --exclude "*.msgpack" "*.h5" "*.ot"
```

### 7. Download Dataset
```bash
huggingface-cli download HuggingFaceH4/Multilingual-Thinking \
    --repo-type dataset \
    --cache-dir $HF_HOME

huggingface-cli download medalpaca/medical_meadow_medical_flashcards \
    --repo-type dataset \
    --cache-dir $HF_HOME
```

- Check results and confirm the model path in `query.py` points to the correct location.

## Querying the Model
- Submit job using the sbatch script located in `apertus-finetuning-recipes/query`.
- Ensure the model path in `query.py` points to  
  `HF_HOME/.../config.json` inside the downloaded Apertus folder.

## Performing LoRA Finetuning
- Double check all paths and evnironment variables/packages
- Submit job using the sbatch script located in `apertus-finetuning-recipes`.
- Check results in the `output` folder

## Querying the Fine-Tuned Model
- Head to `query_finetuned`
- Here a series of infos about the Fine-Tuned model will be displayed as well as a series of test prompts
- Submit the test using `submit_verify.sbatch`


  

