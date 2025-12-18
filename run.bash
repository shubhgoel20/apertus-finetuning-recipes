#!/bin/bash

# uenv start pytorch/v2.6.0:v1 --view=default
# source .venv/bin/activate

# set +e
# python make_yaml_sbatch.py
ret=12

for ((id = 0; id < ret; id++)); do
    mkdir -p "/users/ddixit/scratch/apertus-project/output_${id}"
    sbatch "/users/ddixit/scratch/apertus-project/apertus-finetuning-recipes/apertus_finetune_${id}.sbatch"
done
