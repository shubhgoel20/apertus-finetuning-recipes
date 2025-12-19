#!/bin/bash

for ((id = 12; id < 36; id++)); do
    sbatch "/users/ddixit/scratch/apertus-project/apertus-finetuning-recipes/query_finetuned/submit_test_finetuned_${id}.sbatch"
done