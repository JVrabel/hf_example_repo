#!/bin/bash
#SBATCH --job-name=hal_probe_main
#SBATCH --output=logs/hal_probe_main_%j.out
#SBATCH --error=logs/hal_probe_main_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=schmidt_sciences
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G

export TOKENIZERS_PARALLELISM=false

cd /data/jakub_vrabel/test_activ/hf_example_repo
uv run hf_example_repo