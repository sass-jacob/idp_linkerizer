#!/bin/bash

# Slurm sbatch options
#SBATCH -o active_learning.log-%j
#SBATCH --time=0-08:00:00
#SBATCH --gres=gpu:volta:1
#SBATCH -c 20
export CUDA_HOME=/usr/local/pkg/cuda/cuda-11.3

# Run the script
/home/gridsan/jsass/.conda/envs/project_env/bin/python active_learning_loop.py
