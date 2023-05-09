#!/bin/bash

# Slurm sbatch options
#SBATCH -o active_learning.log-%j
#SBATCH --time=0-04:00:00
#SBATCH --gres=gpu:volta:2
#SBATCH -c 40

# set cuda path
export CUDA_HOME=/usr/local/pkg/cuda/cuda-11.3

# Run the script (replace /home/gridsan/jsass with your home directory)
/home/gridsan/jsass/.conda/envs/project_env/bin/python active_learning_loop.py