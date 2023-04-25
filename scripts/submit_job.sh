#!/bin/bash

# Slurm sbatch options
#SBATCH -o reduced_linker_sub.log-%j

# Loading the required module
module load anaconda/2022b

# Run the script
python reduced_linker_set.py
