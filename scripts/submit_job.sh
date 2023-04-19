#!/bin/bash

# Slurm sbatch options
#SBATCH -o submission.sh.log-%j

# Loading the required module
module load anaconda/2022b

source activate ../../../.conda/envs/esmfold

# Run the script
python reduced_linker_set.py
