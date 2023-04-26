#!/bin/bash

# Slurm sbatch options
#SBATCH -n 64
#SBATCH --time=0-04:00:00
#SBATCH --mem-per-cpu=4000
#SBATCH -o reduced_linker_sub.log-%j


# Loading the required module
module load anaconda/2022b

echo "Running python script" 
# Run the script
python reduced_linker_set.py