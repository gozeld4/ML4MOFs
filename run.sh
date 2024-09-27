#!/bin/bash
#SBATCH --job-name=combined
#SBATCH -N 1 #node count
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=46
#SBATCH  -p xeon-p8
#SBATCH -t 96:00:00
#SBATCH -e combined.err
#SBATCH -o combined.out

python ML_models_water_uptake_all_train.py

