#!/bin/bash -l
#SBATCH -A naiss2023-22-69 -p alvis
#SBATCH --gpus-per-node=T4:1
#SBATCH -t 1:00:00 
#SBATCH -J data
#SBATCH -o ./logs
#SBATCH -e ./logs

#must load run_env_mnist_vgg19 to have correct environment/packages
#source run_env_mnist_vgg19

#run script
python count_samples.py