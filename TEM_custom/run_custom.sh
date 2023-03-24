#!/bin/bash -l
PATH='/mimer/NOBACKUP/groups/naiss2023-22-69/exjobb/results/custom/LR7e-5_BS64_WD001_ep300'
mkdir -p $PATH
#SBATCH -A naiss2023-22-69 -p alvis
#SBATCH --gpus-per-node=V100:1
#SBATCH -t 10:00:00 
#SBATCH -J CNN
#SBATCH -o /mimer/NOBACKUP/groups/naiss2023-22-69/exjobb/results/custom/logs/2022-02-28
#SBATCH -e /mimer/NOBACKUP/groups/naiss2023-22-69/exjobb/results/custom/logs/2022-02-28

#must load run_env_mnist_vgg19 to have correct environment/packages
#source /cephyr/users/karlby/Alvis/run_env_mnist_vgg19

#run script
python custom_main.py
