#!/bin/bash -l
#SBATCH -A naiss2023-22-69 -p alvis
#SBATCH --gpus-per-node=A40:1
#SBATCH -t 30:00:00
#SBATCH -J g_vgg16n
#SBATCH -o /mimer/NOBACKUP/groups/naiss2023-22-69/exjobb/results_seed0/g_vgg16/non_augmented_D4_LR7e-05_BS64_WD001_ep300
#SBATCH -e /mimer/NOBACKUP/groups/naiss2023-22-69/exjobb/results_seed0/g_vgg16/non_augmented_D4_LR7e-05_BS64_WD001_ep300


#add your paths. 
#This Script sends main.py into Slurm.

mkdir /mimer/NOBACKUP/groups/naiss2023-22-69/exjobb/results_seed0/g_vgg16/non_augmented_D4_LR7e-05_BS64_WD001_ep300_dir3

#must load run_env_mnist_vgg19 to have correct environment/packages
#source /cephyr/users/karlby/Alvis/run_env_mnist_vgg19

#run script
python main.py
mv /mimer/NOBACKUP/groups/naiss2023-22-69/exjobb/results_seed0/g_vgg16/non_augmented_D4_LR7e-05_BS64_WD001_ep300 /mimer/NOBACKUP/groups/naiss2023-22-69/exjobb/results_seed0/g_vgg16/non_augmented_D4_LR7e-05_BS64_WD001_ep300_dir3/log
