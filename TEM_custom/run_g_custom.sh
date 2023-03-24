#!/bin/bash -l
#SBATCH -A naiss2023-22-69 -p alvis
#SBATCH --gpus-per-node=A40:1
#SBATCH -t 60:00:00 
#SBATCH -J N8_300
#SBATCH -o /mimer/NOBACKUP/groups/naiss2023-22-69/exjobb/results/g_custom/N8_LR7e-05_BS64_WD001_ep300
#SBATCH -e /mimer/NOBACKUP/groups/naiss2023-22-69/exjobb/results/g_custom/N8_LR7e-05_BS64_WD001_ep300

mkdir /mimer/NOBACKUP/groups/naiss2023-22-69/exjobb/results/g_custom/N8_LR7e-05_BS64_WD001_ep300_dir2

#must load run_env_mnist_vgg19 to have correct environment/packages
#source /cephyr/users/karlby/Alvis/run_env_mnist_vgg19

#run script
python g_custom_main.py
mv /mimer/NOBACKUP/groups/naiss2023-22-69/exjobb/results/g_custom/N8_LR7e-05_BS64_WD001_ep300 /mimer/NOBACKUP/groups/naiss2023-22-69/exjobb/results/g_custom/N8_LR7e-05_BS64_WD001_ep300_dir2/log