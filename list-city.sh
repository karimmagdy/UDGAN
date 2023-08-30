#!/bin/sh
#SBATCH --account=g.hlwn029
#SBATCH --job-name=MonoGAN
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=200:00:00

# list-gpu-devices/list.sh (lurm)

python train.py --name MonoGAN_cityscapes_x1 --dataset_mode cityscapes --dataroot /lfs01/workdirs/hlwn029u1/DBs/SPADE2
