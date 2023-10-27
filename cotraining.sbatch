#!/bin/bash
# Tiffany Le
#
# When you use this batch file:
# Change the email address to yours! (I don’t want email about your experiments)
# Change the chdir line to match the location of where your code is located
#
#SBATCH --partition=ai2es
#SBATCH --cpus-per-task=12
# The number of cores you get
#SBATCH --ntasks=1
# memory in MB
#SBATCH --mem=8192
#SBATCH --gres=gpu:1
# The %04a is translated into a 4-digit number that encodes the SLURM_ARRAY_TASK_ID
#SBATCH --output=/ourdisk/hpc/ai2es/jroth/tiffanyle_home/out.txt
#SBATCH --error=/ourdisk/hpc/ai2es/jroth/tiffanyle_home/err.txt
#SBATCH --time=00:30:00
#SBATCH --job-name=cotraining
#SBATCH --mail-user=jay.c.rothenberger@ou.edu
#SBATCH --mail-type=ALL
#SBATCH --chdir=/ourdisk/hpc/ai2es/jroth/tiffanyle_home/co-training
#
#################################################
# Do not change this line unless you have your own python/tensorflow/keras set up

. /home/fagg/tf_setup.sh
conda activate /home/jroth/.conda/envs/torch
pwd
# Change this line to start an instance of your experiment
python cotraining.py --epochs 10 --batch_size 64 -lr 1e-3 --momentum 0.9
