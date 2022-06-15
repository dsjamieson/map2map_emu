#!/bin/bash

#SBATCH --job-name=train_d2d
#SBATCH --output=%x-%j.out
#SBATCH --partition=cpu_partition
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --partition=gpu_partition
#SBATCH --gres=gpu:1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=10
#SBATCH --time=10-00:00:00

hostname; pwd; date

# Training data
train_style_files=   # Path to style file(s)
train_dis_in_files=  # Path to input ZA displacements
train_dis_tgt_files= # Path to target displacements

# Validation data
val_style_files=  # Path to style file(s)
val_dis_in_files= # Path to input ZA displacements
val_dis_tgt_dir=  # Path to target displacements

git checkout emud2d # The test.py and train.py are specific to the d2d model,
                    # so you must be on the emud2d branch

python ./map2map/m2m.py test \
    --train-style-pattern $train_style_files \
    --train-in-patterns $train_dis_in_files \
    --train-tgt-patterns $train_dis_tgt_files \
    --val-style-pattern $val_style_files \
    --val-in-patterns $val_dis_in_files \
    --val-tgt-patterns $val_dis_tgt_files \
    --in-norms cosmology.dis --tgt-norms cosmology.dis \
    --augment --aug-shift 16 \
    --crop 112 --crop-step 128 --pad 48 \
    --model d2d.StyledVNet \
    --lr 4e-4 --optimizer AdamW --optimizer-args '{"betas": [0.9, 0.999]}' \
    --reduce-lr-on-plateau --scheduler-args '{"factor": 0.1, "patience": 2, "threshold": 1e-3, "verbose": true}' \
    --batches 1 --loader-workers 9 --div-data --div-shuffle-dist 9 \
    --epochs 128
date
