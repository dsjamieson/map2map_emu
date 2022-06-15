#!/bin/bash

#SBATCH --job-name=test_v2v
#SBATCH --output=%x-%j.out
#SBATCH --partition=cpu_partition
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --partition=gpu_partition
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=0-01:00:00

hostname; pwd; date

style_files=  # Path to style file(s)
vel_in_files= # Path to input ZA 
vel_tgt_dir=  # Path to target displacements

git checkout emuv2v # The test.py and train.py are specific to the v2v model,
                    # so you must be on the emuv2v branch

python ./map2map/m2m.py test \
    --test-style-pattern $style_files \
    --test-in-patterns $vel_in_files \
    --test-tgt-patterns $vel_tgt_files \
    --in-norms cosmology.dis --tgt-norms cosmology.dis \
    --crop 128 --crop-step 128 --pad 48 \
    --model v2v.StyledVNet --callback-at . \
    --batches 1 --loader-workers 7 \
    --load-state "./map2map/weights/v2v_weights.pt"
date
