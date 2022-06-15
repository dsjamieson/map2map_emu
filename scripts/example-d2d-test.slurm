#!/bin/bash

#SBATCH --job-name=test_d2d
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
dis_in_files= # Path to input ZA 
dis_tgt_dir=  # Path to target displacements

git checkout emud2d # The test.py and train.py are specific to the d2d model,
                    # so you must be on the emud2d branch

python ./map2map/m2m.py test \
    --test-style-pattern $style_files \
    --test-in-patterns $dis_in_files \
    --test-tgt-patterns $dis_tgt_files \
    --in-norms cosmology.dis --tgt-norms cosmology.dis \
    --crop 128 --crop-step 128 --pad 48 \
    --model d2d.StyledVNet --callback-at . \
    --batches 1 --loader-workers 7 \
    --load-state "./map2map/weights/d2d_weights.pt"
date
