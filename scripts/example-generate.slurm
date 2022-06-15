#!/bin/bash

#SBATCH --job-name=generate
#SBATCH --output=%x-%j.out
#SBATCH --partition=cpu_partition
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --partition=gpu_partition
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=0-04:00:00

hostname; pwd; date

num_mesh_1d=512         # Size of grid along 1 dimension, box size is fixed
                        # by training resolution: (512 particles) / (1.e3 Mpc/h)
power_spectrum_files="" # File(s) containing power spectra
style_files=""                                # File(s) containing cosmology parameters, two columns  k [h/Mpc] P(k) [(Mpc/h)**3]
output_dir=""           # Path to directory where output will be written

# generate is the same for d2d and v2v, so it does not matter
# which branch you are on

python ./map2map generate \
    --num_mesh_1d "$num_mesh_1d" \
    --power_spectrum "$power_spectrum_files" \
    --gen-style-pattern "$style_files" \
    --out_dir "$output_dir" \
    --crop 128 --crop-step 128 --pad 48 \
    --batches 1 --loader-workers 1
#    --no_dis --no_vel # Can output either only dis, only vel, or both dis and vel
date
