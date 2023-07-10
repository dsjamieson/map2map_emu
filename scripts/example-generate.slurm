#!/bin/bash
#SBATCH --job-name=generate
#SBATCH --output=Logs/gen.out
#SBATCH --time=0-04:00:00
#SBATCH --partition=gpu
#SBATCH --gpus=a100-sxm4-80gb:1
#SBATCH --constraint=a100-80gb
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10

#
# Generates random gaussian dislpacements fields from tabulated power spectra, then
# runs the forward model prediction on these
#

hostname; pwd; date

num_mesh_1d=                                   # Size of grid along 1 dimension, box size is fixed
                                               # by training resolution: (512 particles) / (1.e3 Mpc/h)
                                               # so the output box length will be 1.e3 Mpc / h * num_mesh_1d / 512

power_spectrum_files=                          # File(s) containing power spectra, two columns  k [h/Mpc] P(k) [(Mpc/h)**3]
                                               # If one is given, all generated data is drawn from the same power specturm

style_files=                                   # File(s) containing cosmology parameters
                                               # Must either be one file or have the same length as power_spectrum_files

output_dirs=                                   # Path to directory where output will be written

export TF_CPP_MIN_LOG_LEVEL=3

srun python ./Map2Map/m2m.py generate \
    --num-mesh-1d "$num_mesh_1d" \
    --pk-pattern "$power_spectrum_files" \
    --style-pattern "$style_files" \
    --out-pattern "$output_dirs" \
    --crop 128 --batches 1 --loader-workers 1 # \
#    --no_dis --no_vel # Can output either only dis, only vel, or both dis and vel

date
