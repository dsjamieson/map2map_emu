#!/bin/bash
#SBATCH --job-name=run
#SBATCH --output=Logs/run.out
#SBATCH --time=0-04:00:00
#SBATCH --partition=gpu
#SBATCH --gpus=a100-sxm4-80gb:1
#SBATCH --constraint=a100-80gb
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10

#
# Runs the forward model prediction on a set of presaved input files containing the linear displacement field
#

hostname; pwd; date

in_files=                                      # Input files, numpy arrays of linear displacements with shape (3, N, N, N)

style_files=                                   # File(s) containing cosmology parameters
                                               # Must either be one file or have the same length as in_files
                                               # If only one is given, the same cosmology is assumed for all input files

output_dirs=                                   # Path to directory where output will be written, directories must be created before running

export TF_CPP_MIN_LOG_LEVEL=3

srun python ./Map2Map/m2m.py run \
    --in-pattern "$in_files" \
    --style-pattern "$style_files" \
    --out-pattern "$output_dirs" \
    --crop 128 --batches 1 --loader-workers 1  # \
#    --no_dis --no_vel # Can output either only dis, only vel, or both dis and vel

date
