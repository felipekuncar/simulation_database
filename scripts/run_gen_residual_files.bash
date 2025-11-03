#!/bin/bash
#SBATCH --job-name=python_test
#SBATCH --output=slurm_genReal_%j.out
#SBATCH --error=slurm_genReal_%j.err
#SBATCH --time=01:00:00
#SBATCH --exclude=n12
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mail-user=felipe.kuncar@canterbury.ac.nz
#SBATCH --mail-type=ALL
source $HOME/.bashrc
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
echo $LD_LIBRARY_PATH
mamba activate emp_calc_env
echo ===== ENVIRONMENT =====
echo PYTHON  $(which python)
echo VERSION $(python --version)
echo PWD     $(pwd)
echo SCRATCH $SCRATCH
echo =======================

# Pass number of CPUs from SLURM to Python
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

python gen_residual_files.py
