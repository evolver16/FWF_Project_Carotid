#!/bin/bash
#SBATCH -A <project name>
#SBATCH -J cmm
#SBATCH -t 00:15:00
#SBATCH --partition=zen3_0512
#SBATCH --qos=zen3_0512
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --output=%x_%j.log

module purge
module load python/3.12.8-gcc-12.2.0-4y5tbpr
source $HOME/venvs/cmm/bin/activate
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MPLCONFIGDIR=${TMPDIR:-/tmp}

cd $SLURM_SUBMIT_DIR
python studies.py "$@"
