#!/bin/bash
#SBATCH -A <project name>
#SBATCH -J vessel
#SBATCH -t 00:30:00
#SBATCH --partition=zen3_0512
#SBATCH --qos=zen3_0512
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --output=vessel_%j.log

module purge
module load python/3.12.8-gcc-12.2.0-4y5tbpr
source $HOME/venvs/cmm/bin/activate
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MPLCONFIGDIR=${TMPDIR:-/tmp}

cd $SLURM_SUBMIT_DIR
python studies.py vessel
