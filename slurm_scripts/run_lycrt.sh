#!/bin/bash
#SBATCH --job-name=lycrt-h113
#SBATCH --output=logs/%x_%A_%a.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --distribution=cyclic:cyclic
#SBATCH --time=01:00:00
#SBATCH --mail-user=kimockb@ufl.edu
#SBATCH --mail-type=all
#SBATCH --qos=narayanan-b
#SBATCH --mem-per-cpu=3800mb
#SBATCH --partition=hpg2-compute
#SBATCH --constraint=haswell
#SBATCH --array=20-179

set -e

module purge
module load intel/2018 openmpi/3.1.2

ID=$(printf "%03d" $SLURM_ARRAY_TASK_ID)
SIMNAME=h113_HR_sn1dy300ro100ss
#array=67,88,102,120,142,172,214,277 for A2

# Have to cd because lycrt just drops its output in the current working directory
mkdir -p ${SIMNAME}/snapdir_${ID}
cd ${SIMNAME}/snapdir_${ID}

pattern="../Groups/caesar_0${ID}*"
files=( $pattern )
CAESAR="${files[0]}"

python ~/lab_pipeline/setup_for_lycrt.py /ufrc/narayanan/kimockb/FIRE2/${SIMNAME}/snapshot_${ID}.0.hdf5 --caesar ${CAESAR}
~/octreert/lycrt/src/lycrt paramfile
python ~/lab_pipeline/lycrt_to_colt.py /ufrc/narayanan/kimockb/FIRE2/${SIMNAME}/snapshot_${ID}.0.hdf5 octreefile

