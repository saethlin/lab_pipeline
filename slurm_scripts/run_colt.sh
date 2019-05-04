#!/bin/bash
#SBATCH --job-name=colt-h113
#SBATCH --output=logs/%x_%A_%a.log
#SBATCH --ntasks=125
#SBATCH --cpus-per-task=8
#SBATCH --cores-per-socket=8
#SBATCH --distribution=cyclic:cyclic
#SBATCH --time=02:00:00
#SBATCH --mail-user=kimockb@ufl.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --qos=narayanan-b
#SBATCH --mem-per-cpu=1500mb
#SBATCH --partition=hpg2-compute
#SBATCH --constraint=haswell
#SBATCH --array=67-179

export OMPI_MCA_pml="^ucx"
#export OMPI_MCA_btl="^vader,tcp,openib"
export OMPI_MCA_btl="self,vader,openib"
export OMPI_MCA_oob_tcp_listen_mode="listen_thread"

module purge
module load intel/2018 openmpi/3.1.2 hdf5

# integer redshifts are snapshots:
#array=172,120,88,67,52,41,33,26,20%3
#A2_res33000:
#array=67,88,102,120,142,172,214,277

ID=$(printf "%03d" ${SLURM_ARRAY_TASK_ID})

WORKDIR=/ufrc/narayanan/kimockb/FIRE2/h113_HR_sn1dy300ro100ss/snapdir_${ID}/
INPUT=${WORKDIR}/converted_snapshot_${ID}.0.hdf5

srun --mpi=pmix_v2 /home/kimockb/colt-ben/bin/colt.exe ${INPUT} 1e7 --output_dir ${WORKDIR} --SB_pixels 512 --j_exp 0.25
