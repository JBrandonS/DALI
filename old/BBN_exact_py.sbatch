#!/bin/bash
#SBATCH --job-name=dbe_array_profiled
#SBATCH --output=chains/slurmd/%x.%j.out
#SBATCH --error=chains/slurmd/%x.%j.err
#SBATCH --partition=standard-mem-m
##SBATCH --nodes=1
##SBATCH --ntasks-per-node=1
#SBATCH --mem=64G  # per node
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --array=[0,1,5,10,30]
#SBATCH --mail-user=stevensonb@smu.edu
#SBATCH --mail-type=BEGIN,END,FAIL

OUTPUT_DIR="chains/$SLURM_JOB_NAME"
mkdir -p "$OUTPUT_DIR"

module purge
module load intel/oneAPI-2021
module load python/3

source ~/.bashrc
conda activate cmb-s4

if [[ $SLURM_ARRAY_TASK_ID -eq 0 ]]; then
    taskid=0.5
else 
    taskid=$SLURM_ARRAY_TASK_ID
fi
log_base="$OUTPUT_DIR/log-$taskid"

LOG_NUM=0
while [[ -f  "${log_base}.$LOG_NUM.out" ]]; do
    LOG_NUM=$((LOG_NUM+1))
done

srun python3 -m cProfile BBN_exact.py $taskid > "${log_base}.$LOG_NUM.out" 2> "${log_base}.$LOG_NUM.err"
