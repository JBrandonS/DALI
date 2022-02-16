#!/bin/bash
#SBATCH --job-name=resume
#SBATCH --output=chains/slurmd/%x.%j.out
#SBATCH --error=chains/slurmd/%x.%j.err
#SBATCH --partition=standard-mem-s
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4

ORG_JOB_NAME="lcdm_full"
ORG_JOB_ID="20750867"

module purge
module load intel/oneAPI-2021
source ~/.venv/cobaya/bin/activate

OUTPUT_DIR="chains/$ORG_JOB_NAME/$ORG_JOB_ID"
YAML_FILE="chains/$ORG_JOB_NAME/$ORG_JOB_NAME.$ORG_JOB_ID.yaml"
LOG_NUM=1
while [[ -f  "$OUTPUT_DIR/log.$LOG_NUM.out" ]]; do
    LOG_NUM=$[$LOG_NUM+1]
done

srun cobaya-run -r "$YAML_FILE" > "$OUTPUT_DIR/log.$LOG_NUM.out" 2> "$OUTPUT_DIR/log.$LOG_NUM.err" || scontrol requeue $SLURM_JOB_ID
