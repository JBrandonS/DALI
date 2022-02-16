#!/bin/bash
#SBATCH --job-name=dali_gen
#SBATCH --output=chains/slurmd/%x.%j.out
#SBATCH --error=chains/slurmd/%x.%j.err
#SBATCH --partition=standard-mem-s
#SBATCH --nodes=2
#SBATCH --mem=64G  # per node
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4

OUTPUT_DIR="chains/$SLURM_JOB_NAME/$SLURM_JOB_ID"
BASE_YAML="$SLURM_JOB_NAME.yaml"
YAML_FILE="$OUTPUT_DIR/../$SLURM_JOB_NAME.$SLURM_JOB_ID.yaml"
mkdir -p "$OUTPUT_DIR"

module purge
module load intel/oneAPI-2021
source ~/.venv/cobaya/bin/activate

if [[ ! -f "$YAML_FILE" ]]; then
    sed "s@output: REPLACEME/REPLACEME@output: $OUTPUT_DIR/$SLURM_JOB_NAME@" "inputs/$BASE_YAML" > "$YAML_FILE"
fi

LOG_NUM=0
while [[ -f  "$OUTPUT_DIR/log.$LOG_NUM.out" ]]; do
    LOG_NUM=$((LOG_NUM+1))
done

srun cobaya-run -r --debug "$YAML_FILE" > "$OUTPUT_DIR/log.$LOG_NUM.out" 2> "$OUTPUT_DIR/log.$LOG_NUM.err" || scontrol requeue $SLURM_JOB_ID
