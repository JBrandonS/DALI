#!/bin/bash
#SBATCH --job-name=t_full
#SBATCH --output=chains/slurmd/%x.%j.out
#SBATCH --error=chains/slurmd/%x.%j.err
#SBATCH --partition=standard-mem-s,standard-mem-m,standard-mem-l
## SBATCH --nodes=1
##SBATCH --ntasks-per-node=32
#SBATCH --ntasks=36
#SBATCH --mem=32G  # per node
#SBATCH --cpus-per-task=1
#SBATCH --array=[0,1,5,10,15,30]
##SBATCH --array=[0]

if [[ $SLURM_ARRAY_TASK_ID -eq 0 ]]; then
    tau=0.5
else 
    tau=$SLURM_ARRAY_TASK_ID
fi
DATA_FILE="BBN_small_eta10_t${tau}.pkl"
BASE_YAML="exact_small_eta10.yaml"

OUTPUT_BASE="/users/stevensonb/scratch/tmp/chains"
OUTPUT_DIR="$OUTPUT_BASE/incomplete/$SLURM_JOB_NAME/$SLURM_ARRAY_JOB_ID"
YAML_DIR="$OUTPUT_DIR/YAMLS"
LOG_DIR="$OUTPUT_DIR/LOGS"
YAML_FILE="$YAML_DIR/$SLURM_ARRAY_TASK_ID.yaml"
log_base="$LOG_DIR/$SLURM_ARRAY_TASK_ID"
SLURM_LOGS="chains/slurmd/${SLURM_JOB_NAME}.${SLURM_JOB_ID}"

module purge
module load intel/oneAPI-2021
module load python/3

source ~/.bashrc
conda activate cmb-s4

mkdir -p "$OUTPUT_DIR" "$YAML_DIR" "$LOG_DIR"

if [[ ! -f "$YAML_FILE" ]]; then
    sed "s@output: REPLACEME/REPLACEME@output: $OUTPUT_DIR/$SLURM_ARRAY_TASK_ID/$SLURM_ARRAY_TASK_ID@" "inputs/$BASE_YAML" > "$YAML_FILE"
    sed -i "s@REPLACEME_PKL@${DATA_FILE}@" "$YAML_FILE"
    sed -i "s@REPLACEME_TS@${tau}@" "$YAML_FILE"
fi

python3 add_params_bounded.py "$YAML_FILE" "bbn_t${tau}.pkl" ${tau} > "${log_base}.out" 2> "${log_base}.err"
time srun cobaya-run "$YAML_FILE" >> "${log_base}.out" 2>> "${log_base}.err"

retval=$?
if [ $retval -eq 0 ]; then
    ret_str="complete"
else
    ret_str="failed"
fi
final_dir="chains/$ret_str/$SLURM_JOB_NAME/$SLURM_ARRAY_JOB_ID"

mkdir -p $final_dir "${final_dir}/LOGS/" "${final_dir}/YAML/"
mv $OUTPUT_DIR/$SLURM_ARRAY_TASK_ID/ $final_dir
mv $YAML_FILE "${final_dir}/YAML/"

mv ${log_base}.* "${final_dir}/LOGS/"
mv $SLURM_LOGS.* "${final_dir}/LOGS/"

exit $retval