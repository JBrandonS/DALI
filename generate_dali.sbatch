#!/bin/bash
#SBATCH --job-name=dali_gen
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err
#SBATCH --partition=standard-mem-s
#SBATCH --nodes=2
#SBATCH --mem=64G  # per node
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4

cd packages/code/lenspower/py/
srun python DALIGenerateDataClass.py
