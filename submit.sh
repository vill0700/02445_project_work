#!/bin/sh
#BSUB -q gpuv100
#BSUB -J gen
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=8GB]"
#BSUB -W 01:00
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -o Output_%J.out
#BSUB -e Output_%J.err

module purge
module load python3/3.11.9
source ~/envs/sd/bin/activate
python generate.py "$CLS" "$ADJ" "$N"