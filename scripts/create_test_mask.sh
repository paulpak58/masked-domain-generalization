#!/bin/bash
#SBATCH --job-name=MASK
#SBATCH --output=/PHShome/pep16/saliency-mae/outputs/MASK.txt
#SBATCH --time=05:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --gpus=2
#SBATCH --mail-type=end
#SBATCH --mail-user=ppak@mgh.harvard.edu
#SBATCH --mem-per-cpu=6000
#SBATCH --partition=Long
#SBATCH --qos=Long


home='/PHShome/pep16'
SRC_PATH=${home}'/saliency-mae/src/saliency_mask.py'
python ${SRC_PATH}
