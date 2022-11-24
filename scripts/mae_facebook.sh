#!/bin/bash
#SBATCH --job-name=MAE_FB
#SBATCH --output=/PHShome/pep16/saliency-mae/outputs/FACEBOOK_MAE_IMAGENETTE_VIT_B.txt
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=60
#SBATCH --gpus=4
#SBATCH --mail-type=end
#SBATCH --mail-user=ppak@mgh.harvard.edu
#SBATCH --mem-per-cpu=6000
#SBATCH --partition=Long
#SBATCH --qos=Long

conda activate saliency_mae

nodes=1 # originally: 8
nproc_per_node=8
mask_ratio=0.75
model=mae_vit_large_patch16
batch_size=64
warmup_epochs=40
epochs=800
base_lr=1.5e-4
weight_decay=0.05

# Set the path to save checkpoints
home='/PHShome/pep16'
OUTPUT_DIR=${home}'/saliency-mae/outputs/pretrain_mae_base_patch16_224'
DATA_PATH=${home}'/saliency-mae/imagenette2/train'
SRC_PATH=${home}'/saliency-mae/mae/submitit_pretrain.py'

python ${SRC_PATH} \
    --job_dir ${OUTPUT_DIR} \
    --nodes ${nodes} \
    --use_volta32 \
    --batch_size ${batch_size} \
    --model ${model} \
    --norm_pix_loss \
    --mask_ratio ${mask_ratio} \
    --epochs ${epochs} \
    --warmup_epochs ${warmup_epochs} \
    --blr ${base_lr} --weight_decay ${weight_decay} \
    --data_path ${DATA_PATH}
