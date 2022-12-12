#!/bin/bash
#SBATCH --job-name=MAE_PTRAINV2
#SBATCH --output=/home/code/saliency_mae/outputs/MAE_PTRAINV2_ViT_B.txt
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mail-type=end
#SBATCH --mail-user=ppak@mgh.harvard.edu
#SBATCH --mem-per-cpu=6000
#SBATCH --partition=Long
#SBATCH --qos=Long

# conda activate saliency_mae

OMP_NUM_THREADS=1
nproc_per_node=1
mask_ratio=0.75
model=pretrain_mae_base_patch16_224
# batch_size=128
batch_size=64
opt=adamw
opt_beta1=0.9
opt_beta2=0.95
warmup_epochs=40
epochs=1600


# Set the path to save checkpoints
home='/home/paulpak/Downloads'
resume=${home}'/saliency_mae/model_ckpts/mae_pretrain_vit_base.pth'

# OUTPUT_DIR=${home}'/saliency_mae/outputs/pretrain_mae_base_patch16_224_salient_mask'
OUTPUT_DIR=${home}'/saliency_mae/outputs'
# DATA_PATH=${home}'/saliency_mae/data/iwildcam_v2.0/train/'
DATA_PATH=${home}'/saliency_mae/data/iwildcam_v2.0/'
SRC_PATH=${home}'/saliency_mae/MAE-pytorch/run_mae_pretrainingv2.py'
ATT_ROOT=${home}'/saliency_mae/data/iwildcam_v2.0/att_maps/'

# batch_size can be adjusted according to the graphics card
OMP_NUM_THREADS=${OMP_NUM_THREADS} python -m torch.distributed.launch --nproc_per_node=${nproc_per_node} ${SRC_PATH} \
        --data_path ${DATA_PATH} \
        --mask_ratio ${mask_ratio} \
        --model ${model} \
        --batch_size ${batch_size} \
        --opt ${opt} \
        --opt_betas ${opt_beta1} ${opt_beta2} \
        --warmup_epochs ${warmup_epochs} \
        --epochs ${epochs} \
        --output_dir ${OUTPUT_DIR} \
        --att_root ${ATT_ROOT}
        # --resume ${resume}
