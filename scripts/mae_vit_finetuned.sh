#!/bin/bash
#SBATCH --job-name=MAE_FTUNE
#SBATCH --output=/PHShome/pep16/saliency-mae/outputs/MAE_FTUNE_ViT_B.txt
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=60
#SBATCH --gpus=8
#SBATCH --mail-type=end
#SBATCH --mail-user=ppak@mgh.harvard.edu
#SBATCH --mem-per-cpu=6000
#SBATCH --partition=Long
#SBATCH --qos=Long

# conda activate saliency_mae

OMP_NUM_THREADS=1
nproc_per_node=1
# mask_ratio=0.75
# model=pretrain_mae_base_patch16_224
model=vit_base_patch16_224
batch_size=64
opt=adamw
opt_beta1=0.9
opt_beta2=0.95
warmup_epochs=20
epochs=100
resume='/PHShome/pep16/saliency-mae/model_ckpts/mae_pretrain_vit_base.pth'


# Set the path to save checkpoints
home='/home'
OUTPUT_DIR=${home}'/code/saliency_mae/outputs/finetune_mae_base_patch16_224'
LOG_DIR=${home}'/code/saliency_mae/tensorboard_logs'
DATA_PATH=${home}'/data2'
SRC_PATH=${home}'/code/saliency_mae/MAE-pytorch/run_class_finetuning.py'
MODEL_PATH=${home}'/code/saliency_mae/model_ckpts/checkpoint-29.pth'
# resume=${home}'/code/saliency_mae/outputs/finetune_mae_base_patch16_224/checkpoint-best.pth'

# batch_size can be adjusted according to the graphics card
OMP_NUM_THREADS=${OMP_NUM_THREADS} python -m torch.distributed.launch --nproc_per_node=${nproc_per_node} ${SRC_PATH} \
        --data_path ${DATA_PATH} \
        --eval_data_path ${DATA_PATH}\
        --data_set iwildcam \
        --nb_classes 182\
        --model ${model} \
        --finetune ${MODEL_PATH} \
        --batch_size ${batch_size} \
        --opt ${opt} \
        --opt_betas ${opt_beta1} ${opt_beta2} \
        --warmup_epochs ${warmup_epochs} \
        --epochs ${epochs} \
        --output_dir ${OUTPUT_DIR} \
        --dist_eval > bruh.log
        # --resume ${resume}
