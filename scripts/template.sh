#!/bin/bash
#SBATCH --job-name=MAE_NETTE
#SBATCH --output=/PHShome/pep16/official_version/results/MAE_IMAGENETTE_VIT_B.txt
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=80
#SBATCH --gpus=8
#SBATCH --mail-type=end
#SBATCH --mail-user=ppak@mgh.harvard.edu
#SBATCH --mem-per-cpu=6000
#SBATCH --partition=Mammoth
#SBATCH --qos=Mamomth

conda activate pretrain

# data_dir=/PHShome/pep16/Data/cholec80/Video
# annotation_dir=/PHShome/pep16/Data/annotations/cholec80_protobuf
# log_dir=/PHShome/pep16/surgical_adventure/final_code/lightning_logs/MAE_VIT_B/
# cache_dir=/PHShome/pep16/cache_mgh/img_cache
# ckpt_dir=/PHShome/pep16/surgical_adventure/final_code/lightning_logs/MAE_VIT_B/checkpoints/epoch=117-step=17388.ckpt

sampling_rate=1.0
training_ratio=1.0
temporal_length=16
num_dataloader_workers_trainer=64
batch_size=2
num_epochs=120
cuda_device=4

python3 /PHShome/pep16/surgical_adventure/final_code/src/pretrain.py --mask_model mae --backbone vit_B --track_name phase --data_dir ${data_dir} --annotation_filename ${annotation_dir} --log_dir ${log_dir} --cache_dir ${cache_dir} --sampling_rate ${sampling_rate} --training_ratio ${training_ratio} --temporal_length ${temporal_length} --num_dataloader_workers ${num_dataloader_workers_trainer} --batch_size ${batch_size} --num_epochs ${num_epochs} --cuda_device ${cuda_device} --ckpt_dir ${ckpt_dir}
