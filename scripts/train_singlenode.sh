#!/bin/bash

# This script is for launching training on a single node

DATE=$(date +"%m-%d")

export TRITON_CACHE_DIR='/tmp/triton_cache'

# Some pytorch settings
export OMP_NUM_THREADS=1

# Add this if using WandB
export WANDB_API_KEY='TODO'

# Initialize Conda
source ~/anaconda3/etc/profile.d/conda.sh
conda activate ttt-video

NUM_GPUS=8

# For 9 seconds and onward, you should use a checkpoint and uncomment the override flag below
CHECKPOINT_WEIGHTS_DIR="./base-model/converted"
CONFIG_FILE="./configs/train/ttt-linear/3s.toml"

EXP_NAME="${DATE}-ttt-linear-video-3s-BS8-5000steps"

torchrun --nproc_per_node=${NUM_GPUS} \
	--rdzv_backend c10d \
	--rdzv_endpoint="localhost:0" \
	--local-ranks-filter 0 \
	--role rank \
	--tee 3 \
	train.py \
	--wandb.disable \
	--job.config_file ${CONFIG_FILE} \
	--job.exp_name="${EXP_NAME}" \
	--training.global_batch_size=8 \
	--parallelism.dp_replicate=1 \
	--parallelism.dp_sharding=8 \
	--parallelism.tp_sharding=1 \
	--checkpoint.init_state_dir=${CHECKPOINT_WEIGHTS_DIR}
