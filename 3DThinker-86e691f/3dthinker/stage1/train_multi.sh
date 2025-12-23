#!/bin/bash
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=eth0

torchrun --nproc_per_node=8 --master_port=29500 src/main_multi.py \
    --model ../../models/Qwen2.5-VL-3B-Instruct \ --epochs 10 \
    --task mindcube \
    --latent_size 12 \
    --stage stage1 \
    --data_path ../../data/data_output3d_begin_10k_resized.jsonl \
    --log_file ./log.txt \
    --gradient_accumulation_steps 8 \
    --save_model_path ../../models/3dthinker_deepspeed \
    --wandb_name supervised_mindcube_10k