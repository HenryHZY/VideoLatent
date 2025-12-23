# lr: 1e-4
# latent: 12
# best model: work-3dthinker-Qwen2.5-VL-3B-Instruct_begin_align_vggt_mlp6_lr1e-4_latent12_flash_74000_best

# NOTE-ZY: modify qwen_vl_utils-0.0.11/src/qwen_vl_utils/vision_process.py following video-r1 config

python src/main_video.py \
    --model /mnt/dolphinfs/ssd_pool/docker/user/hadoop-hldy-nlp/3A/multimodal/zhangquan/models/Qwen2.5-VL-3B-Instruct \
    --epochs 10 \
    --task video_multi_images \
    --latent_size 12 \
    --per_device_train_batch_size 1 \
    --learning_rate 1e-4 \
    --warmup_steps 10 \
    --weight_decay 0.01 \
    --logging_steps 20 \
    --save_steps 2000 \
    --save_total_limit 1 \
    --stage stage1 \
    --data_path ../../data/example.jsonl \
    --log_file ./log.txt \
    --save_model_path ../../models/3DThinker-S1-Qwen2.5-VL-3B_mlp6_lr1e-4_latent12 \
    --wandb_name 3DThinker-S1-Qwen2.5-VL-3B_mlp6_lr1e-4_latent12