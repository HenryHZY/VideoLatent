## 模型名称含有end/begin/mid，+ 3dthinker
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1

python scripts/run_inference_clean.py \
  --model-type qwen2.5vl \
  --model-path ../models/3DThinker-S1-Qwen2.5-VL-3B_mlp6_lr1e-4_latent12 \
  --input-file ../MindCube-main/data/prompts/general/MindCube_tinybench_raw_qa.jsonl \
  --output-dir ../MindCube-main/data/results/3DThinker-S1-Qwen2.5-VL-3B_mlp6_lr1e-4_latent12