import torch
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLConfig, AutoTokenizer, AutoProcessor
from PIL import Image
import os
import logging
import deepspeed
import json

from trl import SFTTrainer, SFTConfig
from qwen_vl_utils import process_vision_info

from utils import *
from task import *
from trainer_deepspeed import CustomTrainerStage1, CustomTrainerStage2
from multimodal_projector.mmprojector import projector
import wandb
import gc
import torch

def optimize_memory():
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        
def setup_wandb(args):
    if torch.distributed.get_rank() == 0:
        wandb.init(
            project="3dthinker-training-single",
            name=f"{args.wandb_name}_latent{args.latent_size}",
            config={
                "model": args.model,
                "epochs": args.epochs,
                "task": args.task,
                "latent_size": args.latent_size,
                "stage": args.stage,
                "data_path": args.data_path,
                "save_model_path": args.save_model_path,
                "learning_rate": 1e-5,
                "per_device_train_batch_size": 1,
                "gradient_accumulation_steps": args.gradient_accumulation_steps,
            }
        )

# def setup_wandb(args):
#     if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
#         os.environ["WANDB_MODE"] = "offline"
        
#         try:
#             wandb.init(
#                 project="3dthinker-training-single",
#                 name=f"{args.wandb_name}_latent{args.latent_size}",
#                 config={
#                     "model": args.model,
#                     "epochs": args.epochs,
#                     "task": args.task,
#                     "latent_size": args.latent_size,
#                     "stage": args.stage,
#                     "data_path": args.data_path,
#                     "save_model_path": args.save_model_path,
#                     "learning_rate": 1e-5,
#                     "per_device_train_batch_size": 1,
#                     "gradient_accumulation_steps": getattr(args, 'gradient_accumulation_steps', 4),
#                 }
#             )
#             print("✅ Wandb initialized in offline mode")
#         except Exception as e:
#             print(f"❌ Wandb offline initialization failed: {e}")
#             os.environ["WANDB_DISABLED"] = "true"

def count_images_recursive(obj):
    """
    'type': 'image' number
    """
    count = 0
    
    if isinstance(obj, dict):
        if obj.get('type') == 'image':
            count += 1
        for value in obj.values():
            count += count_images_recursive(value)
    elif isinstance(obj, list):
        for item in obj:
            count += count_images_recursive(item)
    
    return count

# init
deepspeed.init_distributed()

seed_everything(seed=42)
args = get_args()

# add deepspeed config
if not hasattr(args, 'deepspeed_config'):
    args.deepspeed_config = 'deepspeed_config.json'

setup_wandb(args)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler(args.log_file, mode='a', encoding='utf-8'),
        logging.StreamHandler()
    ],
)

if torch.distributed.get_rank() == 0:
    logging.info('=='*20)
    logging.info(args)
    logging.info('=='*20)

# Load the model and processor
cache_dir = args.cache_dir
os.environ['HF_HOME'] = cache_dir

# processor = AutoProcessor.from_pretrained(args.model, cache_dir=cache_dir)
processor = AutoProcessor.from_pretrained(args.model, cache_dir=cache_dir, use_fast=False)
processor.tokenizer.add_tokens("<|latent_pad|>", special_tokens=True)
processor.tokenizer.add_tokens("<|latent_start|>", special_tokens=True)
processor.tokenizer.add_tokens("<|latent_end|>", special_tokens=True)

processor.image_processor.max_pixels = 1024 * 1024

if torch.distributed.get_rank() == 0:
    print(f"max_pixels:{processor.image_processor.max_pixels}")

if args.stage in ['stage1']: 
    model_path = args.model
    config = Qwen2_5_VLConfig.from_pretrained(model_path, cache_dir=cache_dir)
    grad_checkpointing = True
elif args.stage in ['stage2']:
    model_path = args.load_model_path
    config = Qwen2_5_VLConfig.from_pretrained(model_path)
    grad_checkpointing = False

config.compress_strategy = args.compress_strategy
config.latent_size = args.latent_size
config.stage = args.stage

if args.stage in ['stage1']:
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path, 
        config=config, 
        torch_dtype=torch.bfloat16, 
        cache_dir=cache_dir
    )
elif args.stage in ['stage2']:
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path, 
        config=config, 
        torch_dtype=torch.bfloat16
    )

model.config.use_cache = False  # unable cache -> gradient checkpointing
model.generation_config.use_cache = False if hasattr(model, 'generation_config') else None

projector_model = projector(mlp_depth=6, mm_hidden_size=2048, hidden_size=2048, latent_size = args.latent_size, fusion_input = 391, fusion_output = 1374).to(model.device)

if args.stage in ['stage1']: 
    model.resize_token_embeddings(len(processor.tokenizer))

latent_token_idx = processor.tokenizer("<|latent_pad|>", return_tensors="pt")["input_ids"][0]
latent_start_idx = processor.tokenizer("<|latent_start|>", return_tensors="pt")["input_ids"][0]
latent_end_idx = processor.tokenizer("<|latent_end|>", return_tensors="pt")["input_ids"][0]
model.config.latent_token_id = int(latent_token_idx)
model.config.latent_start_id = int(latent_start_idx)
model.config.latent_end_id = int(latent_end_idx)

for param in projector_model.parameters():
    param.requires_grad = True

model.projector_model = projector_model

for param in model.visual.parameters():
    param.requires_grad = False

if torch.distributed.get_rank() == 0:
    for name, param in model.named_parameters():
        print(f"  {name}: {param.shape}, requires_grad={param.requires_grad}")

def collate_fn_stage1(examples):
    idx_list = []
    for example in examples:
        idx_list.append(example[0]['idx'])
        del example[0]
    texts = [processor.apply_chat_template(example, tokenize=False) for example in examples]
    texts = [place_input_image(text) for text in texts]
    texts = [place_output_image(text) for text in texts]
    texts = replace_visual_spectial_tokens(texts)

    image_inputs, _ = process_vision_info(examples)
    user_examples = remove_assistant_images(examples)
    user_text = [processor.apply_chat_template(example, tokenize=False) for example in user_examples]
    user_text = replace_visual_spectial_tokens(user_text)
    user_image_inputs, _ = process_vision_info(user_examples)
    user_batch = processor(text=user_text, images=user_image_inputs, return_tensors="pt", padding=True)
    assistant_examples = remove_user_images(examples)
    assistant_text = [processor.apply_chat_template(example, tokenize=False) for example in assistant_examples]
    assistant_text = replace_visual_spectial_tokens(assistant_text)
    assistant_image_inputs, _ = process_vision_info(assistant_examples)
    assistant_batch = processor(text=assistant_text, images=assistant_image_inputs, return_tensors="pt", padding=True)
    
    batch = processor(text=texts, images=image_inputs, return_tensors="pt", padding=True)
    
    batch['pixel_values'] = user_batch['pixel_values']
    batch['image_grid_thw'] = user_batch['image_grid_thw']

    batch['pixel_values_latent'] = assistant_batch['pixel_values']
    batch['image_grid_thw_latent'] = assistant_batch['image_grid_thw']

    latent_token_idx = processor.tokenizer("<|latent_pad|>", return_tensors="pt")["input_ids"][0]
    latent_start_idx = processor.tokenizer("<|latent_start|>", return_tensors="pt")["input_ids"][0]
    latent_end_idx = processor.tokenizer("<|latent_end|>", return_tensors="pt")["input_ids"][0]

    pad_token_idx = processor.tokenizer("<|endoftext|>", return_tensors="pt")["input_ids"][0]

    new_input_ids, new_attention_mask = process_batch(batch["input_ids"], batch["attention_mask"], 
                                                      latent_start_idx, latent_end_idx, latent_token_idx, args.latent_size, pad_token_idx)

    batch["input_ids"] = new_input_ids
    batch["attention_mask"] = new_attention_mask
    batch['idx'] = idx_list

    answer_start_token_pattern = processor.tokenizer("<|im_start|>assistant", return_tensors="pt")["input_ids"][0]
    
    labels = generate_labels_after_multi_token_start(batch["input_ids"], answer_start_token_pattern, pad_token_idx, latent_token_idx)
    batch["labels"] = labels
    
    image_out_mask = mask_image_output_tokens(batch["input_ids"], latent_start_idx, latent_token_idx)
    batch["image_out_mask"] = image_out_mask
    for i, example in enumerate(examples):
        example.insert(0, {"idx": idx_list[i]})

    gc.collect()
    torch.cuda.empty_cache()
    
    return batch

def collate_fn_stage2(examples):
    texts = [processor.apply_chat_template(example, tokenize=False) for example in examples]
    
    texts = [place_input_image(text) for text in texts]
    texts = [place_output_image(text) for text in texts]
    texts = replace_visual_spectial_tokens(texts)
    
    image_inputs, _ = process_vision_info(examples)

    user_examples = remove_assistant_images(examples)
    user_text = [processor.apply_chat_template(example, tokenize=False) for example in user_examples]
    user_text = replace_visual_spectial_tokens(user_text)
    user_image_inputs, _ = process_vision_info(user_examples)
    user_batch = processor(text=user_text, images=user_image_inputs, return_tensors="pt", padding=True)

    batch = processor(text=texts, images=image_inputs, return_tensors="pt", padding=True)
    
    batch['pixel_values'] = user_batch['pixel_values']
    batch['image_grid_thw'] = user_batch['image_grid_thw']

    latent_token_idx = processor.tokenizer("<|latent_pad|>", return_tensors="pt")["input_ids"][0]
    latent_start_idx = processor.tokenizer("<|latent_start|>", return_tensors="pt")["input_ids"][0]
    latent_end_idx = processor.tokenizer("<|latent_end|>", return_tensors="pt")["input_ids"][0]

    pad_token_idx = processor.tokenizer("<|endoftext|>", return_tensors="pt")["input_ids"][0]

    new_input_ids, new_attention_mask = process_batch(batch["input_ids"], batch["attention_mask"], 
                                                      latent_start_idx, latent_end_idx, latent_token_idx, args.latent_size, pad_token_idx)

    batch["input_ids"] = new_input_ids
    batch["attention_mask"] = new_attention_mask

    answer_start_token_pattern = processor.tokenizer("<|im_start|>assistant", return_tensors="pt")["input_ids"][0]

    labels = generate_labels_after_multi_token_start(batch["input_ids"], answer_start_token_pattern, pad_token_idx, latent_token_idx)
    batch["labels"] = labels
    
    return batch

preprocess_function = task_preporcess_config[args.task]
train_dataset = load_jsonl_dataset(args.data_path)
train_dataset = [preprocess_function(sample) for sample in train_dataset]

if args.stage in ['stage1']:
    CustomTrainer = CustomTrainerStage1
    collate_fn = collate_fn_stage1
else:
    CustomTrainer = CustomTrainerStage2
    collate_fn = collate_fn_stage2

training_args = SFTConfig(
    output_dir=args.save_model_path,
    run_name=f"{args.wandb_name}_latent{args.latent_size}",  # run_name
    num_train_epochs=args.epochs,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=args.gradient_accumulation_steps,
    warmup_steps=10,
    learning_rate=1e-4,
    weight_decay=0.01,
    logging_steps=20,
    save_strategy="steps",
    save_steps=1000,
    save_total_limit=1,
    optim="adamw_torch",
    bf16=True,
    push_to_hub=False,
    remove_unused_columns=False,
    gradient_checkpointing=True,  
    dataset_text_field="",
    dataset_kwargs={"skip_prepare_dataset": True},
    report_to=["wandb"] if torch.distributed.get_rank() == 0 else [],
    logging_dir='./logs/',
    logging_strategy='steps',
    deepspeed=args.deepspeed_config,  
    dataloader_pin_memory=False,  
    max_grad_norm=1.0,
    ddp_find_unused_parameters=False,
    ddp_bucket_cap_mb=25, 
)

# Initialize the trainer
trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=collate_fn,
    # tokenizer=processor.tokenizer
    processing_class=processor
)

trainer.train()
trainer.save_model(training_args.output_dir)

if torch.distributed.get_rank() == 0:
    wandb.finish()
