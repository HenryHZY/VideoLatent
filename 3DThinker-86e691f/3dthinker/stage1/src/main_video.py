import torch
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLConfig, AutoProcessor
import os
import logging
from trl import SFTConfig
from qwen_vl_utils import process_vision_info # NOTE-ZY: modify qwen_vl_utils-0.0.11/src/qwen_vl_utils/vision_process.py following video-r1 config

from utils_video import seed_everything, get_args, place_input_image, place_output_image, replace_visual_spectial_tokens, process_batch, generate_labels_after_multi_token_start, mask_image_output_tokens, remove_assistant_images, load_jsonl_dataset
from task import task_preporcess_config
from trainer_single_video import CustomTrainerStage1, CustomTrainerStage2
import wandb
        
def setup_wandb(args):
    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        # Set offline mode
        os.environ["WANDB_MODE"] = "offline"
        
        try:
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
                    "gradient_accumulation_steps": getattr(args, 'gradient_accumulation_steps', 1),
                }
            )
            print("✅ Wandb initialized in offline mode")
        except Exception as e:
            print(f"❌ Wandb offline initialization failed: {e}")
            os.environ["WANDB_DISABLED"] = "true"
        
seed_everything(seed=42)
args=get_args()
setup_wandb(args)

logging.basicConfig(
    level=logging.INFO,  # Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format='%(asctime)s - %(levelname)s - %(message)s',  # Log format
    datefmt='%Y-%m-%d %H:%M:%S',  # Date format
    handlers=[
        logging.FileHandler(args.log_file, mode='a', encoding='utf-8'),
        logging.StreamHandler()
    ],
)

logging.info('=='*20)
logging.info(args)
logging.info('=='*20)

# Load the model and processor
cache_dir = args.cache_dir
os.environ['HF_HOME'] = cache_dir

processor = AutoProcessor.from_pretrained(args.model, cache_dir=cache_dir)
processor.tokenizer.add_tokens("<|latent_pad|>", special_tokens=True)
processor.tokenizer.add_tokens("<|latent_start|>", special_tokens=True)
processor.tokenizer.add_tokens("<|latent_end|>", special_tokens=True)

if args.stage in ['stage1']: 
    model_path = args.model
    config = Qwen2_5_VLConfig.from_pretrained(model_path, cache_dir=cache_dir)
    grad_checkpointing = True
elif args.stage in ['stage2']: # TODO-ZY: not used in 3DThinker
    # TODO-ZY: set grad_checkpointing = False for stage2, refer to Mirage: https://github.com/UMass-Embodied-AGI/Mirage/issues/7
    # Mirage stage2 = SFT, not used in 3DThinker
    # 3DThinker stage2 = RL
    model_path = args.load_model_path
    config = Qwen2_5_VLConfig.from_pretrained(model_path)
    grad_checkpointing = False

config.compress_strategy = args.compress_strategy
config.latent_size = args.latent_size
config.stage = args.stage

if args.stage in ['stage1']:
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, config=config, device_map="auto", torch_dtype=torch.bfloat16, cache_dir=cache_dir)
elif args.stage in ['stage2']:
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, config=config, device_map="auto", torch_dtype=torch.bfloat16)

if args.stage in ['stage1']: model.resize_token_embeddings(len(processor.tokenizer))

latent_token_idx = processor.tokenizer("<|latent_pad|>", return_tensors="pt")["input_ids"][0]
latent_start_idx = processor.tokenizer("<|latent_start|>", return_tensors="pt")["input_ids"][0]
latent_end_idx = processor.tokenizer("<|latent_end|>", return_tensors="pt")["input_ids"][0]
model.config.latent_token_id = int(latent_token_idx)
model.config.latent_start_id = int(latent_start_idx)
model.config.latent_end_id = int(latent_end_idx)

for param in model.visual.parameters():
    param.requires_grad = False
for name, param in model.named_parameters():
    print(f"  {name}: {param.shape}, requires_grad={param.requires_grad}")
print(f"learning_rate: {args.learning_rate}")

def collate_fn_stage1(examples):
    ## Replace corresponding region <output_image> -> <|latent_start|><|image_pad|><|latent_end|>
    idx_list = []
    for example in examples:
        idx_list.append(example[0]['idx'])
        del example[0]
    texts = [processor.apply_chat_template(example, tokenize=False) for example in examples]
    texts = [place_input_image(text) for text in texts]
    texts = [place_output_image(text) for text in texts]
    texts = replace_visual_spectial_tokens(texts)

    image_inputs, _ = process_vision_info(examples)
    # image_inputs = [<PIL.Image.Image image mode=RGB size=308x308 at 0x7F284412DA50>, <PIL.Image.Image image mode=RGB size=308x308 at 0x7F284412D9C0>]
    user_examples = remove_assistant_images(examples)
    user_text = [processor.apply_chat_template(example, tokenize=False) for example in user_examples]
    user_text = replace_visual_spectial_tokens(user_text)
    user_image_inputs, _ = process_vision_info(user_examples)
    ## Only user has image token
    user_batch = processor(text=user_text, images=user_image_inputs, return_tensors="pt", padding=True)
    
    batch = processor(text=texts, images=image_inputs, return_tensors="pt", padding=True)
    batch['pixel_values'] = user_batch['pixel_values']
    batch['image_grid_thw'] = user_batch['image_grid_thw']

    latent_token_idx = processor.tokenizer("<|latent_pad|>", return_tensors="pt")["input_ids"][0] # 151665
    latent_start_idx = processor.tokenizer("<|latent_start|>", return_tensors="pt")["input_ids"][0] # 151666
    latent_end_idx = processor.tokenizer("<|latent_end|>", return_tensors="pt")["input_ids"][0] # 151667

    pad_token_idx = processor.tokenizer("<|endoftext|>", return_tensors="pt")["input_ids"][0]

    # Padding for images
    new_input_ids, new_attention_mask = process_batch(batch["input_ids"], batch["attention_mask"], 
                                                      latent_start_idx, latent_end_idx, latent_token_idx, args.latent_size, pad_token_idx)
    
    # v_start = processor.tokenizer("<|vision_start|>", return_tensors="pt")["input_ids"][0] # 151652
    # img_pad = processor.tokenizer("<|image_pad|>", return_tensors="pt")["input_ids"][0] # 151655
    # v_end = processor.tokenizer("<|vision_end|>", return_tensors="pt")["input_ids"][0] # 151653
    # ## After padding, input_ids should be xxx xxx 151655 15165 15165 ... 151665 151665 151665.... corresponding to image tokens
    batch["input_ids"] = new_input_ids
    batch["attention_mask"] = new_attention_mask
    batch['idx'] = idx_list

    answer_start_token_pattern = processor.tokenizer("<|im_start|>assistant", return_tensors="pt")["input_ids"][0]
    # Find the first occurrence of start_sequence (a series of token ids) in each row (one sample). Mask this start_sequence and all tokens before it (set to -100), these positions will not be used for loss calculation.
    # Mask all pad tokens (e.g. id=0) and img tokens (e.g. id=151655) as well (set to -100). Keep remaining tokens as is for training.
    
    # Mask everything before predict to -100
    labels = generate_labels_after_multi_token_start(batch["input_ids"], answer_start_token_pattern, pad_token_idx, latent_token_idx)
    batch["labels"] = labels
    # In each sequence, find the position of the first <image_start_token>, then mark all positions equal to <image_token> after it as 1, other positions as 0
    
    # Mark 4 latent tokens as 1
    image_out_mask = mask_image_output_tokens(batch["input_ids"], latent_start_idx, latent_token_idx)
    batch["image_out_mask"] = image_out_mask
    for i, example in enumerate(examples):
        example.insert(0, {"idx": idx_list[i]})

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
    num_train_epochs=args.epochs,
    per_device_train_batch_size=args.per_device_train_batch_size,
    gradient_accumulation_steps=args.gradient_accumulation_steps,
    warmup_steps=args.warmup_steps,
    learning_rate=args.learning_rate,
    weight_decay=args.weight_decay,
    logging_steps=args.logging_steps,
    save_strategy="steps",
    save_steps=args.save_steps,
    save_total_limit=args.save_total_limit,
    optim="adamw_torch_fused",
    bf16=True,
    push_to_hub=False,
    remove_unused_columns=False,
    gradient_checkpointing=grad_checkpointing,
    dataset_text_field="",
    dataset_kwargs={"skip_prepare_dataset": True},
    report_to=["wandb"],
    logging_dir='./logs/',
    logging_strategy='steps',
)

# Initialize the trainer
trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=collate_fn,
    tokenizer=processor.tokenizer
)

trainer.train()
trainer.save_model(training_args.output_dir)
wandb.finish()

