import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
import os
import json
import logging
from tqdm import tqdm

from qwen_vl_utils import process_vision_info
from mathruler.grader import extract_boxed_content
from PIL import Image
from src.utils import *
from src.task import *

def resize_image_proportionally(image, max_size=640):
    """
    Proportionally resize image, keeping the longest side no more than max_size
    """
    width, height = image.size
    
    # If image dimensions are all less than or equal to max_size, no need to resize
    if width <= max_size and height <= max_size:
        return image
    
    # Calculate scale ratio
    if width > height:
        # Width is the longest side
        scale_ratio = max_size / width
    else:
        # Height is the longest side
        scale_ratio = max_size / height
    
    # Calculate new dimensions
    new_width = int(width * scale_ratio)
    new_height = int(height * scale_ratio)
    
    # Perform resize
    return image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
seed_everything(seed=42)
args=get_args()

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

file_path = "../data/data_output3d_begin_10k_resized.jsonl"
load_model_path = "../models/3DThinker-S1-Qwen2.5-VL-3B_mlp6_lr1e-4_latent12"

# processor = AutoProcessor.from_pretrained(load_model_path, trust_remote_code=True, cache_dir=cache_dir)
processor = AutoProcessor.from_pretrained(load_model_path, trust_remote_code=True)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(load_model_path, device_map="auto", torch_dtype=torch.bfloat16)

processor.tokenizer.add_tokens("<|latent_pad|>", special_tokens=True)
processor.tokenizer.add_tokens("<|latent_start|>", special_tokens=True)
processor.tokenizer.add_tokens("<|latent_end|>", special_tokens=True)

model.eval()

## mid
# PROMPT = "\nFirst think about the reasoning process in the mind and then provide the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively. Special tokens should be used to represent 3D imagery during the reasoning process, i.e., <think> reasoning process with special tokens here </think><answer> answer here </answer>."
## end
# PROMPT = "\nFirst think about the reasoning process in the mind and then provide the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively. Special tokens should be used to represent mental 3D scene at the end of your response, i.e., <think> reasoning process here </think><answer> answer here </answer>...mental 3D scene here."
## begin
PROMPT = "\nFirst imagine the mental 3D scene, think about the reasoning process in the mind and then provide the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively. Special tokens should be used to represent mental 3D scene at the beginning of your response, i.e., mental 3D scene here <think> reasoning process here </think><answer> answer here </answer>."
# PROMPT = "\nFirst, visualize a mental 3D scene in your mind. Represent this mental 3D scene using special tokens, where <|latent_start|> marks the beginning, <|latent_end|> marks the end, and <|latent_pad|> tokens are placed between <|latent_start|> and <|latent_end|>. For example, <|latent_start|><|latent_pad|><|latent_pad|>...<|latent_end|>. Then, offer the user the reasoning process and the answer. Enclose the reasoning process within <think> </think> tags and the answer within <answer> </answer> tags. Your response should start with the representation of the mental 3D scene using the special tokens, i.e., <|latent_start|><|latent_pad|><|latent_pad|>...<|latent_end|><think> reasoning process here </think><answer> answer here </answer>."
with open(file_path, 'r', encoding='utf-8') as file:
        for line_number, line in enumerate(file, 1):
            # 去除行尾的换行符并解析JSON
            data = json.loads(line.strip())
            prompt = data["text_input"]
            image_list = data["image_input"]
            content_list = []
            for img_input in image_list:
                image = Image.open(img_input).convert("RGB")
                # resized_image = resize_image_proportionally(image)
                content_list.append({'type': 'image', 'image': image})
                
            # prompt = "Based on these four images (image 1, 2, 3, and 4) showing the black sneaker from different viewpoints (front, left, back, and right), with each camera aligned with room walls and partially capturing the surroundings: From the viewpoint presented in image 1, what is to the left of the black sneaker? A. Washing machine B. Wall and window C. Curtain D. White wood rack"    
            
            question = prompt+PROMPT
            image_prompt = "<image>"*len(image_list)
            content_list.append({'type': 'text', 'text': image_prompt + question})
            print(f"prompt:{question}")
            conversations = [{'role': 'user', 'content': content_list}]

            texts = [processor.apply_chat_template(conversations, tokenize=False)]
            texts = [place_input_image(text, sep_token=None) for text in texts]
            image_inputs, _ = process_vision_info(conversations)

            inputs = processor(text=[t+'<|im_start|>assistant' for t in texts], images=image_inputs, return_tensors="pt", padding=True)
            inputs = inputs.to(model.device)

            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=2048,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    tokenizer=processor.tokenizer,
                )
            print(output_ids)
            decoded_output = processor.tokenizer.decode(output_ids[0], skip_special_tokens=False)
            answer = decoded_output.split('<|im_start|>assistant')[-1]
            print(answer)
            print("======")