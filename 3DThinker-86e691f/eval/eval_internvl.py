import json
import math
import os
import re
from collections import defaultdict
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer, AutoConfig
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional, Any, Union
import datetime

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image_file, input_size=448, max_num=12):
    if not os.path.exists(image_file):
        print(f"Warning: Image file {image_file} not found")
        # Create a blank image as placeholder
        image = Image.new('RGB', (input_size, input_size), color='white')
    else:
        image = Image.open(image_file).convert('RGB')
    
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

def extract_option(text: str) -> Optional[str]:
    """
    Extract the answer from model response text using regular expressions.
    Returns the last occurrence of the letter of the answer (A, B, C, D, or E)
    based on pattern priority - tries higher priority patterns first.
    
    Args:
        text: The model response text
        
    Returns:
        The last answer letter found by the highest priority matching pattern,
        or None if not found
    """
    if not text:
        return None
    
    # First, try to match simple answer format: A., B., C., D., E. with highest priority
    simple_pattern_matches = list(re.finditer(r'([A-E])\.', text))
    if simple_pattern_matches:
        return simple_pattern_matches[-1].group(1)
    
    # Then check if <Answer> tag exists and extract content after it
    answer_section_match = re.search(r'<Answer>(.*?)(?:<|$)', text, re.DOTALL)
    if answer_section_match:
        answer_section = answer_section_match.group(1)
        # Check for specific patterns in the answer section
        for pattern in [
            r'[Mm]y answer is ([A-E])',
            r'[Mm]y answer is ([A-E])\.',
            r'[Tt]he answer is ([A-E])',
            r'(?:Answer: )?([A-E])\.',
            r'\b([A-E])\b'
        ]:
            matches = list(re.finditer(pattern, answer_section))
            if matches:
                return matches[-1].group(1)
    
    # If no matches found after <Answer> tag, proceed with regular priority patterns
    patterns = [
        r'(?:Answer: )?([A-E])\. [A-Za-z0-9 \-\(\)\'",]+(?=(?:\n|$|\.|"))',  # Full answer with description
        r'(?:Answer: )?([A-E])\. [A-Za-z0-9 \-\(\)\'"]+',  # Answer with partial description
        r'(?:^|\n)(?:Answer: )?([A-E])(?:\.|$|\s)',  # Answer at line beginning
        r'[\*\"]([A-E])[\*\"]',  # Answer in quotes or asterisks
        r'\bAnswer:?\s*([A-E])\b',  # Answer following "Answer:"
        r'[Mm]y answer is ([A-E])',  # Added pattern for "My answer is X"
        r'[Mm]y answer is ([A-E])\.',  # Added pattern for "My answer is X."
        r'answer is ([A-E])',  # Added pattern for phrases like "The answer is X"
    ]
    
    # Try each pattern in order of priority
    for pattern in patterns:
        matches = list(re.finditer(pattern, text))
        if matches:
            # Return the last match found by this pattern
            return matches[-1].group(1)
    
    # If none of the priority patterns match, try line-by-line parsing
    # First, try the more specific pattern on each line
    lines = text.split('\n')
    line_matches = []
    
    for i, line in enumerate(lines):
        # Look for full answer pattern in each line
        match = re.search(r'([A-E])\. [A-Za-z0-9 \-\(\)\'",]+', line)
        if match:
            line_matches.append((i, match.group(1)))
    
    if line_matches:
        # Return the answer from the last line that matched
        return line_matches[-1][1]
    
    # Finally, try the most general pattern on each line
    for i in reversed(range(len(lines))):  # Start from bottom
        line = lines[i]
        match = re.search(r'\b([A-E])\b', line)
        if match:
            return match.group(1)
    
    return None  # No answer found

def extract_answer(response):
    """Extract answer from model response"""
    # Find <answer>...</answer> tags
    answer_match = re.search(r'<answer>(.*?)</answer>', response, re.IGNORECASE)
    response = answer_match.group(1).strip()
    return extract_option(response)

def categorize_sample(sample_id):
    """Categorize samples based on sample ID"""
    if 'around' in sample_id.lower():
        return 'Around'
    elif 'rotation' in sample_id.lower():
        return 'Rotation'
    elif 'among' in sample_id.lower():
        return 'Among'
    else:
        return 'Other'

def load_model(model_path):
    
    try:
        # # Check GPU availability
        # if not torch.cuda.is_available():
        #     print("CUDA not available. Loading model on CPU...")
        #     device_map = None
        #     torch_dtype = torch.float32
        # else:
        #     print(f"CUDA available. Using {torch.cuda.device_count()} GPU(s)")
        # device_map = "auto"  # Let transformers automatically allocate devices
        device_map = "auto"  # Let transformers automatically allocate devices
        torch_dtype = torch.bfloat16
        
        model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            load_in_8bit=False,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            device_map=device_map,
        ).eval()
        
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, 
            trust_remote_code=True, 
            use_fast=False
        )
        
        return model, tokenizer
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None

def process_jsonl_file(jsonl_file_path, base_image_path="", model_path="", output_dir="results",model_name=""):
    """Process jsonl file and perform inference"""
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model
    print("Loading model...")
    model, tokenizer = load_model(model_path)
    if model is None or tokenizer is None:
        print("Failed to load model. Exiting.")
        return
    
    # Check model device
    device = next(model.parameters()).device
    print(f"Model loaded on device: {device}")
    
    # Read jsonl file
    samples = []
    with open(jsonl_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    
    print(f"Loaded {len(samples)} samples")
    
    # Statistics variables
    total_samples = 0
    correct_predictions = 0
    category_stats = defaultdict(lambda: {'total': 0, 'correct': 0})
    
    # Store detailed results
    detailed_results = []
    
    generation_config = dict(max_new_tokens=512, do_sample=False, temperature=0.0)
    
    # Process each sample
    for sample in tqdm(samples, desc="Processing samples"):
        try:
            sample_id = sample['id']
            question = sample['input_prompt']
            image_paths = sample['images']
            gt_answer = sample['gt_answer']
            
            # Determine category
            category = categorize_sample(sample_id)
            
            # Load images
            pixel_values_list = []
            for img_path in image_paths:
                full_img_path = os.path.join(base_image_path, img_path) if base_image_path else img_path
                try:
                    pixel_values = load_image(full_img_path, max_num=10)  # Further reduce max_num
                    pixel_values_list.append(pixel_values)
                except Exception as e:
                    print(f"Error loading image {full_img_path}: {e}")
                    continue
            
            if not pixel_values_list:
                print(f"No valid images for sample {sample_id}")
                continue
            
            # Merge all images and ensure they are on the correct device
            pixel_values = torch.cat(pixel_values_list, dim=0)
            
            # Ensure data type and device match
            if torch.cuda.is_available():
                pixel_values = pixel_values.to(device=device, dtype=torch.bfloat16)
            else:
                pixel_values = pixel_values.to(device=device, dtype=torch.float32)
            
            # Build prompt
            image_tokens = '<image>' * len(image_paths)
            prompt = f"{image_tokens}\n{question}"
            
            # Perform inference
            try:
                # Clear GPU cache
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                response, _ = model.chat(
                    tokenizer, 
                    pixel_values, 
                    prompt, 
                    generation_config,
                    history=None, 
                    return_history=True
                )
                
                # print(prompt)
                # print(response)
                # Extract answer
                predicted_answer = extract_answer(response)
                
                # Compare answers
                is_correct = predicted_answer.upper() == gt_answer.upper()
                
                # Update statistics
                total_samples += 1
                if is_correct:
                    correct_predictions += 1
                
                category_stats[category]['total'] += 1
                if is_correct:
                    category_stats[category]['correct'] += 1
                
                # Store detailed result
                result_entry = {
                    "sample_id": sample_id,
                    "category": category,
                    "question": question,
                    "gt_answer": gt_answer,
                    "predicted_answer": predicted_answer,
                    "is_correct": is_correct,
                    "response": response,
                    "image_paths": image_paths
                }
                detailed_results.append(result_entry)
                
                # Print detailed information (optional)
                # if total_samples <= 5:  # Only print detailed info for first 5 samples
                print(f"\nSample {sample_id}:")
                print(f"Question: {question[:100]}...")
                print(f"GT Answer: {gt_answer}")
                print(f"Predicted: {predicted_answer}")
                print(f"Correct: {is_correct}")
                
            except Exception as e:
                print(f"Error during inference for sample {sample_id}: {e}")
                # Print more detailed error information
                import traceback
                traceback.print_exc()
                continue
                
        except Exception as e:
            print(f"Error processing sample {sample.get('id', 'unknown')}: {e}")
            continue
    
    # Calculate and print results
    print("\n" + "="*50)
    print(f"{model_name} EVALUATION RESULTS")
    print("="*50)
    
    overall_accuracy = correct_predictions / total_samples if total_samples > 0 else 0
    print(f"Total examples: {total_samples}")
    print(f"Answer accuracy: {overall_accuracy:.2%} ({correct_predictions}/{total_samples})")
    
    print("\n=== RESULTS BY CATEGORY ===")
    category_results = {}
    for category, stats in category_stats.items():
        if stats['total'] > 0:
            accuracy = stats['correct'] / stats['total']
            category_results[category] = {
                "accuracy": accuracy,
                "correct": stats['correct'],
                "total": stats['total']
            }
            print(f"{category}: {accuracy:.2%} ({stats['correct']}/{stats['total']})")
    
    # Save results to JSON files
    summary_results = {
        "overall_accuracy": overall_accuracy,
        "total_samples": total_samples,
        "correct_predictions": correct_predictions,
        "category_results": category_results,
    }
    
    # Save summary results
    summary_file = os.path.join(output_dir, f"{model_name}_evaluation_summary.json")
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary_results, f, indent=2, ensure_ascii=False)
    
    # Save detailed results
    detailed_file = os.path.join(output_dir, f"{model_name}_detailed_results.json")
    with open(detailed_file, 'w', encoding='utf-8') as f:
        json.dump(detailed_results, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to:")
    print(f"Summary: {summary_file}")
    print(f"Details: {detailed_file}")

def main():
    # Configuration parameters
    jsonl_file_path = "/prompts/general/MindCube_tinybench_raw_qa.jsonl"  # Replace with your jsonl file path
    base_image_path = "../MindCube-main/data"  # Base path for image files, if image paths are relative
    ## TODO
    model_path = 'models/OpenGVLab/InternVL3-78B/main'
    model_name = "InternVL3-78B"
    
    output_dir = "evaluation_results"  # Directory to save JSON results
    
    # Check if file exists
    if not os.path.exists(jsonl_file_path):
        print(f"JSONL file not found: {jsonl_file_path}")
        return
    
    # Process file
    process_jsonl_file(jsonl_file_path, base_image_path, model_path, output_dir, model_name)

if __name__ == "__main__":
    main()
