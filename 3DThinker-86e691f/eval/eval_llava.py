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
from transformers import AutoTokenizer, AutoProcessor, AutoModelForCausalLM
from qwen_vl_utils import process_vision_info
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional, Any, Union
import datetime

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
    if answer_match:
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
    """Load LLaVA-One-Vision model"""
    try:
        print(f"Loading model from {model_path}...")
        
        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            torch_dtype="auto", 
            device_map="auto", 
            trust_remote_code=True
        )
        
        # Load processor
        processor = AutoProcessor.from_pretrained(
            model_path, 
            trust_remote_code=True
        )
        
        print("Model and processor loaded successfully!")
        return model, processor
        
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None

def prepare_images_for_model(image_paths, base_image_path=""):
    """Prepare images for LLaVA-One-Vision model"""
    images = []
    valid_paths = []
    
    for img_path in image_paths:
        full_img_path = os.path.join(base_image_path, img_path) if base_image_path else img_path
        
        if not os.path.exists(full_img_path):
            print(f"Warning: Image file {full_img_path} not found")
            continue
            
        try:
            # Load image
            image = Image.open(full_img_path).convert('RGB')
            images.append(image)
            valid_paths.append(img_path)
        except Exception as e:
            print(f"Error loading image {full_img_path}: {e}")
            continue
    
    return images, valid_paths

def process_jsonl_file(jsonl_file_path, base_image_path="", model_path="", output_dir="results", model_name=""):
    """Process jsonl file and perform inference"""
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model
    print("Loading model...")
    model, processor = load_model(model_path)
    if model is None or processor is None:
        print("Failed to load model. Exiting.")
        return
    
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
    
    # Process each sample
    for sample in tqdm(samples, desc="Processing samples"):
        try:
            sample_id = sample['id']
            question = sample['input_prompt']
            image_paths = sample['images']
            gt_answer = sample['gt_answer']
            
            # Determine category
            category = categorize_sample(sample_id)
            
            # Prepare images
            images, valid_image_paths = prepare_images_for_model(image_paths, base_image_path)
            
            if not images:
                print(f"No valid images for sample {sample_id}")
                continue
            
            # Prepare messages for the model
            content = []
            
            # Add images to content
            for image in images:
                content.append({
                    "type": "image",
                    "image": image
                })
            
            # Add text question
            content.append({
                "type": "text", 
                "text": question
            })
            
            messages = [
                {
                    "role": "user",
                    "content": content
                }
            ]
            
            # Perform inference
            try:
                # Clear GPU cache
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # Preparation for inference
                text = processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                
                image_inputs, video_inputs = process_vision_info(messages)
                
                inputs = processor(
                    text=[text],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                )
                
                # Move inputs to GPU if available
                if torch.cuda.is_available():
                    inputs = inputs.to("cuda")
                
                # Generate response
                generated_ids = model.generate(**inputs, max_new_tokens=512, do_sample=False)
                generated_ids_trimmed = [
                    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                
                output_text = processor.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )
                
                response = output_text[0] if output_text else ""
                
                print(response)
                # Extract answer
                predicted_answer = extract_answer(response)
                
                # Compare answers
                is_correct = predicted_answer and predicted_answer.upper() == gt_answer.upper()
                
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
                    "image_paths": valid_image_paths
                }
                detailed_results.append(result_entry)
                
                # Print detailed information
                print(f"\nSample {sample_id}:")
                print(f"Question: {question[:100]}...")
                print(f"GT Answer: {gt_answer}")
                print(f"Predicted: {predicted_answer}")
                print(f"Correct: {is_correct}")
                
            except Exception as e:
                print(f"Error during inference for sample {sample_id}: {e}")
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
    jsonl_file_path = "/prompts/general/MindCube_tinybench_raw_qa.jsonl"
    base_image_path = "../MindCube-main/data"
    
    # Updated model configuration
    model_path = "models/lmms-lab/LLaVA-OneVision-1.5-8B-Instruct/main"
    # model_path = "models/lmms-lab/LLaVA-OneVision-1.5-4B-Instruct/main"
    model_name = "LLaVA-OneVision-1.5-8B"
    
    output_dir = "evaluation_results"
    
    # Check if file exists
    if not os.path.exists(jsonl_file_path):
        print(f"JSONL file not found: {jsonl_file_path}")
        return
    
    # Process file
    process_jsonl_file(jsonl_file_path, base_image_path, model_path, output_dir, model_name)

if __name__ == "__main__":
    main()
