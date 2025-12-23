from openai import OpenAI
import base64
import os
import json
from tqdm import tqdm
from collections import defaultdict
import re
from typing import Dict, List, Tuple, Optional, Any, Union
import datetime
import time

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

def encode_image_to_base64(image_path):
    """Read image and convert to base64 string"""
    with open(image_path, 'rb') as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')

def process_jsonl_file_gpt(jsonl_file_path, base_image_path="", output_dir="results", model_name="GPT-4o"):
    os.makedirs(output_dir, exist_ok=True)
    ## place your api key here
    client = OpenAI(
        api_key = "xxx",
        base_url = "xxx"
    )
    # Read samples
    samples = []
    with open(jsonl_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))

    print(f"Loaded {len(samples)} samples")

    total_samples = 0
    correct_predictions = 0
    category_stats = defaultdict(lambda: {'total': 0, 'correct': 0})
    detailed_results = []

    for sample in tqdm(samples, desc="Processing samples"):
        sample_id = sample['id']
        question = sample['input_prompt']
        image_paths = sample['images']
        gt_answer = sample['gt_answer']
        category = categorize_sample(sample_id)

        # Encode all images to base64
        valid_imgs = []
        valid_paths = []
        for img_path in image_paths:
            full_img_path = os.path.join(base_image_path, img_path) if base_image_path else img_path
            if os.path.exists(full_img_path):
                try:
                    img_base64 = encode_image_to_base64(full_img_path)
                    valid_imgs.append(img_base64)
                    valid_paths.append(img_path)
                except Exception as e:
                    print(f"Error loading image {full_img_path}: {e}")
            else:
                print(f"Image not found: {full_img_path}")

        if not valid_imgs:
            print(f"No valid images for sample {sample_id}")
            continue

        # Construct GPT multimodal input format
        content = [{"type": "text", "text": question}]
        for img_base64 in valid_imgs:
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{img_base64}"
                }
            })

        messages = [
            {
                "role": "user",
                "content": content
            }
        ]
        
        max_retries = 10
        retry_interval = 10
        # Inference
        try:
            # Retry mechanism
            for attempt in range(max_retries):
                try:
                    result = client.chat.completions.create(
                        model=model_name,
                        messages=messages,
                        stream=False,
                        extra_headers={
                            "M-TraceId": "2136218312678"
                        }
                    )
                    # 解析返回
                    response = result.choices[0].message.content if hasattr(result.choices[0].message, 'content') else str(result)
                    break
                except Exception as e:
                    print(f"GPT API call failed, attempt {attempt+1}: {e}")
                    time.sleep(retry_interval)
            # print(f"Error: GPT API failed {max_retries} consecutive times, skipping this sample.")
            print(response)
            predicted_answer = extract_answer(response)
            is_correct = predicted_answer and predicted_answer.upper() == gt_answer.upper()
            print(predicted_answer)
            print(gt_answer)
            print(is_correct)
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
                "image_paths": valid_paths
            }
            detailed_results.append(result_entry)

        except Exception as e:
            print(f"Error during GPT inference for sample {sample_id}: {e}")
            continue

    # Print and save results
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

    # Save results
    summary_results = {
        "overall_accuracy": overall_accuracy,
        "total_samples": total_samples,
        "correct_predictions": correct_predictions,
        "category_results": category_results,
    }
    summary_file = os.path.join(output_dir, f"{model_name}_evaluation_summary.json")
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary_results, f, indent=2, ensure_ascii=False)
    detailed_file = os.path.join(output_dir, f"{model_name}_detailed_results.json")
    with open(detailed_file, 'w', encoding='utf-8') as f:
        json.dump(detailed_results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to:\nSummary: {summary_file}\nDetails: {detailed_file}")

def main():
    jsonl_file_path = "/prompts/general/MindCube_tinybench_raw_qa.jsonl"
    base_image_path = "../MindCube-main/data"
    # model_name = "gpt-4o-2024-11-20"
    # model_name = "gpt-4.1"
    # model_name = "glm-4.5v"
    # model_name = "Doubao-Seed-1.6"
    # model_name = "o3-2025-04-16"
    model_name = "vertex.claude-sonnet-4"
    # model_name = "gemini-2.5-pro"
    output_dir = "evaluation_results"

    if not os.path.exists(jsonl_file_path):
        print(f"JSONL file not found: {jsonl_file_path}")
        return

    process_jsonl_file_gpt(jsonl_file_path, base_image_path, output_dir, model_name)

if __name__ == "__main__":
    main()