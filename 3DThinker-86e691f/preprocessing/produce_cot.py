import json
import os
from openai import OpenAI
import sys
import base64
import re
from tqdm import tqdm 
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from queue import Queue

# SYSTEM_PROMPT = '''
# You are an advanced AI reasoning assistant specialized in 3D spatial understanding. Please complete the intermediate reasoning process based on the given Question, Images, and Answer. Your reasoning process should incorporate spatial 3D scene that may help, meaning you will mentally construct a 3D layout of the scene to assist in your reasoning. The 3D scene is virtually representated by special token: <output_3D>.
# Wrap your reasoning process with <think></think> tags, and wrap your final result with <answer></answer> tags. <output_3D> must be inclued to represet the mental 3D scene you imagine. **Only one <output_3D> is allowed in the reasoning process. </output_3D> should not be included.**


# For example:

# <think>To determine what is to the left of the Coke can from the viewpoint presented in image 4, I need to analyze all four images and mentally imagine the 3D scene.

# 1.  **Image 1 (Front view):** The Coke can is placed on a table with a light blue tablecloth that has a black-and-white cartoon character design.  Behind the Coke can, there is a black chair leaning against a white wall.  A white cup and a green lighter are visible on the table's surface to the right of the Coke can from this perspective.

# 2.  **Image 2 (Left view):** From the left-side viewpoint, we see a window partially covered with blinds behind the Coke can.  A portion of a wooden chair and another part of the room are also visible.

# 3.  **Image 3 (Back view):** In the back view of the Coke can, display shelves and a window are visible in the background.  These objects form the central focus of this viewpoint, with various items arranged in the shelving.

# 4.  **Image 4 (Right view):** The Coke can is now viewed from the right, with a brown wall with paneling visible in the background.  A light switch or control panel on the wall and part of a doorway can also be seen in the frame.

# I imagine the 3D scene below:


# <output_3D>


# By mentally reconstructing the 3D scene above, I can determine that from image 4's perspective (right side view), the area to the left of the Coke can corresponds to what is visible behind the can in image 3. In this setup, ** the items "to the left" in the constructed 3D scene are the display shelves and window. ** 

# So, the answer is B. Display shelves and window.  </think><answer>B.  Display shelves and window</answer>
# '''


# SYSTEM_PROMPT = '''
# You are an advanced AI reasoning assistant with expertise in 3D spatial understanding. Your task is to complete the intermediate reasoning process using the provided Question, Images, and Answer. To assist your reasoning, you should mentally construct a 3D layout of the scene.

# Wrap your reasoning process with <think></think> tags and your final result with <answer></answer> tags. The mental 3D scene you imagine will be represented by the special token <output_3D>, which should be presented at the end of your response. Only one <output_3D> is allowed, and </output_3D> should not be included.

# For example:

# <think>To figure out what is on the left of the Coke can from the perspective in image 4, I'll analyze all four images and construct a mental 3D scene.

# 1. **Image 1 (Front view)**: The Coke can sits on a table covered with a light - blue tablecloth featuring a black - and - white cartoon character design. Behind the Coke can, a black chair leans against a white wall. On the table to the right of the Coke can, a white cup and a green lighter are visible.

# 2. **Image 2 (Left view)**: From the left side, there's a window partially covered by blinds behind the Coke can. A part of a wooden chair and other parts of the room are also in sight.

# 3. **Image 3 (Back view)**: In the back view of the Coke can, display shelves and a window are prominent in the background. These objects are the main focus, with various items arranged on the shelves.

# 4. **Image 4 (Right view)**: Viewed from the right, the Coke can has a brown paneled wall in the background. A light switch or control panel on the wall and part of a doorway are also visible in the frame.

# By mentally reconstructing the 3D scene, I can conclude that from image 4's right - side view, the area to the left of the Coke can matches what's behind the can in image 3. In this 3D setup, **the items "to the left" are the display shelves and window**. 

# So, the answer is B. Display shelves and window. </think><answer>B. Display shelves and window</answer>

# The 3D scene I imagined is:

# <output_3D>
# '''


SYSTEM_PROMPT = '''
You are an advanced AI reasoning assistant with expertise in 3D spatial understanding. Your task is to complete the intermediate reasoning process using the provided Question, Images, and Answer. To assist your reasoning, you should mentally construct a 3D layout of the scene.

Wrap your reasoning process with <think></think> tags and your final result with <answer></answer> tags. The mental 3D scene you imagine will be represented by the special token <output_3D>, which should be presented at the beginning of your response. Only one <output_3D> is allowed, and </output_3D> should not be included.

For example:

<output_3D>
<think> I have imagined a 3D scene shown above based on the given images, where there is a white water cup placed on a table.
The room has distinct features on each side, including walls, windows, a fridge, and possibly furniture like cabinets and chairs. The cameras capturing the images are positioned at the front, left, back, and right of the cup, providing different perspectives of the surroundings.
From the perspective of image 1 provided, we notice that there is a window with daylight coming through, and a fridge is visible next to it. This orientation suggests that the specific viewpoint from image 1 is facing these objects: the window and the fridge. In my mind, I imagine adjusting the position of the 3D scene to align it with the direction shown in image 1. Now, I need to think about what might be behind the viewpoint based on this setup. I imagine myself in the 3D scene, walking to the opposite side, that is, rotating the view horizontally 180 degrees. I saw the Window and fridge, which is what image 2 shows.
So, the answer is B. Window and fridge </think>
<answer>B. Window and fridge</answer>
'''

# prompt_old =
# '''
# <output_3D>
# <think> I have constructed a mental 3D scene from the provided images. The main landmark objects are: a large potted plant, a brass-colored floor lamp, several wooden chairs with green cushions, a round side table, and surrounding walls with tiled surfaces. The three images each show the scene from different perspectives: front, left, and right of the potted plant.
# Image 1 (front view): The potted plant is in the center, with the floor lamp directly behind it. Two chairs are beside the lamp, and a round table is partially visible to the right side.
# Image 2 (left view): The camera is facing the potted plant from the left. Looking past the plant, the lamp’s base and pole extend vertically behind the plant. The nearest chair sits a little farther to the right and back.
# Image 3 (right view): Now we see the scene from the right side of the plant. The lamp again is visible behind the plant, and chairs are closer to the viewer.
# The question asks: From the viewpoint presented in image 2 (left of the plant), what is the nearest object behind the plant? I need to determine what object is physically closest right behind the plant when standing in that location.
# Analyzing image 2, the lamp pole is almost directly lined up behind the plant, with the lamp shade slightly above the plant leaves. By mentally layering the 3D space using information from all three perspectives, it’s clear that the floor lamp is positioned immediately behind the plant, even closer than the chairs or table. I imagine myself in the 3D scene, the chair is offset and a bit farther back, and the round table is not directly behind the plant from this viewpoint. The wall is also further back.
# Therefore, the answer is: The floor lamp is the nearest object behind the potted plant when viewed from image 2 (left side perspective).</think>
# <answer>A. floor lamp</answer>
# '''

PROMPT = '''
The Question, Images, and Answer are presented as follows. Please help me start with the mental 3D scene special token <output_3D> and complete the reasoning chain.
Question: {question},
Answer: {answer}
'''

## Put your api key here       
client = OpenAI(
    api_key = "xxx",
    base_url = "xxx"
)

# Thread-safe file writing
write_lock = threading.Lock()

def encode_image(image_path):
    """Encode image to base64 string"""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    except Exception as e:
        print(f"Error encoding image {image_path}: {e}")
        return None

def get_content_after_question(text):
    """Extract content after [Question] marker"""
    pattern = r'\[Question\]\n(.*)'
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1)
    return ""

def extract_answer(xml_text):
    """Extract answer content from XML tags"""
    pattern = r'<answer>\n?(.*?)\n?</answer>'
    match = re.search(pattern, xml_text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""

def get_gpt_response(messages, max_retries=3):
    """Get response from GPT model with retry mechanism"""
    for attempt in range(max_retries):
        try:
            result = client.chat.completions.create(
                model="gpt-4.1",
                messages=messages,
                stream=False,
                extra_headers={
                    "M-TraceId": f"2136218312678_{threading.current_thread().ident}_{attempt}"
                }
            )
            return result.choices[0].message.content
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"Failed to get GPT response after {max_retries} attempts: {e}")
                return f"Error: Failed to get response - {str(e)}"
            time.sleep(1)  # Wait before retry

def write_to_file(output_file, data):
    """Thread-safe file writing"""
    with write_lock:
        with open(output_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(data, ensure_ascii=False) + '\n')
            f.flush()  # Ensure data is written immediately

def process_single_item(args):
    """Process a single data item"""
    idx, item, image_prefix, output_file = args
    
    try:
        # Extract conversations content
        conversations = item.get('conversations', [])
        
        human_value = ""
        gpt_value = ""
        
        for conv in conversations:
            if conv.get('from') == 'human':
                human_value = conv.get('value', '')
            elif conv.get('from') == 'gpt':
                gpt_value = conv.get('value', '')
        
        # Process image paths
        images = item.get('images', [])
        if image_prefix:
            if not image_prefix.endswith(os.sep) and not image_prefix.endswith('/'):
                image_prefix += os.sep
            
            images_with_prefix = []
            for img_path in images:
                if os.path.isabs(img_path):
                    full_path = img_path
                else:
                    full_path = os.path.join(image_prefix, img_path)
                images_with_prefix.append(full_path)
        else:
            images_with_prefix = images
        
        question = get_content_after_question(human_value)
        answer = extract_answer(gpt_value)
        
        if not question or not answer:
            return {
                'success': False, 
                'idx': idx, 
                'error': 'Missing question or answer',
                'question_len': 0
            }
        
        # Build content list
        content_list = [{"type": "text", "text": PROMPT.replace("{question}", question).replace("{answer}", answer)}]
        
        # Encode images
        for image_path in images_with_prefix:
            base64_image = encode_image(image_path)
            if base64_image:
                content_list.append({
                    "type": "image_url", 
                    "image_url": {"url": f"data:image/png;base64,{base64_image}"}
                })
        
        # Prepare messages
        messages_list = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": content_list,
            }
        ]
        pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
        # Get GPT response
        for i in range(10):
            gpt_completion = get_gpt_response(messages_list)
            if gpt_completion.count("<output_3D>") == 1 and re.search(pattern, gpt_completion, re.DOTALL) and ("</output_3D>" not in gpt_completion):
                break
            else:
                print("try again")
                
        # Build result data
        new_item = {
            "text_input": question,
            "image_input": images_with_prefix,
            "text_output": gpt_completion,
            "answer": answer,
            "mindcube_input": human_value,
            "mindcube_output": gpt_value
        }
        
        # Write to file immediately
        write_to_file(output_file, new_item)
        
        return {
            'success': True, 
            'idx': idx, 
            'question_len': len(question),
            'response_len': len(gpt_completion)
        }
        
    except Exception as e:
        return {
            'success': False, 
            'idx': idx, 
            'error': str(e),
            'question_len': 0
        }

def convert_json_to_jsonl(input_file, output_file, image_prefix="", max_workers=5):
    """
    Convert JSON file to JSONL format with concurrent processing
    
    Args:
        input_file: Input JSON file path
        output_file: Output JSONL file path
        image_prefix: Image path prefix (folder path)
        max_workers: Maximum number of concurrent workers
    """
    
    # Read JSON file
    print("Reading JSON file...")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    total_items = len(data)
    print(f"Found {total_items} data items")
    print(f"Using {max_workers} concurrent workers")
    
    # Clear output file
    with open(output_file, 'w', encoding='utf-8') as f:
        pass  # Create empty file
    
    # Statistics variables
    success_count = 0
    error_count = 0
    start_time = time.time()
    
    # Prepare arguments for concurrent processing
    args_list = [(idx, item, image_prefix, output_file) for idx, item in enumerate(data)]
    
    # Process with thread pool
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_idx = {executor.submit(process_single_item, args): args[0] for args in args_list}
        
        # Process completed tasks with progress bar
        with tqdm(total=total_items, desc="Processing", unit="items") as pbar:
            for future in as_completed(future_to_idx):
                try:
                    result = future.result()
                    
                    if result['success']:
                        success_count += 1
                        pbar.set_postfix({
                            'Success': success_count,
                            'Failed': error_count,
                            'Rate': f"{success_count/(success_count+error_count)*100:.1f}%" if (success_count+error_count) > 0 else "0%"
                        })
                    else:
                        error_count += 1
                        print(f"\nError processing item {result['idx']+1}: {result.get('error', 'Unknown error')}")
                        pbar.set_postfix({
                            'Success': success_count,
                            'Failed': error_count,
                            'Rate': f"{success_count/(success_count+error_count)*100:.1f}%" if (success_count+error_count) > 0 else "0%"
                        })
                    
                except Exception as e:
                    error_count += 1
                    print(f"\nUnexpected error: {e}")
                
                finally:
                    pbar.update(1)
    
    # Calculate statistics
    end_time = time.time()
    total_time = end_time - start_time
    
    # Print final statistics
    print(f"\nProcessing completed!")
    print(f"Total processed: {total_items} items")
    print(f"Successfully processed: {success_count} items")
    print(f"Failed to process: {error_count} items")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Average time per item: {total_time/total_items:.2f} seconds")
    if success_count > 0:
        print(f"Success rate: {success_count/total_items*100:.1f}%")
        print(f"Throughput: {success_count/total_time:.2f} items/second")

# Usage example
if __name__ == "__main__":
    input_file = "../MindCube-main/data/prompts/training/qwen2.5vl/MindCube_train_raw_qa_qwen_sft.json"
    output_file = "../data/data_output3d_begin_10k_resized_raw.jsonl"
    image_prefix = "../MindCube-main/data"
    
    # You can adjust max_workers based on your system and API rate limits
    max_workers = 5 # Start with 5, can increase if API allows
    
    try:
        convert_json_to_jsonl(input_file, output_file, image_prefix, max_workers)
        print(f"Conversion completed! Output file: {output_file}")
        print(f"Image path prefix: {image_prefix}")
    except FileNotFoundError:
        print(f"Error: Input file not found {input_file}")
    except json.JSONDecodeError:
        print("Error: Input file is not valid JSON format")
    except Exception as e:
        print(f"Error occurred during conversion: {e}")
