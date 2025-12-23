from PIL import Image

def single_input_image_preprocess_function(sample):
    # Load images
    
    image = Image.open(sample["image_input"]).convert("RGB") 
    image_output = Image.open(sample["image_output"]).convert("RGB")

    # Format conversations
    conversations = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": sample["text_input"]},
            ],
        },
        {
            "role": "assistant",
            "content": [
                {"type": "image", "image": image_output},
                {"type": "text", "text": sample["text_output"]},
                ],
        }
    ]

    return conversations

def single_input_image_test_preprocess_function(sample):
    # Load images
    image = Image.open(sample["image_input"]).convert("RGB") 

    # Format conversations
    conversations = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": sample["text_input"]},
            ],
        },
    ]

    return conversations

def multiple_input_images_preprocess_function(sample):

    # Multiple input images
    user_content = []
    for image in sample['image_input']:
        user_content.append({"type": "image", "image": Image.open(image).convert("RGB") })
    user_content.append({"type": "text", "text": sample["text_input"]})

    image_output = Image.open(sample["image_output"]).convert("RGB")

    conversations = [
        {
            "role": "user", 
            "content": user_content
        }, 
        {
            "role": "assistant", 
            "content": [
                {"type": "image", "image": image_output}, 
                {"type": "text", "text": sample["text_output"]}
                ],
        },
    ]

    return conversations

# def vggt_multiple_input_images_preprocess_function(sample):

#     # Multiple input images
#     user_content = []
#     for image in sample['image_input']:
#         user_content.append({"type": "image", "image": Image.open(image).convert("RGB") })
#     user_content.append({"type": "text", "text": sample["text_input"]})
    
#     image_output = Image.open(sample["image_input"][0]).convert("RGB")
    
#     conversations = [
#         {
#             "idx": sample['idx']
#         },
#         {
#             "role": "user", 
#             "content": user_content
#         }, 
#         {
#             "role": "assistant", 
#             "content": [
#                 {"type": "image", "image": image_output}, 
#                 {"type": "text", "text": sample["text_output"]}
#                 ],
#         },
#     ]
    
#     return conversations


def vggt_multiple_input_images_preprocess_function(sample):
    
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

    ## mid
    # PROMPT = "\nFirst think about the reasoning process in the mind and then provide the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively. Special tokens should be used to represent 3D imagery during the reasoning process, i.e., <think> reasoning process with special tokens here </think><answer> answer here </answer>."
    ## end
    # PROMPT = "\nFirst think about the reasoning process in the mind and then provide the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively. Special tokens should be used to represent mental 3D scene at the end of your response, i.e., <think> reasoning process here </think><answer> answer here </answer>...mental 3D scene here."
    ## begin
    PROMPT = "\nFirst imagine the mental 3D scene, think about the reasoning process in the mind and then provide the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively. Special tokens should be used to represent mental 3D scene at the beginning of your response, i.e., mental 3D scene here <think> reasoning process here </think><answer> answer here </answer>."

    # Multiple input images
    user_content = []
    for image in sample['image_input']:
        original_image = Image.open(image).convert("RGB")
        # resized_image = resize_image_proportionally(original_image)
        user_content.append({"type": "image", "image": original_image})
    user_content.append({"type": "text", "text": sample["text_input"]+PROMPT})
    
    ## Placeholder - also resize output image
    # image_output = Image.open(sample["image_input"][0]).convert("RGB")
    # image_output = resize_image_proportionally(image_output)
    
    conversations = [
        {
            "idx": sample['idx']
        },
        {
            "role": "user", 
            "content": user_content
        }, 
        {
            "role": "assistant", 
            "content": [
                # {"type": "image", "image": image_output}, 
                {"type": "text", "text": sample["text_output"]}
                ],
        },
    ]
    
    return conversations


def multiple_input_images_test_preprocess_function(sample):

    # Multiple input images
    user_content = []
    for image in sample['image_input']:
        user_content.append({"type": "image", "image": Image.open(image).convert("RGB") })
    user_content.append({"type": "text", "text": sample["text_input"]})

    conversations = [
        {
            "role": "user", 
            "content": user_content
        }, 
    ]

    return conversations

task_preporcess_config = {
    'vsp-spatial-reasoning': single_input_image_preprocess_function,
    'vsp-spatial-planning': single_input_image_preprocess_function,
    'blink-jigsaw': multiple_input_images_preprocess_function,
    'sat': multiple_input_images_preprocess_function,
    'mindcube': vggt_multiple_input_images_preprocess_function,
}

task_test_preporcess_config = {
    'vsp-spatial-reasoning': single_input_image_test_preprocess_function,
    'vsp-spatial-planning': single_input_image_test_preprocess_function,
    'blink-jigsaw': multiple_input_images_test_preprocess_function,
    'sat': multiple_input_images_test_preprocess_function,
    'mindcube': vggt_multiple_input_images_preprocess_function,
}

# Define VSP valid actions and their corresponding (row, column) changes.
ACTION_MAP = {
    "LEFT":  (0, -1),
    "DOWN":  (1,  0),
    "RIGHT": (0,  1),
    "UP":    (-1, 0),
    "L":  (0, -1),
    "D":  (1,  0),
    "R": (0,  1),
    "U":  (-1, 0)
}

def parse_action_sequence(path_str):

    path_str = path_str.upper()
    action_sequence = [char for char in list(path_str) if char in ['U', 'D', 'R', 'L']]

    return action_sequence

def simulate_vsp(map_desc, path_str):
    """
    Simulate the action sequence on the provided map.
    """
    action_sequence = parse_action_sequence(path_str)

    start = None
    for r, row in enumerate(map_desc):
        for c, val in enumerate(row):
            if val == 1:
                start = (r, c)
                break
        if start is not None:
            break

    if start is None:
        raise ValueError("The map description does not contain a start position (cell value 1).")
    
    current_position = start
    for action in action_sequence:
        if action not in ACTION_MAP:
            return {
                "success": False,
                "status": "Invalid action",
                "invalid": True,
            }
            raise ValueError(f"Invalid action: {action}. Valid actions are {list(ACTION_MAP.keys())}.")
        
        dr, dc = ACTION_MAP[action]
        new_r = current_position[0] + dr
        new_c = current_position[1] + dc

        # Check boundaries; ignore the move if it goes out-of-bounds.
        if new_r < 0 or new_r >= len(map_desc) or new_c < 0 or new_c >= len(map_desc[0]):
            continue
        
        current_position = (new_r, new_c)
        
        # If the new cell is a hole (-1), immediately return failure.
        if map_desc[new_r][new_c] == -1:
            return {
                "success": False,
                "status": "Fell in hole",
                "invalid": False,
            }

    # Check if the final position is the goal (2).
    final_r, final_c = current_position
    if map_desc[final_r][final_c] == 2:
        return {
            "success": True,
            "status": "Reached goal",
            "invalid": False,
        }
    else:
        return {
            "success": False,
            "status": "Did not reach goal",
            "invalid": False,
        }