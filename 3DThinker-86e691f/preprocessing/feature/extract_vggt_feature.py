import os
import torch
import numpy as np
from tqdm import tqdm
import json

# -------------------------------
# Import new model and related utility functions
# -------------------------------
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map
from PIL import Image
import os

from PIL import Image
import os

def resize_image(image_path, new_size=(480, 640)):
    # 打开图片
    with Image.open(image_path) as img:
        # 进行缩放
        resized_img = img.resize(new_size, Image.LANCZOS)

        # 构建新的文件路径，将文件夹名称替换为目标文件夹名称
        new_image_path = image_path.replace("other_all_image", "other_all_image_resize")
        
        # 创建新的文件路径对应的文件夹（如果不存在）
        new_folder = os.path.dirname(new_image_path)
        if not os.path.exists(new_folder):
            os.makedirs(new_folder)

        # 保存图片到新文件夹
        resized_img.save(new_image_path)

        return new_image_path

def process_images(image_paths):
    # 新图片路径列表
    new_image_paths = []

    # 对每一张图片进行处理
    for image_path in image_paths:
        new_image_path = resize_image(image_path)
        new_image_paths.append(new_image_path)

    return new_image_paths

# -------------------------------
# Device and data type settings
# -------------------------------
# If you have only one GPU, change "cuda:1" to "cuda:0"
device = "cuda:0" if torch.cuda.is_available() else "cpu"
if torch.cuda.is_available():
    # Get the current GPU's compute capability: Ampere GPU (Compute Capability 8.0+) supports bfloat16
    dev_capability = torch.cuda.get_device_capability(torch.cuda.current_device())
    dtype = torch.bfloat16 if dev_capability[0] >= 8 else torch.float16
else:
    dtype = torch.float32

# -------------------------------
# Initialize the VGGT model and load pretrained weights
# -------------------------------
model = VGGT()
checkpoint = torch.load('../../models/vggt/model.pt', map_location=device)
msg = model.load_state_dict(checkpoint)
print('loading status:', msg)
model = model.to(device).eval()

# -------------------------------
# Data storage path settings
# -------------------------------
root_save_3d_feature = '../../data/feature_vggt'
file_path = '../../data/idx.jsonl'

if not os.path.exists(root_save_3d_feature):
    os.makedirs(root_save_3d_feature)
    print("create dir", root_save_3d_feature)

with open(file_path, 'r', encoding='utf-8') as file:
    for line_num, line in enumerate(file, 1):
        if not line.strip():
            continue
        data = json.loads(line.strip())
        image_input = data.get('image_input')
        idx = data.get('idx')
        scene_save_dir = os.path.join(root_save_3d_feature, str(idx))
        if not os.path.exists(scene_save_dir):
            os.makedirs(scene_save_dir)

        # 输出新图片的文件夹路径
        output_folder = "../../data/resized_images"

        # 调用函数处理图片，并返回新的图片路径
        new_image_paths = process_images(image_input)
        print(new_image_paths)
        images = load_and_preprocess_images(new_image_paths)
        images = images.to(device, non_blocking=True)
        images = images.to(dtype)
        # Add batch dimension, final shape is (1, num_frames, 3, H, W)
        images = images.unsqueeze(0)
        # -------------------------------
        # Model inference, feature extraction
        # -------------------------------
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=True, dtype=dtype):
                aggregated_tokens_list, ps_idx = model.aggregator(images)

        # -------------------------------
        # Save aggregated_tokens_list[-1] and ps_idx
        # -------------------------------
        # Move the output to CPU and convert to numpy format
        print(aggregated_tokens_list[-1].shape)
        feature = aggregated_tokens_list[-1].cpu().numpy()
        # Assume ps_idx is a tensor, otherwise save directly
        ps_idx_np = ps_idx.cpu().numpy() if isinstance(ps_idx, torch.Tensor) else ps_idx

        save_path = os.path.join(scene_save_dir, 'vggt.npz')
        np.savez_compressed(save_path, feature=feature, ps_idx=ps_idx_np)