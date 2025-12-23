import os
import torch
import numpy as np
from tqdm import tqdm
import json
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
import os
import json
import logging
from tqdm import tqdm
from qwen_vl_utils import process_vision_info
# -------------------------------
# Import new model and related utility functions
# -------------------------------
import torch.nn.functional as F
from vggt.models.vggt import VGGT
# from vggt.utils.load_fn import load_and_preprocess_images_square
from vggt.utils.load_fn import load_and_preprocess_images, load_and_preprocess_images_square
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map
from vggt.utils.helper import create_pixel_coordinate_grid, randomly_limit_trues
from vggt.dependency.track_predict import predict_tracks
from vggt.dependency.np_to_pycolmap import batch_np_matrix_to_pycolmap, batch_np_matrix_to_pycolmap_wo_track
from PIL import Image
import os
import scipy
from PIL import Image
import os
from trl import SFTTrainer, SFTConfig
import torch
from safetensors import safe_open
import torch.nn as nn
import scipy.io as sio
import trimesh
import shutil
import copy

def rename_colmap_recons_and_rescale_camera(
    reconstruction, image_paths, original_coords, img_size, shift_point2d_to_original_res=False, shared_camera=False
):
    rescale_camera = True

    for pyimageid in reconstruction.images:
        # Reshaped the padded&resized image to the original size
        # Rename the images to the original names
        pyimage = reconstruction.images[pyimageid]
        pycamera = reconstruction.cameras[pyimage.camera_id]
        pyimage.name = image_paths[pyimageid - 1]

        if rescale_camera:
            # Rescale the camera parameters
            pred_params = copy.deepcopy(pycamera.params)

            real_image_size = original_coords[pyimageid - 1, -2:]
            resize_ratio = max(real_image_size) / img_size
            pred_params = pred_params * resize_ratio
            real_pp = real_image_size / 2
            pred_params[-2:] = real_pp  # center of the image

            pycamera.params = pred_params
            pycamera.width = real_image_size[0]
            pycamera.height = real_image_size[1]

        if shift_point2d_to_original_res:
            # Also shift the point2D to original resolution
            top_left = original_coords[pyimageid - 1, :2]

            for point2D in pyimage.points2D:
                point2D.xy = (point2D.xy - top_left) * resize_ratio

        if shared_camera:
            # If shared_camera, all images share the same camera
            # no need to rescale any more
            rescale_camera = False

    return reconstruction

def save_point_cloud_as_obj(point_data, filename_prefix, data_name):
    """
    保存点云数据为 OBJ 文件
    
    Args:
        point_data: 点云数据 (torch.Tensor 或 numpy.ndarray)
        filename_prefix: 文件名前缀
        data_name: 数据名称，用于打印信息
    """
    # 转换为 numpy 并移到 CPU
    if hasattr(point_data, 'detach'):
        points = point_data.detach().cpu().numpy()
    else:
        points = point_data
    
    print(f"{data_name} shape: {points.shape}")
    
    # 处理不同的形状
    if len(points.shape) == 4:  # (4, 518, 518, 3) 或 (1, 4, 518, 518, 3) 去掉第一维
        if points.shape[0] == 1:
            points = points.squeeze(0)  # 去掉 batch 维度
        
        # 现在应该是 (4, 518, 518, 3)，为每个视图保存单独的文件
        for view_idx in range(points.shape[0]):
            view_points = points[view_idx]  # (518, 518, 3)
            
            # 重塑为 (N, 3)
            view_points_flat = view_points.reshape(-1, 3)
            
            # 过滤无效点
            valid_mask = (~np.isnan(view_points_flat).any(axis=1) & 
                         ~np.isinf(view_points_flat).any(axis=1) &
                         (np.abs(view_points_flat).sum(axis=1) > 1e-6))  # 过滤零点
            
            valid_points = view_points_flat[valid_mask]
            
            # 保存单个视图
            obj_filename = f"{filename_prefix}_view_{view_idx}.obj"
            with open(obj_filename, 'w') as f:
                f.write(f"# {data_name} - View {view_idx}\n")
                f.write(f"# Original shape: {points.shape}\n")
                f.write(f"# Valid points: {len(valid_points)}\n")
                
                for point in valid_points:
                    f.write(f"v {point[0]:.6f} {point[1]:.6f} {point[2]:.6f}\n")
            
            print(f"View {view_idx} saved to {obj_filename} with {len(valid_points)} valid points")
        
        # 同时保存合并的所有视图
        all_points = points.reshape(-1, 3)
        valid_mask_all = (~np.isnan(all_points).any(axis=1) & 
                         ~np.isinf(all_points).any(axis=1) &
                         (np.abs(all_points).sum(axis=1) > 1e-6))
        valid_points_all = all_points[valid_mask_all]
        
        combined_filename = f"{filename_prefix}_combined.obj"
        with open(combined_filename, 'w') as f:
            f.write(f"# {data_name} - All Views Combined\n")
            f.write(f"# Original shape: {points.shape}\n")
            f.write(f"# Total valid points: {len(valid_points_all)}\n")
            
            for point in valid_points_all:
                f.write(f"v {point[0]:.6f} {point[1]:.6f} {point[2]:.6f}\n")
        
        print(f"Combined point cloud saved to {combined_filename} with {len(valid_points_all)} valid points")
    
    else:
        # 其他形状的处理
        points_flat = points.reshape(-1, 3)
        valid_mask = (~np.isnan(points_flat).any(axis=1) & 
                     ~np.isinf(points_flat).any(axis=1) &
                     (np.abs(points_flat).sum(axis=1) > 1e-6))
        valid_points = points_flat[valid_mask]
        
        obj_filename = f"{filename_prefix}.obj"
        with open(obj_filename, 'w') as f:
            f.write(f"# {data_name}\n")
            f.write(f"# Original shape: {points.shape}\n")
            f.write(f"# Valid points: {len(valid_points)}\n")
            
            for point in valid_points:
                f.write(f"v {point[0]:.6f} {point[1]:.6f} {point[2]:.6f}\n")
        
        print(f"Point cloud saved to {obj_filename} with {len(valid_points)} valid points")
    
def save_image_paths_object(paths, filename):
    data = {
        "images": paths,
        "count": len(paths),
        "created_at": "2025"
    }
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
        
def save_tensor_to_mat_v1(tensor, filename):
    # 将torch张量转换为numpy数组
    numpy_array = tensor.detach().cpu().numpy()
    
    # 保存到.mat文件
    sio.savemat(filename, {'tensor_data': numpy_array})
    print(f"张量已保存到 {filename}")
    
class projector(nn.Module):
    def __init__(self, mlp_depth, mm_hidden_size, hidden_size, latent_size, fusion_input, fusion_output):
        # [N, mm_hidden_size] -> [N, hidden_size]
        super().__init__()
        ## mm_hidden_size不可改，mlp_depth和hidden_size可以
        self.mlp_depth = mlp_depth
        self.mm_hidden_size = mm_hidden_size
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.fusion_input = fusion_input # P_image
        self.fusion_output = fusion_output # P_3D = 1374
        # self.bn0 = nn.BatchNorm1d(64)
        # self.bn = nn.BatchNorm1d(fusion_output)
        
        modules = [nn.Linear(mm_hidden_size, hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(hidden_size, hidden_size))
        self.proj_3d_model = nn.Sequential(*modules)
        
        
        modules = [nn.Linear(latent_size, 64)]
        for _ in range(1, 2):
            modules.append(nn.GELU())
            modules.append(nn.Linear(64, 64))
        self.prompte_model = nn.Sequential(*modules)

        modules = [nn.Linear(fusion_input+64, fusion_output)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(fusion_output, fusion_output))
        self.fusion_model = nn.Sequential(*modules)

    def forward(self, hidden_states_feature, image_embeddings):
        hidden_states_feature = hidden_states_feature.unsqueeze(0) # [1, T=12, 2048]
        hidden_states_feature = self.prompte_model(hidden_states_feature.squeeze().permute(1,0)) # [B,64,C]   
        hidden_states_feature = hidden_states_feature.permute(1,0).unsqueeze(0)
        # hidden_states_feature = self.conv0(hidden_states_feature) # [B,64,C]
        # image_embeddings: [1,N=4,P=391,2048]
        x_list = torch.tensor([]).to(hidden_states_feature.device, dtype=hidden_states_feature.dtype)
        # print(image_embeddings.shape) # [1, 4, 391, 2048]
        for i in range(image_embeddings.shape[1]):
            image_feature = image_embeddings[:,i,:,:] # [B,P,C]
            # hidden_states_feature: [B,64,C], image_feature: [B,P,C]
            x = torch.cat((hidden_states_feature, image_feature), dim=1) # [B,64+P,C]
            x_unify = self.fusion_model(x.squeeze().permute(1,0)) # [B,P_3D,C]
            x_unify = x_unify.permute(1,0)
            # x_unify = self.conv(x) # [B,P_3D,C]
            x_output = self.proj_3d_model(x_unify) # [B,P_3D,C]
            # Append x_output to x_list
            if x_list.numel() == 0:
                x_list = x_output.unsqueeze(0)
            else:
                x_list = torch.cat((x_list, x_output.unsqueeze(0)), dim=0)
        feature_proj = x_list.squeeze(0)
        # print(feature_proj.shape)
        # print(feature_3d.shape)
        # x_list # [N,P_3D,C]

        return feature_proj
    
def load_model_with_projector(checkpoint_path):
    """完整加载包含projector的模型"""
    
    # 1. 加载所有权重
    all_weights = {}
    for filename in os.listdir(checkpoint_path):
        if filename.endswith('.safetensors'):
            filepath = os.path.join(checkpoint_path, filename)
            with safe_open(filepath, framework="pt", device="cpu") as f:
                for key in f.keys():
                    all_weights[key] = f.get_tensor(key)
    
    # 2. 分离权重
    qwen_weights = {}
    projector_weights = {}
    
    for key, value in all_weights.items():
        if key.startswith('projector_model'):
            projector_weights[key] = value
        else:
            qwen_weights[key] = value
    
    # 4. 返回模型和projector权重
    return projector_weights

# 移除多余的前缀
def remove_prefix_from_state_dict(state_dict, prefix="projector_model."):
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith(prefix):
            new_key = key[len(prefix):]
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value
    return new_state_dict

def extract_embeddings_between_ids_default(predict_embeddings, output_ids, latent_size, start_id=151652, end_id=151653):
    """
    从predict_embeddings中提取在output_ids中start_id和end_id之间的embedding
    
    Args:
        predict_embeddings: torch.Tensor, shape [1, 2223, 2048]
        output_ids: torch.Tensor, shape [1, 2224]
        start_id: int, 起始ID (默认151652)
        end_id: int, 结束ID (默认151653)
    
    Returns:
        extracted_embeddings: torch.Tensor, 提取的embedding
        start_idx: int, 起始位置索引
        end_idx: int, 结束位置索引
    """
    
    # 获取predict_embeddings所在的设备
    device = predict_embeddings.device
    
    # 确保output_ids在同一设备上
    if output_ids.device != device:
        output_ids = output_ids.to(device)
    
    # 将output_ids转换为一维tensor便于处理
    ids = output_ids.squeeze(0)  # shape: [2224]
    
    # 确保start_id和end_id也在同一设备上（转换为tensor）
    start_id_tensor = torch.tensor(start_id, device=device)
    end_id_tensor = torch.tensor(end_id, device=device)
    
    # 寻找start_id和end_id的位置
    start_positions = (ids == start_id_tensor).nonzero(as_tuple=True)[0]
    end_positions = (ids == end_id_tensor).nonzero(as_tuple=True)[0]
    
    if len(start_positions) == 0:
        print(f"未找到起始ID {start_id}")
        return None, None, None
    
    if len(end_positions) == 0:
        print(f"未找到结束ID {end_id}")
        return None, None, None
    
    # 取第一个匹配的位置
    start_idx = start_positions[0].item()
    end_idx = end_positions[0].item()
    
    # 确保end_idx在start_idx之后
    if end_idx <= start_idx:
        print(f"结束位置 {end_idx} 不在起始位置 {start_idx} 之后")
        return None, None, None
    
    # 注意：predict_embeddings的长度是2223，而output_ids的长度是2224
    # 需要确保索引不超出predict_embeddings的范围
    max_embed_idx = predict_embeddings.shape[1] - 1  # 2222
    
    # 调整索引以适应embedding的维度
    embed_start_idx = min(start_idx, max_embed_idx)
    embed_end_idx = min(end_idx, max_embed_idx + 1)  # +1因为是切片的结束位置
    
    # 提取embedding (不包括start_id和end_id本身，只提取中间部分)
    # if embed_start_idx + 1 >= embed_end_idx:
    #     print("起始和结束ID之间没有embedding")
    #     return torch.empty(1, 0, 2048, device=device), start_idx, end_idx
    
    extracted_embeddings = predict_embeddings[:, embed_start_idx+1:embed_start_idx+latent_size+1, :]
    
    print(f"找到起始ID {start_id} 在位置 {start_idx}")
    print(f"找到结束ID {end_id} 在位置 {end_idx}")
    print(f"提取的embedding shape: {extracted_embeddings.shape}")
    
    return extracted_embeddings, start_idx, end_idx

def extract_embeddings_between_ids(predict_embeddings, output_ids, start_id=151652, end_id=151653):
    """
    从predict_embeddings中提取在output_ids中start_id和end_id之间的embedding
    
    Args:
        predict_embeddings: torch.Tensor, shape [1, 2223, 2048]
        output_ids: torch.Tensor, shape [1, 2224]
        start_id: int, 起始ID (默认151652)
        end_id: int, 结束ID (默认151653)
    
    Returns:
        extracted_embeddings: torch.Tensor, 提取的embedding
        start_idx: int, 起始位置索引
        end_idx: int, 结束位置索引
    """
    
    # 获取predict_embeddings所在的设备
    device = predict_embeddings.device
    
    # 确保output_ids在同一设备上
    if output_ids.device != device:
        output_ids = output_ids.to(device)
    
    # 将output_ids转换为一维tensor便于处理
    ids = output_ids.squeeze(0)  # shape: [2224]
    
    # 确保start_id和end_id也在同一设备上（转换为tensor）
    start_id_tensor = torch.tensor(start_id, device=device)
    end_id_tensor = torch.tensor(end_id, device=device)
    
    # 寻找start_id和end_id的位置
    start_positions = (ids == start_id_tensor).nonzero(as_tuple=True)[0]
    end_positions = (ids == end_id_tensor).nonzero(as_tuple=True)[0]
    
    if len(start_positions) == 0:
        print(f"未找到起始ID {start_id}")
        return None, None, None
    
    if len(end_positions) == 0:
        print(f"未找到结束ID {end_id}")
        return None, None, None
    
    # 取第一个匹配的位置
    start_idx = start_positions[0].item()
    end_idx = end_positions[0].item()
    
    # 确保end_idx在start_idx之后
    if end_idx <= start_idx:
        print(f"结束位置 {end_idx} 不在起始位置 {start_idx} 之后")
        return None, None, None
    
    # 注意：predict_embeddings的长度是2223，而output_ids的长度是2224
    # 需要确保索引不超出predict_embeddings的范围
    max_embed_idx = predict_embeddings.shape[1] - 1  # 2222
    
    # 调整索引以适应embedding的维度
    embed_start_idx = min(start_idx, max_embed_idx)
    embed_end_idx = min(end_idx, max_embed_idx + 1)  # +1因为是切片的结束位置
    
    # 提取embedding (不包括start_id和end_id本身，只提取中间部分)
    if embed_start_idx + 1 >= embed_end_idx:
        print("起始和结束ID之间没有embedding")
        return torch.empty(1, 0, 2048, device=device), start_idx, end_idx
    
    extracted_embeddings = predict_embeddings[:, embed_start_idx+1:embed_end_idx, :]
    
    print(f"找到起始ID {start_id} 在位置 {start_idx}")
    print(f"找到结束ID {end_id} 在位置 {end_idx}")
    print(f"提取的embedding shape: {extracted_embeddings.shape}")
    
    return extracted_embeddings, start_idx, end_idx

# 包含边界的版本也需要类似修复
def extract_embeddings_including_boundaries(predict_embeddings, output_ids, start_id=151652, end_id=151653):
    """
    提取包含起始和结束ID对应的embedding
    """
    # 获取predict_embeddings所在的设备
    device = predict_embeddings.device
    
    # 确保output_ids在同一设备上
    if output_ids.device != device:
        output_ids = output_ids.to(device)
    
    ids = output_ids.squeeze(0)
    
    # 确保ID也在同一设备上
    start_id_tensor = torch.tensor(start_id, device=device)
    end_id_tensor = torch.tensor(end_id, device=device)
    
    start_positions = (ids == start_id_tensor).nonzero(as_tuple=True)[0]
    end_positions = (ids == end_id_tensor).nonzero(as_tuple=True)[0]
    
    if len(start_positions) == 0 or len(end_positions) == 0:
        return None, None, None
    
    start_idx = start_positions[0].item()
    end_idx = end_positions[0].item()
    
    if end_idx <= start_idx:
        return None, None, None
    
    max_embed_idx = predict_embeddings.shape[1] - 1
    embed_start_idx = min(start_idx, max_embed_idx)
    embed_end_idx = min(end_idx + 1, max_embed_idx + 1)
    
    extracted_embeddings = predict_embeddings[:, embed_start_idx:embed_end_idx, :]
    
    return extracted_embeddings, start_idx, end_idx

def mask_image_output_tokens(
    input_ids: torch.Tensor,
    image_start_token: int,
    image_token: int
) -> torch.Tensor:
    """
    Creates a mask of the same shape as `input_ids`, with 1's wherever we want to
    'mask out' <image_token> after the first <image_start_token> has appeared,
    and 0's everywhere else.

    Args:
      input_ids: shape [batch_size, seq_len]
      image_start_token: the token ID that marks the start of an image chunk
      image_token: the token ID for image tokens

    Returns:
      A mask (torch.Tensor of the same shape) containing 0/1:
        - 1 = this position should be masked
        - 0 = this position is kept
    """
    batch_size, seq_len = input_ids.shape
    mask = torch.zeros_like(input_ids)

    for i in range(batch_size):
        seq = input_ids[i]
        # Find first occurrence of image_start_token
        first_start_pos = -1
        for j in range(seq_len):
            if seq[j] == image_start_token:
                first_start_pos = j
                break
        
        if first_start_pos == -1:
            continue
        
        # For every position after the first <image_start_token>,
        # if the token is <image_token>, set mask = 1
        for k in range(first_start_pos + 1, seq_len):
            if seq[k] == image_token:
                mask[i, k] = 1

    return mask

def place_input_image(text, image_pad="<|vision_start|><|image_pad|><|vision_end|>", image_placeholder="<image>", sep_token="<|im_start|>assistant") -> str:

    if sep_token is not None:
        assert sep_token in text

        t1, t2 = text.split(sep_token)

        if image_placeholder in t1:
            t1 = t1.replace(image_pad, '')
            t1 = t1.replace(image_placeholder, image_pad)

        return t1 + sep_token + t2
    else:
        return text.replace(image_pad, '').replace(image_placeholder, image_pad)
    
def resize_image(image_path, new_size=(480, 640)):
    # 打开图片
    with Image.open(image_path) as img:
        # 进行缩放
        resized_img = img.resize(new_size, Image.LANCZOS)

        # 构建新的文件路径，将文件夹名称替换为目标文件夹名称
        new_image_path = image_path.replace("other_all_image", "resized_images_visual")
        
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
# Initialize the 3dthinker model and load pretrained weights
# -------------------------------

load_model_path = "../models/3DThinker-S1-Qwen2.5-VL-3B_mlp6_lr1e-4_latent12"
processor = AutoProcessor.from_pretrained(load_model_path, trust_remote_code=True)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(load_model_path, device_map="auto", torch_dtype=torch.bfloat16)
processor.tokenizer.add_tokens("<|latent_pad|>", special_tokens=True)
processor.tokenizer.add_tokens("<|latent_start|>", special_tokens=True)
processor.tokenizer.add_tokens("<|latent_end|>", special_tokens=True)
latent_token_idx = processor.tokenizer("<|latent_pad|>", return_tensors="pt")["input_ids"][0] # 151665
latent_start_idx = processor.tokenizer("<|latent_start|>", return_tensors="pt")["input_ids"][0] # 151666
latent_end_idx = processor.tokenizer("<|latent_end|>", return_tensors="pt")["input_ids"][0] # 151667

model.eval()

projector_weights = load_model_with_projector(load_model_path)
latent_size = 16
projector_model = projector(mlp_depth=4, mm_hidden_size=2048, hidden_size=2048, latent_size = latent_size, fusion_input = 391, fusion_output = 1374).to(model.device)

# 加载权重
projector_weights = remove_prefix_from_state_dict(projector_weights)
projector_model.load_state_dict(projector_weights)

# 设置为评估模式
projector_model.eval()

# -------------------------------
# Initialize the VGGT model and load pretrained weights
# -------------------------------
vggtmodel = VGGT()
checkpoint = torch.load('../models/vggt/model.pt', map_location=device)
scene_dir = "recon_results"
msg = vggtmodel.load_state_dict(checkpoint)
print('loading status:', msg)
vggtmodel = vggtmodel.to(device).eval()

# -------------------------------
# Data storage path settings
# -------------------------------
file_path = '../MindCube-main/data/prompts/general/MindCube_tinybench_raw_qa.jsonl'

## mid
# PROMPT = "\nFirst think about the reasoning process in the mind and then provide the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively. Special tokens should be used to represent 3D imagery during the reasoning process, i.e., <think> reasoning process with special tokens here </think><answer> answer here </answer>."
## end
# PROMPT = "\nFirst think about the reasoning process in the mind and then provide the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively. Special tokens should be used to represent mental 3D scene at the end of your response, i.e., <think> reasoning process here </think><answer> answer here </answer>...mental 3D scene here."
## begin
PROMPT = "\nFirst imagine the mental 3D scene, think about the reasoning process in the mind and then provide the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively. Special tokens should be used to represent mental 3D scene at the beginning of your response, i.e., mental 3D scene here <think> reasoning process here </think><answer> answer here </answer>."

with open(file_path, 'r', encoding='utf-8') as file:
    num = 1
    for line_num, line in enumerate(file, 1):
        if not line.strip():
            continue
        data = json.loads(line.strip())
        image_input = data.get('image_input')
        idx = data.get('idx')
        prompt = data.get('text_input')
        answer = data.get('mindcube_output')
        # 调用函数处理图片，并返回新的图片路径
        new_image_paths = process_images(image_input)
        
        print(new_image_paths)
        
        content_list = []
        for img_input in new_image_paths:
            image = Image.open(img_input).convert("RGB")
            content_list.append({'type': 'image', 'image': image})
            
        # prompt = "Based on these four images (image 1, 2, 3, and 4) showing the black sneaker from different viewpoints (front, left, back, and right), with each camera aligned with room walls and partially capturing the surroundings: From the viewpoint presented in image 1, what is to the left of the black sneaker? A. Washing machine B. Wall and window C. Curtain D. White wood rack"    
        question = prompt+PROMPT
        image_prompt = "<image>"*len(new_image_paths)
        content_list.append({'type': 'text', 'text': image_prompt + question})
        print(f"prompt:{question}")
        conversations = [{'role': 'user', 'content': content_list}]

        texts = [processor.apply_chat_template(conversations, tokenize=False)]
        texts = [place_input_image(text, sep_token=None) for text in texts]
        image_inputs, _ = process_vision_info(conversations)

        inputs = processor(text=[t+'<|im_start|>assistant' for t in texts], images=image_inputs, return_tensors="pt", padding=True)
        inputs = inputs.to(model.device)

        # print(inputs["input_ids"][0].tolist())
        # 直接使用模型进行前向传播
        with torch.no_grad():  # 如果不需要梯度
            outputs = model(**inputs)
            input_embeddings = outputs.inputs_embeds
            # predict_embeddings = outputs.hidden_states
            mask = (inputs["input_ids"][0] == 151655).int()
            mask = mask.unsqueeze(0)
            image_embeddings = input_embeddings[mask.to(input_embeddings.device) != 0].contiguous()
            image_tokens = image_embeddings.shape[0]
            image_embed_dim = image_embeddings.shape[1]
            image_number = len(new_image_paths)
            
            patch_size = int(image_tokens/image_number)
            image_embeddings = image_embeddings.view(image_number, patch_size, image_embed_dim).unsqueeze(0)
            generation_outputs = model.generate(
                **inputs,
                max_new_tokens=2048,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                tokenizer=processor.tokenizer,
                output_hidden_states=True,
                return_dict_in_generate=True,
            )
            output_ids = generation_outputs.sequences
            generated_hidden_states = generation_outputs.hidden_states
            predict_embeddings = torch.cat(generated_hidden_states, dim=1)
            # extracted_emb, start_pos, end_pos = extract_embeddings_between_ids(predict_embeddings, output_ids, start_id=latent_start_idx, end_id=latent_end_idx)
            extracted_emb, start_pos, end_pos = extract_embeddings_between_ids_default(predict_embeddings, output_ids, latent_size, start_id=latent_start_idx, end_id=latent_end_idx)
            if extracted_emb.shape[1] == latent_size:
                target_dtype = torch.float32  # 或者 torch.bfloat16
                decoder_feature = projector_model(extracted_emb.to(target_dtype), image_embeddings.to(target_dtype))
                
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
                        aggregated_tokens_list, ps_idx = vggtmodel.aggregator(images)
                        aggregated_tokens_list_org = copy.deepcopy(aggregated_tokens_list)
                ## replace with the mllm feature
                decoder_feature = decoder_feature.unsqueeze(0)
                aggregated_tokens_list[-1] = decoder_feature
                # Predict Cameras
                pose_enc = vggtmodel.camera_head(aggregated_tokens_list)[-1]
                pose_enc_org = vggtmodel.camera_head(aggregated_tokens_list_org)[-1]
                # Extrinsic and intrinsic matrices, following OpenCV convention (camera from world)
                extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])
                extrinsic_org, intrinsic_org = pose_encoding_to_extri_intri(pose_enc_org, images.shape[-2:])
                # Predict Depth Maps
                depth_map, depth_conf = vggtmodel.depth_head(aggregated_tokens_list, images, ps_idx)
                depth_map_org, depth_conf_org = vggtmodel.depth_head(aggregated_tokens_list_org, images, ps_idx)
                # Predict Point Maps
                point_map, point_conf = vggtmodel.point_head(aggregated_tokens_list, images, ps_idx)
                point_map_org, point_conf_org = vggtmodel.point_head(aggregated_tokens_list_org, images, ps_idx)
                
                extrinsic = extrinsic.squeeze(0).cpu().numpy()
                intrinsic = intrinsic.squeeze(0).cpu().numpy()
                depth_map = depth_map.squeeze(0).cpu().numpy()
                depth_conf = depth_conf.squeeze(0).cpu().numpy()
                points_3d = unproject_depth_map_to_point_map(depth_map, extrinsic, intrinsic)
                
                extrinsic_org = extrinsic_org.squeeze(0).cpu().numpy()
                intrinsic_org = intrinsic_org.squeeze(0).cpu().numpy()
                depth_map_org = depth_map_org.squeeze(0).cpu().numpy()
                depth_conf_org = depth_conf_org.squeeze(0).cpu().numpy()
                points_3d_org = unproject_depth_map_to_point_map(depth_map_org, extrinsic_org, intrinsic_org)
                
                conf_thres_value = 1
                max_points_for_colmap = 100000  # randomly sample 3D points
                shared_camera = False  # in the feedforward manner, we do not support shared camera
                camera_type = "PINHOLE"  # in the feedforward manner, we only support PINHOLE camera
                vggt_fixed_resolution = 518
                
                image_size = np.array([vggt_fixed_resolution, vggt_fixed_resolution])
                num_frames, height, width, _ = points_3d.shape
                
                points_rgb = images.squeeze(0) # num_frames, 3, H, W
                points_rgb = (points_rgb.float().cpu().numpy() * 255).astype(np.uint8)
                points_rgb_mid = points_rgb.transpose(0, 2, 3, 1)

                # (S, H, W, 3), with x, y coordinates and frame indices
                points_xyf = create_pixel_coordinate_grid(num_frames, height, width)

                conf_mask = depth_conf >= conf_thres_value
                # at most writing 100000 3d points to colmap reconstruction object
                conf_mask = randomly_limit_trues(conf_mask, max_points_for_colmap)

                points_3d = points_3d[conf_mask]
                points_xyf = points_xyf[conf_mask]
                points_rgb = points_rgb_mid[conf_mask]

                conf_mask_org = depth_conf_org >= conf_thres_value
                conf_mask_org = randomly_limit_trues(conf_mask_org, max_points_for_colmap)
                # at most writing 100000 3d points to colmap reconstruction object
                points_3d_org = points_3d_org[conf_mask_org]
                points_rgb_org = points_rgb_mid[conf_mask_org]
                
                print("Converting to COLMAP format")
                print(points_3d.shape)
                print(points_xyf.shape)
                print(points_rgb.shape)
                reconstruction = batch_np_matrix_to_pycolmap_wo_track(
                    points_3d,
                    points_xyf,
                    points_rgb,
                    extrinsic,
                    intrinsic,
                    image_size,
                    shared_camera=shared_camera,
                    camera_type=camera_type,
                )

                print(f"Saving reconstruction to {scene_dir}/sparse_{num}")
                sparse_reconstruction_dir = os.path.join(scene_dir, f"sparse_{num}")
                os.makedirs(sparse_reconstruction_dir, exist_ok=True)
                reconstruction.write(sparse_reconstruction_dir)

                # Save point cloud for fast visualization
                trimesh.PointCloud(points_3d, colors=points_rgb).export(os.path.join(scene_dir, f"sparse_{num}/points.ply"))
                trimesh.PointCloud(points_3d_org, colors=points_rgb_org).export(os.path.join(scene_dir, f"sparse_{num}/vggt_points.ply"))
                
                for img_path in new_image_paths:
                    img_name = os.path.basename(img_path)  # 提取原始文件名
                    dst_path = os.path.join(sparse_reconstruction_dir, img_name)
                    shutil.copy(img_path, dst_path)
                # Save prompt and generated answer
                # prompt_text = processor.tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)
                generated_text = processor.tokenizer.decode(output_ids[0], skip_special_tokens=True)

                with open(os.path.join(sparse_reconstruction_dir, "prompt.txt"), "w", encoding="utf-8") as f:
                    f.write(prompt)

                with open(os.path.join(sparse_reconstruction_dir, "gt_answer.txt"), "w", encoding="utf-8") as f:
                    f.write(answer)
                    
                with open(os.path.join(sparse_reconstruction_dir, "generated_answer.txt"), "w", encoding="utf-8") as f:
                    f.write(generated_text)
    
                num = num + 1