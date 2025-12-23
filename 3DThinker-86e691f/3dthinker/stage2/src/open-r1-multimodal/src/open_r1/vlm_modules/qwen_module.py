from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2VLForConditionalGeneration, AutoProcessor
from typing import Dict, Any, Union
from trl.data_utils import maybe_apply_chat_template
import torch
import re
from open_r1.vlm_modules.vlm_module import VLMBaseModule
import os
from datetime import datetime
from shapely.geometry import Polygon
from shapely.ops import unary_union
import numpy as np
from scipy.optimize import linear_sum_assignment
import time
import ast
import os
from openai import OpenAI
from transformers.utils.versions import require_version
import json
import base64
from open_r1.trainer.record import reward_record
import cv2
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
import random
import math
import sys
import json
import re
from typing import Dict, List, Tuple, Optional, Any, Union
import torch.nn as nn
from safetensors import safe_open

client = OpenAI(
    api_key = "1955916774340870205",
    base_url = "https://aigc.sankuai.com/v1/openai/native"
)

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

def remove_prefix_from_state_dict(state_dict, prefix="projector_model."):
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith(prefix):
            new_key = key[len(prefix):]
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value
    return new_state_dict

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

## TODO:latent
projector_weights = load_model_with_projector("../../models/3DThinker-S1-Qwen2.5-VL-3B_mlp6_lr1e-4_latent12")
latent_size = 12
projector_model = projector(mlp_depth=6, mm_hidden_size=2048, hidden_size=2048, latent_size = latent_size, fusion_input = 391, fusion_output = 1374).to("cuda:0")
projector_weights = remove_prefix_from_state_dict(projector_weights)
projector_model.load_state_dict(projector_weights)
projector_model.eval()

### finishing loading projector
  
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

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def extract_answer(xml_text):
    # 使用正则表达式匹配<answer>和</answer>之间的内容，包括换行符
    pattern = r'<answer>\n?(.*?)\n?</answer>'  # 处理可能的换行符
    match = re.search(pattern, xml_text, re.DOTALL)
    
    if match:
        # 提取内容并去除首尾空白
        return match.group(1).strip()
    return ""

# def judge_score_func(client, question, gt_response, pred_response, base64_image):
#     question = question[0]['content'][-1]['text'].replace(' First think about the reasoning process in the mind and then provide the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>.','')
#     question = question.replace(' The first image serves as the main image, followed by four generated images that show different perspectives. The last one is the depth image corresponding to the main image.','')
#     question = question.replace(' The first image serves as the main image, followed by four generated images that show different perspectives.','')
#     question = question.replace(' The first image serves as the main image, followed by the depth image corresponding to the main image.','')
#     SYSTEM_PROMPT = f'''You are responsible for proofreading the answers, you need to give the score to the model's answer by referring to the standard answer, based on the given question and image.
#     The full score is 1 point and the minimum score is 0 points. Please directly provide the score in JSON format, for example, {{"score": 0.8}}, without showing the intermediate process.
#     The evaluation criteria require that the closer the model's answer is to the standard answer, the higher the score.
#     '''
#     PROMPT = f'''
#     Question: {question}
#     Standard answer: {gt_response}
#     Model's answer: {pred_response}
#     '''
#     messages_list = [
#         {
#             "role": "system",
#             "content": SYSTEM_PROMPT
#         },
#         {
#             "role": "user",
#             "content": [
#                 {"type": "text", "text": PROMPT},
#                 {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}},
#             ],
#         }]
    
#     result = client.chat.completions.create(
#         model="gpt-4o-2024-11-20",
#         messages=messages_list,
#         stream=False,
#         extra_headers={
#             "M-TraceId": "2136218312678"
#         }
#     )
#     response = result.choices[0].message.content
#     return response

class Qwen2VLModule(VLMBaseModule):
    def __init__(self):
        super().__init__()

    def get_vlm_key(self):
        return "qwen"

    def get_model_class(self, model_id: str, model_init_kwargs: dict):
        if "Qwen2-VL" in model_id:
            model_cls = Qwen2VLForConditionalGeneration
        elif "Qwen2.5-VL" in model_id:
            model_cls = Qwen2_5_VLForConditionalGeneration
        else:
            raise ValueError(f"Unsupported model: {model_id}")
        return model_cls
    
    def post_model_init(self, model, processing_class):
        pass
    
    def get_processing_class(self):
        return AutoProcessor
    
    def get_vision_modules_keywords(self):  
        return ['visual']
    
    def get_custom_multimodal_keywords(self):
        return ['pixel_values', 'image_grid_thw']

    def get_non_generate_params(self):
        return []
    
    def get_custom_processing_keywords(self):
        return [('image_processor', 'max_pixels'), ('image_processor', 'min_pixels')]
    
    def prepare_prompt(self, processing_class, inputs: dict[str, Union[torch.Tensor, Any]]):
        prompts_text = [maybe_apply_chat_template(example, processing_class)["prompt"] for example in inputs]
        return prompts_text
    
    def prepare_model_inputs(self, processing_class, prompts_text, images, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False):
        # FIXME
        # This could only process pure-multimodal or pure-text inputs
        if len(images) > 0:
            prompt_inputs = processing_class(
                text=prompts_text,
                images=images,
                return_tensors=return_tensors,
                padding=padding,
                padding_side=padding_side,
                add_special_tokens=add_special_tokens)
        else:
            prompt_inputs = processing_class(
                text=prompts_text,
                return_tensors=return_tensors,
                padding=padding,
                padding_side=padding_side,
                add_special_tokens=add_special_tokens)
        return prompt_inputs
    
    @staticmethod
    def get_question_template(task_type: str):
        ## todo
        match task_type:
            case "rec":
                # return "{Question}\nFirst, visualize a mental 3D scene in your mind. Represent this mental 3D scene using special tokens, where <|latent_start|> marks the beginning, <|latent_end|> marks the end, and <|latent_pad|> tokens are placed between <|latent_start|> and <|latent_end|>. For example, <|latent_start|><|latent_pad|><|latent_pad|>...<|latent_end|>. Then, offer the user the reasoning process and the answer. Enclose the reasoning process within <think> </think> tags and the answer within <answer> </answer> tags. Your response should start with the representation of the mental 3D scene using the special tokens, i.e., <|latent_start|><|latent_pad|><|latent_pad|>...<|latent_end|><think> reasoning process here </think><answer> answer here </answer>."
                # return "{Question}\nFirst imagine the mental 3D scene using special tokens like <|latent_start|><|latent_pad|><|latent_pad|>...<|latent_end|> in the mind and then provide the user with the reasoning process and answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively. Special tokens should be used to represent mental 3D scene at the beginning of your response, i.e., <|latent_start|><|latent_pad|><|latent_pad|>...<|latent_end|><think> reasoning process here </think><answer> answer here </answer>."
                return "{Question}\nFirst imagine the mental 3D scene, think about the reasoning process in the mind and then provide the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively. Special tokens should be used to represent mental 3D scene at the beginning of your response, i.e., mental 3D scene here <think> reasoning process here </think><answer> answer here </answer>."
                # return "{Question} First think about the reasoning process in the mind and then provide the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>."
            case "ic":
                return "{Question} First think about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>."
            case "odLength":
                SYSTEM_PROMPT = (
                    "First thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
                    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
                    "<think> reasoning process here </think><answer> answer here </answer>"
                )
                return SYSTEM_PROMPT + '\n' + "{Question}"
            case _:
                return "{Question} First output the thinking process in <think> </think> tags and then output the final answer in <answer> </answer> tags."
    
    
    ## TODO: 添加了latent校验逻辑   
    @staticmethod
    def format_reward(completions, **kwargs):
        # pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
        # pattern = r".*?<\|latent_start\|>(<\|latent_pad\|>)+<\|latent_end\|>.*?<think>.*?</think>\s*<answer>.*?</answer>"
        pattern = r".*?<\|latent_start\|>.*?<think>.*?</think>\s*<answer>.*?</answer>"
        # pattern = r"<think>.*?<\|latent_start\|>(<\|latent_pad\|>)+<\|latent_end\|>.*?</think>\s*<answer>.*?</answer>"
        completion_contents = [completion[0]["content"] for completion in completions]
        matches = [re.search(pattern, content, re.DOTALL) is not None for content in completion_contents]
        rewards = [1.0 if match else 0.0 for match in matches]
        print("--------Format Reward--------")
        for content, reward in zip(completion_contents, rewards):
            print(content)
            if reward == 1.0:
                print(f"Matching Content: {content}")
        print(f"Reward: {rewards}")
        return rewards

    # @staticmethod
    # def response_text_reward(completions, answer, prompts, img_path_str, **kwargs):
    #     contents = [completion[0]["content"] for completion in completions]
    #     rewards = []
    #     current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    #     for content, gt_response, prompt, img in zip(contents, answer, prompts,img_path_str):
    #         reward = 0.0
    #         try:
    #             # base64_image = encode_image(img)
    #             pred_response = extract_answer(content)
    #             if pred_response.strip().lower() in gt_response.strip().lower():
    #                 reward = 1.0
    #         except Exception:
    #              print("service error")
    #              pass  # Continue to next verification method if this fails

    #         print("--------Response Reward--------")
    #         print(f"Question:{prompt[0]['content'][-1]['text']}, GT:{gt_response}, Pred:{pred_response}, Response Accuracy: {reward}")

    #         rewards.append(reward)
    #         if os.getenv("DEBUG_MODE") == "true":
    #             log_path = os.getenv("LOG_PATH")
    #             with open(log_path, "a", encoding='utf-8') as f:
    #                 f.write(f"------------- {current_time} response_reward Accuracy: {reward} -------------\n")
    #                 f.write(f"Content: {content}\n")
    #                 f.write(f"Answer: {gt_response}\n")
    #     return rewards

    @staticmethod
    def sim_reward(index, prompts, image_embeddings, extracted_emb, **kwargs):
        rewards = []
        current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
        for idx, prompt in zip(index, prompts):
            reward = 0.0
            try:
                target_dtype = torch.float32  # 或者 torch.bfloat16
                decoder_feature = projector_model(extracted_emb.to(target_dtype).to("cuda:0"), image_embeddings.to(target_dtype).to("cuda:0"))
                decoder_feature_norm = decoder_feature / decoder_feature.norm(dim=-1, p=2, keepdim=True)

                data = np.load('../../data/feature_vggt/' + str(idx) + '/vggt.npz')
                feature_3d = data['feature'] # [1,N=4,P_3D = 1374,2048]
                feature_3d = torch.tensor(feature_3d).to(device=decoder_feature.device, dtype=decoder_feature.dtype)
                feature_3d = feature_3d.squeeze()
                feature_3d_norm = feature_3d / feature_3d.norm(dim=-1, p=2, keepdim=True)
                cos_sim = torch.nn.functional.cosine_similarity(decoder_feature_norm, feature_3d_norm).mean()
                reward = (cos_sim + 1) / 2
            except Exception:
                print("service error")
                pass  # Continue to next verification method if this fails

            print("--------Sim Reward--------")
            # print(f"Question:{prompt[0]['content'][-1]['text']}, content: {content}, GT:{gt_response}, Pred:{pred_response}, Response Accuracy: {reward}")
            print(f"Question:{prompt[0]['content'][-1]['text']}, Sim Accuracy: {reward}")

            rewards.append(reward)
            if os.getenv("DEBUG_MODE") == "true":
                log_path = os.getenv("LOG_PATH")
                with open(log_path, "a", encoding='utf-8') as f:
                    f.write(f"------------- {current_time} sim_reward Accuracy: {reward} -------------\n")
        return rewards
    
    @staticmethod
    def response_text_reward(completions, answer, prompts, **kwargs):
        contents = [completion[0]["content"] for completion in completions]
        rewards = []
        current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
        for content, gt_response, prompt in zip(contents, answer, prompts):
            reward = 0.0
            try:
                pred_response = extract_answer(content)
                if extract_option(pred_response) and (extract_option(gt_response) == extract_option(pred_response)):
                    reward = 1.0
                # judge_score = judge_score_func(client, prompt, gt_response, pred_response, base64_image)
                # try:
                #     score_data = json.loads(judge_score)
                #     reward = score_data["score"]
                # except:
                #     try:
                #         clean_score = re.sub(r'```json|```', '', judge_score).strip()
                #         score_data = json.loads(clean_score)
                #         if isinstance(score_data, float) or isinstance(score_data, int):
                #             reward = score_data
                #         else:
                #             reward = score_data["score"]
                #     except Exception as e:
                #         print(judge_score)
                #         print(f"response fail: {e}")
            except Exception:
                pred_response = ""
                print("service error")
                pass  # Continue to next verification method if this fails

            print("--------Response Reward--------")
            # print(f"Question:{prompt[0]['content'][-1]['text']}, content: {content}, GT:{gt_response}, Pred:{pred_response}, Response Accuracy: {reward}")
            print(f"Question:{prompt[0]['content'][-1]['text']}, GT:{gt_response}, Pred:{pred_response}, Response Accuracy: {reward}")

            rewards.append(reward)
            if os.getenv("DEBUG_MODE") == "true":
                log_path = os.getenv("LOG_PATH")
                with open(log_path, "a", encoding='utf-8') as f:
                    f.write(f"------------- {current_time} response_reward Accuracy: {reward} -------------\n")
                    f.write(f"Content: {content}\n")
                    f.write(f"Answer: {gt_response}\n")
        return rewards

    @staticmethod
    def select_reward_func(func: str, task_type: str):
        if func == "format":
            match task_type:
                case "rec":
                    return Qwen2VLModule.format_reward
                case _:
                    raise ValueError(f"Unsupported reward function: {func}")
        elif func == "response":
            match task_type:
                case "rec":
                    return Qwen2VLModule.response_text_reward
                case _:
                    raise ValueError(f"Unsupported reward function: {func}")
        elif func == "latent_sim":
            match task_type:
                case "rec":
                    return Qwen2VLModule.sim_reward
                case _:
                    raise ValueError(f"Unsupported reward function: {func}")                
        else:
            raise ValueError(f"Unsupported reward function: {func}")