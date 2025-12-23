import torch.nn as nn
from multimodal_projector.pooler_projector import *
from multimodal_projector.builder import *
from typing import List, Optional, Tuple, Union, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
import numpy as np

# def feature_3d_alignment(self, video_dict, hidden_states, img_pos_list=None, img_length_list=None, margin=0.0):
#     C = hidden_states.shape[-1]
#     feature = hidden_states[:,img_pos_list[0]:img_pos_list[0]+img_length_list[0],:].view(32,14,15,C)[:,:,:-1,:].contiguous().view(32*14*14,C)
#     feature_proj = self.model.proj_3d(feature)
#     feature_3d = video_dict['feature_3d']
#     feature_3d = feature_3d.to(device=feature.device, dtype=feature.dtype)
#     feature_3d = feature_3d.squeeze()

#     S, L, D = feature_3d.shape
#     assert feature_proj.shape[-1] == D and S == 32
#     if L == 768:
#         feature_3d = feature_3d.view(S, 24, 32, D).permute(0, 3, 1, 2).contiguous()
#     elif L == 1036:
#         feature_3d = feature_3d.view(S, 28, 37, D).permute(0, 3, 1, 2).contiguous()
#     elif L == 256:
#         feature_3d = feature_3d.view(S, 16, 16, D).permute(0, 3, 1, 2).contiguous()
#     else:
#         raise NotImplementedError

#     feature_3d = F.adaptive_avg_pool2d(feature_3d, (14, 14)).view(S, D, 14, 14)
#     feature_3d = feature_3d.view(S, D, 14*14).permute(0, 2, 1).contiguous().view(S*14*14, D)

#     feature_proj_norm = feature_proj / feature_proj.norm(dim=-1, p=2, keepdim=True)
#     feature_3d_norm = feature_3d / feature_3d.norm(dim=-1, p=2, keepdim=True)

#     feature_sim = ((feature_proj_norm - feature_3d_norm.detach()) ** 2).sum(dim=-1)
#     feature_sim_loss = feature_sim.mean()

#     return feature_sim_loss

# def feature_3d_similarity(self, video_dict, hidden_states, img_pos_list=None, img_length_list=None, tau=0.07):
#     # 1. 从hidden_states提取对应区域，并 reshape 为 (32*14*14, C)
#     C = hidden_states.size(-1)
#     # 假定hidden_states的提取区域可以重构为 (32, 14, 15, C)
#     feature = hidden_states[:, img_pos_list[0]:img_pos_list[0] + img_length_list[0], :].view(32, 14, 15, C)[:, :, :-1, :].contiguous().view(32 * 14 * 14, C)
#     text_feature = hidden_states[:,img_pos_list[0]+img_length_list[0]:,:].view(-1, C).mean(dim=0,keepdim=True)
#     l = text_feature.shape[0]

#     # 重构成 (32, 14, 15, C)，然后沿第二维丢弃最后一个token变为 (32, 14, 14, C)
#     feature_norm = feature / feature.norm(dim=-1, p=2, keepdim=True)
#     text_feature_norm = text_feature / text_feature.norm(dim=-1, p=2, keepdim=True)
#     text2vis_sim = F.softmax((feature_norm @ text_feature_norm.T) / tau,dim=0).view(32*14*14, l)

#     # 2. 投影hidden_states特征（如果self.model中有3d投影层）
#     feature_proj = self.model.proj_3d(feature)  # shape: (32*14*14, D)

#     # 3. 从video_dict中获取3d特征，并调整形状
#     feature_3d = video_dict['feature_3d']
#     feature_3d = feature_3d.to(device=feature_proj.device, dtype=feature_proj.dtype)
#     feature_3d = feature_3d.squeeze()  # 去除多余的维度

#     # 假设feature_3d形状为 (S, L, D)
#     S, L, D = feature_3d.shape
#     assert feature_proj.shape[-1] == D and S == 32, "Hidden特征和3D特征维度/样本数不匹配！"

#     if L == 768:
#         # 假设重构为 (S, 24, 32, D), 再转为 (S, D, 24, 32)
#         feature_3d = feature_3d.view(S, 24, 32, D).permute(0, 3, 1, 2).contiguous()
#     elif L == 1036:
#         # 假设重构为 (S, 28, 37, D), 再转为 (S, D, 28, 37)
#         feature_3d = feature_3d.view(S, 28, 37, D).permute(0, 3, 1, 2).contiguous()
#     elif L == 256:
#         # 假设重构为 (S, 16, 16, D), 再转为 (S, D, 16, 16)
#         feature_3d = feature_3d.view(S, 16, 16, D).permute(0, 3, 1, 2).contiguous()
#     else:
#         raise NotImplementedError(f"Unsupported feature_3d shape with L={L}")

#     # 通过自适应平均池化将空间尺寸调整为 (14, 14)，再 reshape 到 (S*14*14, D)
#     feature_3d = F.adaptive_avg_pool2d(feature_3d, (14, 14)).view(S, D, 14, 14)
#     feature_3d = feature_3d.view(S, D, 14 * 14).permute(0, 2, 1).contiguous().view(S * 14 * 14, D)
    
#     # 4. 特征归一化后计算余弦相似度矩阵
#     feature_proj_norm = feature_proj / feature_proj.norm(dim=-1, p=2, keepdim=True)
#     feature_3d_norm = feature_3d / feature_3d.norm(dim=-1, p=2, keepdim=True) # (L, D)

#     # 计算两个特征矩阵各自两两的余弦相似度，shape均为(32*14*14, 32*14*14)
#     feature_sim = torch.matmul(feature_proj_norm, feature_proj_norm.transpose(0, 1))
#     feature_3d_sim = torch.matmul(feature_3d_norm, feature_3d_norm.transpose(0, 1))
    
#     # 5. 用均方误差来使hidden_states的相似度矩阵与3d特征的相似度矩阵对齐
#     loss_vis = F.l1_loss(feature_sim, feature_3d_sim.detach())

#     return loss_vis

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims = [], bias = True, act = nn.ELU()):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        self.act = act
        if len(hidden_dims)>0:
            fc = [nn.Linear(input_dim, self.hidden_dims[0], bias = bias)]
            fc.append(act)
            for i in range(len(self.hidden_dims)-1):
                fc.append(nn.Linear(self.hidden_dims[i], self.hidden_dims[i+1], bias = bias))
                fc.append(act)
            fc.append(nn.Linear(self.hidden_dims[-1], output_dim, bias = bias))
        else:
            fc = [nn.Linear(input_dim, output_dim, bias = bias), act]
        self.linear = nn.Sequential(*fc)
    
    def forward(self, x):
        return self.linear(x)
    
def build_vision_projector(mlp_depth, mm_hidden_size, hidden_size):
    modules = [nn.Linear(mm_hidden_size, hidden_size)]
    for _ in range(1, mlp_depth):
        modules.append(nn.GELU())
        modules.append(nn.Linear(hidden_size, hidden_size))
    return nn.Sequential(*modules)
    
    
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
        # self.conv0 = nn.Sequential(nn.Conv1d(latent_size, 64, kernel_size=1, bias=False),
        #                            self.bn0,
        #                            nn.LeakyReLU(negative_slope=0.2))  
          
        # self.conv = nn.Sequential(nn.Conv1d(fusion_input+64, fusion_output, kernel_size=1, bias=False),
        #                            self.bn,
        #                            nn.LeakyReLU(negative_slope=0.2))
        
    # [S, L, 2048] -> [N=16, 2048]
    # def forward(self, idx, hidden_states_feature, image_embeddings):
    #     data = np.load('3DThinker/posed_images_3d_feature_vggt_resized/' + str(idx[0]) + '/vggt.npz')
    #     feature_3d = data['feature'] # [1,N=4,P_3D = 1374,2048]
    #     feature_3d = torch.tensor(feature_3d).to(device=hidden_states_feature.device, dtype=hidden_states_feature.dtype)
    #     feature_3d = feature_3d.squeeze()
    #     hidden_states_feature = hidden_states_feature.unsqueeze(0) # [1, T=12, 2048]
    #     hidden_states_feature = self.prompte_model(hidden_states_feature.squeeze().permute(1,0)) # [B,64,C]   
    #     hidden_states_feature = hidden_states_feature.permute(1,0).unsqueeze(0)
    #     # hidden_states_feature = self.conv0(hidden_states_feature) # [B,64,C]
    #     # image_embeddings: [1,N=4,P=391,2048]
    #     x_list = torch.tensor([]).to(hidden_states_feature.device, dtype=hidden_states_feature.dtype)
    #     # print(image_embeddings.shape) # [1, 4, 391, 2048]
    #     for i in range(image_embeddings.shape[1]):
    #         image_feature = image_embeddings[:,i,:,:] # [B,P,C]
    #         # hidden_states_feature: [B,64,C], image_feature: [B,P,C]
    #         x = torch.cat((hidden_states_feature, image_feature), dim=1) # [B,64+P,C]
    #         x_unify = self.fusion_model(x.squeeze().permute(1,0)) # [B,P_3D,C]
    #         x_unify = x_unify.permute(1,0)
    #         # x_unify = self.conv(x) # [B,P_3D,C]
    #         x_output = self.proj_3d_model(x_unify) # [B,P_3D,C]
    #         # Append x_output to x_list
    #         if x_list.numel() == 0:
    #             x_list = x_output.unsqueeze(0)
    #         else:
    #             x_list = torch.cat((x_list, x_output.unsqueeze(0)), dim=0)
    #     feature_proj = x_list.squeeze(0)
    #     # print(feature_proj.shape)
    #     # print(feature_3d.shape)
    #     # x_list # [N,P_3D,C]
    #     feature_proj_norm = feature_proj / feature_proj.norm(dim=-1, p=2, keepdim=True)
    #     feature_3d_norm = feature_3d / feature_3d.norm(dim=-1, p=2, keepdim=True)

    #     print(feature_proj_norm.shape)
    #     print(feature_3d_norm.shape)
    #     feature_sim = ((feature_proj_norm - feature_3d_norm.detach()) ** 2).sum(dim=-1)
    #     feature_sim_loss = feature_sim.mean()

    #     return feature_sim_loss
    
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