from trl import SFTTrainer, SFTConfig
import torch
import wandb
import torch.distributed as dist

class CustomTrainerStage1(SFTTrainer):        
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Compute training loss and additionally compute token accuracies
        """
        idx = inputs['idx']
        del inputs['idx']
        
        (ce_loss, outputs) = super().compute_loss(
            model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch
        )
        predicted_ids = outputs.logits.argmax(dim=-1)
        decoded_text = self.tokenizer.batch_decode(predicted_ids, skip_special_tokens=False)
        predict_embeddings = outputs.hidden_states
        
        # all: 620
        image_out_mask = inputs["image_out_mask"]
        
        shift_image_mask = image_out_mask[:, -(predict_embeddings.shape[1] - 1) :].to(predict_embeddings.device)
        shift_predict_embeddings = predict_embeddings[..., :-1, :][shift_image_mask.to(predict_embeddings.device) != 0].contiguous()

        # k_proj_weight = model.get_parameter('model.layers.17.self_attn.k_proj.weight')
        # x = [[151644, 8948, 198, 2610, 525, 264, 10950, 17847, 13, 151645, 198, 151644, 872, 198, 151652]]
        # decoded_text = self.tokenizer.batch_decode(torch.tensor(x).to("cuda:0"), skip_special_tokens=False)
        
        ## the same
        input_embeddings = outputs.inputs_embeds
        
        mask = (inputs["input_ids"][0] == 151655).int()
        mask = mask.unsqueeze(0)
        image_embeddings = input_embeddings[mask.to(input_embeddings.device) != 0].contiguous()
        
        image_tokens = image_embeddings.shape[0]
        image_embed_dim = image_embeddings.shape[1]
        image_number = inputs["image_grid_thw"].shape[0]
        patch_size = int(image_tokens/image_number)
        image_embeddings = image_embeddings.view(image_number, patch_size, image_embed_dim).unsqueeze(0)
        
        feature_proj = model.projector_model(shift_predict_embeddings, image_embeddings)
        feature_proj_norm = feature_proj / feature_proj.norm(dim=-1, p=2, keepdim=True)
        
        # input_embeddings = outputs.inputs_embeds
        # gt_embeddings = input_embeddings[..., 1:, :][shift_image_mask.to(input_embeddings.device) != 0].contiguous()
        
        # sim_loss = torch.nn.functional.cosine_similarity(gt_embeddings, shift_predict_embeddings).mean()
        # sim_loss = 1 - sim_loss

        data = np.load('../../data/feature_vggt/' + str(idx[0]) + '/vggt.npz')
        feature_3d = data['feature'] # [1,N=4,P_3D = 1374,2048]
        feature_3d = torch.tensor(feature_3d).to(device=shift_predict_embeddings.device, dtype=shift_predict_embeddings.dtype)
        feature_3d = feature_3d.squeeze()
        feature_3d_norm = feature_3d / feature_3d.norm(dim=-1, p=2, keepdim=True)
        
        # print(feature_proj_norm.shape)
        # print(feature_3d_norm.shape)
        # feature_sim = ((feature_proj_norm - feature_3d_norm.detach()) ** 2).sum(dim=-1)
        feature_sim = ((feature_proj - feature_3d.detach()) ** 2).sum(dim=-1)
        sim_loss = feature_sim.mean()*0.0005
        
        loss = 0.1 * ce_loss + sim_loss
        # loss = ce_loss + sim_loss
        
        # 只在主进程记录wandb
        if dist.get_rank() == 0:
            wandb.log({
                "train/ce_loss": ce_loss.item(),
                "train/sim_loss": sim_loss.item(),
                "train/total_loss": loss.item(),
                "train/step": self.state.global_step,
                "train/epoch": self.state.epoch,
            })
        
        if dist.get_rank() == 0:
            print(f"Step {self.state.global_step}: CE Loss: {ce_loss.item():.4f}, Sim Loss: {sim_loss.item():.4f}, Total Loss: {loss.item():.4f}")
        
        return (loss, outputs) if return_outputs else loss

class CustomTrainerStage2(SFTTrainer):
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Compute training loss and additionally compute token accuracies
        """
        (ce_loss, outputs) = super().compute_loss(
            model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch
        )

        loss = ce_loss
        
        # 只在主进程记录wandb
        if dist.get_rank() == 0:
            wandb.log({
                "train/ce_loss": ce_loss.item(),
                "train/total_loss": loss.item(),
                "train/step": self.state.global_step,
                "train/epoch": self.state.epoch,
            })
            print(f"Step {self.state.global_step}: CE Loss: {ce_loss.item():.4f}")
        
        return (loss, outputs) if return_outputs else loss
