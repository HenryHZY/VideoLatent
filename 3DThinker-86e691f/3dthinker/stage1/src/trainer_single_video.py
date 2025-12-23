from trl import SFTTrainer
import torch
import wandb
import numpy as np

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
        
        # predicted_ids = outputs.logits.argmax(dim=-1)
        # decoded_text = self.tokenizer.batch_decode(predicted_ids, skip_special_tokens=False)
        # NOTE-ZY: debug to print raw output
        
        predict_embeddings = outputs.hidden_states
        image_out_mask = inputs["image_out_mask"]
        
        shift_image_mask = image_out_mask[:, -(predict_embeddings.shape[1] - 1) :].to(predict_embeddings.device)
        shift_predict_embeddings = predict_embeddings[..., :-1, :][shift_image_mask.to(predict_embeddings.device) != 0].contiguous()
                
        input_embeddings = outputs.inputs_embeds
        mask = (inputs["input_ids"][0] == 151655).int()
        mask = mask.unsqueeze(0)
        image_embeddings = input_embeddings[mask.to(input_embeddings.device) != 0].contiguous()
        
        image_tokens = image_embeddings.shape[0]
        image_embed_dim = image_embeddings.shape[1]
        image_number = inputs["image_grid_thw"].shape[0]
        patch_size = int(image_tokens/image_number)
        image_embeddings = image_embeddings.view(image_number, patch_size, image_embed_dim).unsqueeze(0)
        
        # TODO-ZY: offline extract video features
        # TODO-ZY: how to prepare video features as supervision? 
            # all frames -> unsupervised clustering
            # all frames -> supervised, pooling? (e.g., grounded videoqa with key frames)
        data = np.load('../../data/offline_features/' + str(idx[0]) + '/features.npz')
        offline_features = data['features']
        offline_features = torch.tensor(offline_features).to(device=shift_predict_embeddings.device, dtype=shift_predict_embeddings.dtype)
        offline_features = offline_features.squeeze()
        
        # TODO-ZY: how to prepare video features as prediction? 
            # shift_predict_embeddings; 
            # shift_predict_embeddings + image_embeddings
            # shift_predict_embeddings + image_embeddings + text_embeddings
        feature_sim = ((shift_predict_embeddings - offline_features.detach()) ** 2).sum(dim=-1)
        sim_loss = feature_sim.mean()*0.0005
        
        loss = 0.1 * ce_loss + sim_loss

        wandb.log({
            "train/ce_loss": ce_loss.item(),
            "train/sim_loss": sim_loss.item(),
            "train/total_loss": loss.item(),
            "train/step": self.state.global_step,
            "train/epoch": self.state.epoch,
        })
        
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
        wandb.log({
            "train/ce_loss": ce_loss.item(),
            "train/total_loss": loss.item(),
            "train/step": self.state.global_step,
            "train/epoch": self.state.epoch,
        })
        print(f"Step {self.state.global_step}: CE Loss: {ce_loss.item():.4f}")
        return (loss, outputs) if return_outputs else loss