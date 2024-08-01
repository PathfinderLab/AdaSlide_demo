import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

class CompressionAgentModule(pl.LightningModule):
    def __init__(self, model, 
                 len_train_dataloader, len_valid_dataloader,
                 learning_rate=1e-5, reward_lambda_coef=0.5, 
                 dice_threshold=0.6, reward_table=[0, 1]):
        super().__init__()
        self.model = model
        self.len_train_dataloader = len_train_dataloader
        self.len_valid_dataloader = len_valid_dataloader
        self.learning_rate = learning_rate
        self.reward_table = reward_table
        self.reward_lambda_coef = reward_lambda_coef
        self.dice_threshold = dice_threshold
            
    def compute_reward_and_loss(self, logits, dice_score):
        batch_size = dice_score.shape[0]
        device = dice_score.device

        prob_dist = Categorical(logits=logits.to(device))
        action = prob_dist.sample().detach().to(device)
        log_prob = prob_dist.log_prob(action).to(device)
        
        action_ = action.clone().detach().to('cpu').numpy()
        dice_score_ = dice_score.clone().detach().to('cpu').numpy()
        
        rewards = torch.zeros((batch_size, 1))
        
        for i, a in enumerate(action_):
            if a == 1 and dice_score_[i] < self.dice_threshold:
                rewards[i] = -1
            else:
                rewards[i] = self.reward_table[a] - self.reward_lambda_coef * (1 - dice_score_[i])
        
        rewards = rewards.to(device)
        loss = -log_prob * rewards
        
        return loss.mean(), action, rewards.mean()
    
    def step(self, batch):
        hr_image = batch['image']
        dice_score = batch['dice']
        
        logits = self.model(hr_image)
        loss, action, reward = self.compute_reward_and_loss(logits, dice_score)
        
        return loss, action, reward
    
    def training_step(self, batch, batch_idx):
        loss, _, reward = self.step(batch)
        
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_reward', reward, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss, _, reward = self.step(batch)
        
        self.log('valid_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('valid_reward', reward / self.len_valid_dataloader, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        pass
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100*self.len_train_dataloader)
        
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

def define_callbacks(project_name, save_top_k=5):
    callbacks = [
        ModelCheckpoint(monitor='valid_reward', mode='max',
                        save_top_k=save_top_k, dirpath=f'weights/{project_name}', 
                        filename='CompressAgent-{epoch:03d}-{valid_loss:.4f}-{valid_reward:.4f}'),
        ]
    
    return callbacks


class CompressionAgentInference:
    def __init__(self, model, device="cuda:0"):
        self.model = model
        self.device = device
        self._init_model()
    
    def _init_model(self):
        self.model.eval()
        self.model.to(self.device)
    
    def predict(self, x):
        with torch.no_grad():
            logits = self.model(x.to(self.device))

            prob_dist = Categorical(logits=logits)
            _, action = torch.max(prob_dist.probs, dim=1)
            action = action.to("cpu").numpy().tolist()

            return action, prob_dist.probs[:, 1].detach().to("cpu").numpy().tolist()