import torch
from torch.distributions import Categorical

import glob
import numpy as np
from tqdm.auto import tqdm

def evaluate(test_dataloader, model, threshold=0.6):
    model.eval()
    
    actions = []
    dice_score = []

    for batch in tqdm(test_dataloader, total=len(test_dataloader), desc="Evaluation", leave=False):
        logits = model(batch['image'])
        
        prob_dist = Categorical(logits=logits)
        _, action = torch.max(prob_dist.probs, dim=1)

        dice_coef = batch['dice']

        for k, a in enumerate(action):
            actions.append(a)
            
            if dice_coef[k] < 1:
                dice_score.append(dice_coef[k].numpy().tolist())
                
    actions = np.array(actions)
    dice_score = np.array(dice_score)
    overall = round(sum(actions == 1) / len(actions), 4)
    low = round(sum(actions[np.where(dice_score < threshold)] == 1) / len(actions[np.where(dice_score < threshold)]), 4)
    high = round(sum(actions[np.where(dice_score >= threshold)] == 1) / len(actions[np.where(dice_score >= threshold)]), 4)
    
    comp_ratio = {
        "overall": overall,
        "low": low, 
        "high": high
    }
    
    return comp_ratio, actions, dice_score