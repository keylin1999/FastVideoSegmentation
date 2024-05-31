import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import defaultdict
    
class CrossEntropyDiceLoss(nn.CrossEntropyLoss):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.counter = 0 
    def forward(self, input, target):
        """Return cross-entropy loss and add auxiliary loss if possible."""
        self.counter += 1
        losses = defaultdict(int)
        losses['total_loss'] = super().forward(input, target)
        losses['ce_loss'] = losses['total_loss']
        prob2 = F.softmax(input, dim=1)

        # compute dice loss
        if self.counter > 150*464/8: # 150 epoch
            mask = prob2[:, 1].flatten(start_dim=1)
            gt = (target==1).float().flatten(start_dim=1)
            numerator = 2 * (mask * gt).sum(-1) # -1 means last axis
            denominator = mask.sum(-1) + gt.sum(-1)
            dice_loss = 1 - (numerator + 1) / (denominator + 1)
            losses['dice_loss'] = dice_loss.mean()
            losses['total_loss'] += losses['dice_loss']
        if torch.isnan(losses['total_loss']):
            print("nan loss")

        losses['total_loss'] /= input.shape[0]
        losses['ce_loss'] /= input.shape[0]
        losses['dice_loss'] /= input.shape[0]

        return losses
    
class LossComputer:
    def __init__(self, config) -> None:
        self.config = config
        self.loss = CrossEntropyDiceLoss()
    def compute(self, data, it):
        return self.loss(data['logit'], data['target'])