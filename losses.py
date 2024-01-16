import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.autograd import Variable

def compute_dice_score(pred, target, smooth=0.0):

    intersection = (pred * target).sum(dim=(2,3,4))
    union = pred.sum(dim=(2,3,4)) + target.sum(dim=(2,3,4))
    dice = (2.0 * intersection + smooth) / (union + smooth)
    
    return dice.mean()


class DiceLoss(nn.Module):
    def __init__(self, reduction='sum'):
        super(DiceLoss, self).__init__()

        self.reduction = reduction

    def forward(self, input, target):
        
        dice_score = compute_dice_score(input, target)
        dice_loss = Variable((1.0-dice_score), requires_grad=True)
        
        if self.reduction == 'mean':
            return dice_loss.mean()
        
        elif self.reduction == 'sum':
            return dice_loss.sum()
        
        else:
            return dice_loss
