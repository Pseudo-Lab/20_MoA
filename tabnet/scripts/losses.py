import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _WeightedLoss


class LabelSmoothing(nn.Module):
    def __init__(self, smoothing=0.1, n_class=206, average='mean'):
        super(LabelSmoothing, self).__init__()
        self.confidence = smoothing
        self.smoothing = smoothing / (n_class - 1)
        self.bceloss = nn.BCEWithLogitsLoss(reduction=average)

    def forward(self, x, target):
        x = x.float()
        target = target.float()
        target_smooth = torch.max(target - self.confidence, torch.ones_like(target) * self.smoothing)
        loss = self.bceloss(x, target_smooth)
        return loss


class SmoothBCEwLogits(_WeightedLoss):
    def __init__(self, weight=None, reduction='mean', smoothing=0.0):
        super().__init__(weight=weight, reduction=reduction)
        self.smoothing = smoothing
        self.weight = weight
        self.reduction = reduction

    @staticmethod
    def _smooth(targets:torch.Tensor, n_labels:int, smoothing=0.0):
        assert 0 <= smoothing < 1
        with torch.no_grad():
            targets = targets * (1.0 - smoothing) + 0.5 * smoothing
        return targets

    def forward(self, inputs, targets):
        targets = SmoothBCEwLogits._smooth(targets, inputs.size(-1),
            self.smoothing)
        loss = F.binary_cross_entropy_with_logits(inputs, targets,self.weight)

        if  self.reduction == 'sum':
            loss = loss.sum()
        elif  self.reduction == 'mean':
            loss = loss.mean()

        return loss