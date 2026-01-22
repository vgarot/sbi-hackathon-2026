import torch.nn as nn
import torch

class QuantileLoss(nn.Module):
    def __init__(self,
                 alpha:float=0.5,
                 normalized:bool=False
                 ):
        super().__init__()
        self.alpha = alpha
        self.factor = max(1/alpha, 1/(1-alpha)) if normalized else 1.0

    def forward(self, y_pred, y_true):
        error = y_true - y_pred
        loss = torch.max(self.alpha * error, (self.alpha - 1) * error)
        return torch.mean(loss) * self.factor