import torch.nn as nn
import torch
class VariancePenaltyLoss(nn.Module):
    def __init__(self, alpha):
        super().__init__()
        self.alpha = alpha

    def forward(self, y_pred, y_true):
        mse = nn.functional.mse_loss(y_pred, y_true)
        var_true = y_true.var(dim=0)
        var_pred = y_pred.var(dim=0)

        variance_penalty = torch.mean(torch.relu(var_true - var_pred))
        
        return mse + self.alpha * variance_penalty