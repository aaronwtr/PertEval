import torch
import torch.nn as nn


class WeightedRMSELoss(nn.Module):
    def __init__(self):
        super(WeightedRMSELoss, self).__init__()

    def forward(self, y_pred, y_true):
        abs_pert_effect = torch.abs(y_true)

        weights = nn.functional.softmax(abs_pert_effect, dim=-1)

        return torch.sqrt(torch.mean(weights * (y_true - y_pred) ** 2))
