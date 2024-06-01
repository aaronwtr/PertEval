import torch
import torch.nn as nn


class WeightedRMSELoss(nn.Module):
    def __init__(self):
        super(WeightedRMSELoss, self).__init__()

    def forward(self, y_pred, y_true, x_input):
        x = x_input[:, :x_input.shape[1] // 2]
        pert_effect = torch.sqrt(torch.abs(y_true - x))

        weights = nn.functional.softmax(pert_effect, dim=-1)

        return torch.mean(weights * (y_true - y_pred) ** 2)
