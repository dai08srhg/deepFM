import torch
import torch.nn as nn
import torch.nn.functional as F


class BinaryFocalLossWithLogits(nn.Module):
    def __init__(self, gamma=0):
        super(BinaryFocalLossWithLogits, self).__init__()
        self.gamma = gamma

    def forward(self, input, target):
        loss = F.binary_cross_entropy_with_logits(input, target, reduction='none')
        prob = torch.sigmoid(input)
        prob_array = prob.detach().numpy()
        prob_array[target == 0] = (1 - prob_array)[target == 0]

        t_prob = torch.tensor(prob_array)
        focal_loss = torch.pow((1 - t_prob), self.gamma) * loss
        return torch.mean(focal_loss)