import torch
import torch.nn.modules.loss as loss
import torch.nn.functional as F


class NegativeLikelihoodLossNegativeSampling(loss._Loss):
    def __init__(self, size_average=None, reduce=None, reduction="mean"):
        super(NegativeLikelihoodLossNegativeSampling, self).__init__(size_average, reduce, reduction)

    def forward(self, positive, negative, target):
        y_positive = target
        y_negative = - target
        outputs = torch.cat([positive, negative], dim=1)
        y = torch.cat([y_positive, y_negative])
        h = torch.logaddexp(torch.ones_like(outputs), -(y * outputs))
        out = torch.mean(h)
        return out