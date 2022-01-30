import torch
import torch.nn.modules.loss as loss


class NegativeLikelihoodLoss(loss._Loss):
    # log likelihood + l2 reg
    def __init__(self, size_average=None, reduce=None, reduction="mean"):
        super(NegativeLikelihoodLoss, self).__init__(size_average, reduce, reduction)

    def forward(self, positive, negative, target):
        h = torch.logaddexp(torch.ones_like(target), -(target * positive))
        out = torch.mean(h)
        return out