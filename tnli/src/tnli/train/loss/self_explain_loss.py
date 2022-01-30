import torch.nn as nn
from torch.nn.modules.loss import _Loss


class SelfExplainLoss():
    def __init__(self, loss, lamb):
        self.loss = loss
        self.lamb = lamb
    
    def __call__(self, outputs, labels):
        return self.loss(outputs[0], labels) - \
            (self.lamb * outputs[1].pow(2).sum(dim=1).mean())
