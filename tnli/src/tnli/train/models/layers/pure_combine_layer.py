import torch
import torch.nn as nn


class PureCombineLayer(nn.Module):
    def __init__(self):
        super(PureCombineLayer, self).__init__()
    
    def forward(self, x):
        x1 = x[0]
        x2 = x[1]
        h = [x1, x2, x1-x2, x1*x2]
        y = torch.cat(h, dim=1)
        return y