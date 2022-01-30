import torch
import torch.nn as nn


class CombineLayer(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super(CombineLayer, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        x1 = x[0]
        x2 = x[1]
        h = [x1, x2, x1-x2, x1*x2]
        h = torch.cat(h, dim=1)
        y = self.linear(h)
        return y