import torch
import torch.nn as nn


class SubtractionLayer(nn.Module):
    def __init__(self):
        super(SubtractionLayer, self).__init__()
    
    def forward(self, x):
        return x[1] - x[0] 