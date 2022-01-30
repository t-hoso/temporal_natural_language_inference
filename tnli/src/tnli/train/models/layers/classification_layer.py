import torch
import torch.nn as nn


class ClassificationLayer(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super(ClassificationLayer, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        #self.softmax = nn.Softmax(dim=output_dim)

    def forward(self, x):
        h = self.linear(x)
        #y = self.softmax(h)
        return h