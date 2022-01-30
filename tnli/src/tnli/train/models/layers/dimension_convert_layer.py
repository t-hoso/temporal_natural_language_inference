import torch
import torch.nn as nn


class DimensionConvertLayer(nn.Module):
    def __init__(self, original_dimension: int, converted_dimension: int):
        super(DimensionConvertLayer, self).__init__()
        self.converter = nn.Linear(original_dimension, converted_dimension)

    def forward(self, x):
        return self.converter(x)