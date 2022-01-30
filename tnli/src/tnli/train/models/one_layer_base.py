import sys

import torch
import torch.nn as nn

sys.path.append(".")
from .one_layer_model import SummationEmbeddingLayer, TanhLayer


class OneLayerBase(nn.Module):
    def __init__(self, embedding_input_dim, embedding_output_dim, output_dim, dropout_rate, glove):
        super(OneLayerBase, self).__init__()
        self.embedding1 = SummationEmbeddingLayer(embedding_input_dim, embedding_output_dim, dropout_rate, glove)
        self.similarity = None
        self.tanh1 = TanhLayer(embedding_output_dim * 2, embedding_output_dim * 2)
        self.tanh2 = TanhLayer(embedding_output_dim * 2, embedding_output_dim * 2)
        self.tanh3 = TanhLayer(embedding_output_dim * 2, output_dim)

    def forward(self, x):
        h1 = self.embedding1(x[0])
        h2 = self.embedding1(x[1])
        h = torch.cat([h1, h2], dim=1)
        h = self.tanh1(h)
        h = self.tanh2(h)
        y = self.tanh3(h)
        return y