import torch
import torch.nn as nn

import sys
sys.path.append(".")
from .layers import (
    KnowledgeEmbeddingLayer,
    DimensionConvertLayer,
    ClassificationLayer
)


class KnowledgeOnlyModel(nn.Module):
    def __init__(self, sentence_transe: nn.Module, encoder_output_dim, output_dim):
        super(KnowledgeOnlyModel, self).__init__()
        self.knowledge_embedding = KnowledgeEmbeddingLayer(sentence_transe)
        self.converter = DimensionConvertLayer(256*2, encoder_output_dim)
        self.linear = nn.Linear(encoder_output_dim, encoder_output_dim)
        self.classification_layer = ClassificationLayer(encoder_output_dim, output_dim)
        self.softmax = nn.Softmax(1)
    
    def forward(self, x):
        str1_h1 = self.knowledge_embedding(x[0])
        str2_h1 = self.knowledge_embedding(x[1])
        h2 = self.converter(torch.cat([str1_h1, str2_h1], dim=1))
        h = self.linear(h2)
        h = self.classification_layer(h)
        y = self.softmax(h)
        return y