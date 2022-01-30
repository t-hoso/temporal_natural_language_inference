import sys

import torch
import torch.nn as nn

sys.path.append(".")
from .layers import (
    KnowledgeEmbeddingLayer,
    DimensionConvertLayer,
    CombineLayer,
    ClassificationLayer
)


class KnowledgeModelBase(nn.Module):
    def __init__(self, contextual_encoder, contextual_output_dim, knowledge_output_dim,
                 sentence_transe, encoder_output_dim, output_dim):
        super(KnowledgeModelBase, self).__init__()
        self.contextual_encoder = contextual_encoder
        self.convert_contextual = DimensionConvertLayer(contextual_output_dim, encoder_output_dim)
        self.knowledge_embedding = KnowledgeEmbeddingLayer(sentence_transe)
        self.converter = DimensionConvertLayer(knowledge_output_dim*2, encoder_output_dim)
        self.combine = CombineLayer(encoder_output_dim*4, encoder_output_dim)
        self.classification_layer = ClassificationLayer(encoder_output_dim, output_dim)
        self.softmax = nn.Softmax()
    
    def forward(self, x):
        x1 = x[0]
        x2 = x[1]
        h1 = self.base_model(x1)
        h1 = self.convert_contextual(h1)
        str1_h2 = self.knowledge_embedding(x2[0])
        str2_h2 = self.knowledge_embedding(x2[1])
        h2 = self.converter(torch.cat([str1_h2, str2_h2], dim=1))
        h = self.combine([h1, h2])
        h = self.classification_layer(h)
        y = self.softmax(h)
        return y