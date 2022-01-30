import sys

import torch
import torch.nn as nn

sys.path.append(".")
from .one_layer_base import OneLayerBase
from .layers import(
    KnowledgeEmbeddingLayer,
    DimensionConvertLayer,
    CombineLayer,
    ClassificationLayer
)


class KnowledgeOneLayerModel(nn.Module):
    def __init__(self, embedding_input_dim, embedding_output_dim, dropout_rate, glove,
                 sentence_transe, encoder_output_dim, output_dim):
        super(KnowledgeOneLayerModel, self).__init__()
        self.one_layer_base = OneLayerBase(embedding_input_dim, embedding_output_dim, encoder_output_dim, dropout_rate, glove)
        self.knowledge_embedding = KnowledgeEmbeddingLayer(sentence_transe)
        self.converter = DimensionConvertLayer(256*2, encoder_output_dim)
        self.combine = CombineLayer(encoder_output_dim*4, encoder_output_dim)
        self.classification_layer = ClassificationLayer(encoder_output_dim, output_dim)
        self.softmax = nn.Softmax()
    
    def forward(self, x):
        x1 = x[0]
        x2 = x[1]
        h1 = self.one_layer_base(x1)
        str1_h2 = self.knowledge_embedding(x2[0])
        str2_h2 = self.knowledge_embedding(x2[1])
        h2 = self.converter(torch.cat([str1_h2, str2_h2], dim=1))
        h = self.combine([h1, h2])
        h = self.classification_layer(h)
        y = self.softmax(h)
        return y