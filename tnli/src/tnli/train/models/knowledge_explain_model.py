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


class KnowledgeExplainModel(nn.Module):
    def __init__(self, explain_model, knowledge_output_dim,
                 sentence_transe, encoder_output_dim, output_dim):
        super(KnowledgeExplainModel, self).__init__()
        self.contextual_encoder = explain_model
        self.knowledge_embedding = KnowledgeEmbeddingLayer(sentence_transe)
        self.converter = DimensionConvertLayer(knowledge_output_dim*2, encoder_output_dim)
        self.textual_converter = DimensionConvertLayer(
            explain_model.model.bert_config.hidden_size, encoder_output_dim
        )
        self.combine = CombineLayer(encoder_output_dim*4, encoder_output_dim)
        self.classification_layer = ClassificationLayer(encoder_output_dim, output_dim)
        self.softmax = nn.Softmax(1)
    
    def forward(self, x):
        x1 = x[0]  # input_ids, start_indices, end_indices, span_masks
        x2 = x[1]  # str1, str2
        h1, a_ij = self.contextual_encoder(*x1)
        h1 = self.textual_converter(h1)
        str1_h2 = self.knowledge_embedding(x2[0])
        str2_h2 = self.knowledge_embedding(x2[1])
        h2 = self.converter(torch.cat([str1_h2, str2_h2], dim=1))
        h = self.combine([h1, h2])
        h = self.classification_layer(h)
        y = self.softmax(h)
        return y, a_ij