import torch
import torch.nn as nn


class KnowledgeEmbeddingLayer(nn.Module):
    def __init__(self, sentence_transe: nn.Module):
        super(KnowledgeEmbeddingLayer, self).__init__()
        self.sentence_encoder = sentence_transe.sentence_encoder
        self.linear = sentence_transe.sentence_w
        self.device = sentence_transe.device
    
    def forward(self, x):
        encoded = torch.FloatTensor(self.sentence_encoder.encode(x)).to(self.device)
        y = self.linear(encoded)
        del encoded
        return y