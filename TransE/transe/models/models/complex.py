import torch
import torch.nn as nn
import numpy as np


class SentenceComplEx(nn.Module):
    """
    The ComplEx model for knowledge bases whose tuples include sentence elements

    Attributes
    ----------
    sentence_encoder: Any
        the encoder for sentence
        see SentenceBertEncoder
    relation_encoder: RelationEncoder
        the encoder for relation
    sentence_embedding_dim: int
        the dimension of output of sentence_encoder
    num_relation: int
        the number of relations
    mapped_embedding_dim: int
        the dimension of final output dimension
    device: torch.device
        the device to be used for calculation
    """
    def __init__(self, sentence_encoder, relation_encoder, sentence_embedding_dim: int,
                 num_relation: int, mapped_embedding_dim: int, device=None):
        """
        Parameters
        ----------
        sentence_encoder: Any
            the encoder for sentence
            see SentenceBertEncoder
        relation_encoder: RelationEncoder
            the encoder for relation
        sentence_embedding_dim: int
            the dimension of output of sentence_encoder
        num_relation: int
            the number of relations
        mapped_embedding_dim: int
            the dimension of final output dimension
        device: torch.device
            the device to be used for calculation
        """
        super(SentenceComplEx, self).__init__()
        self.sentence_encoder = sentence_encoder
        self.relation_encoder = relation_encoder
        self.sentence_w_real = nn.Linear(sentence_embedding_dim, mapped_embedding_dim)
        self.sentence_w_imaginary = nn.Linear(sentence_embedding_dim, mapped_embedding_dim)
        self.sentence_w = ComplexSentenceLayer(self.sentence_w_real, self.sentence_w_imaginary)
        self.relation_embedding_real = nn.Embedding(num_relation, mapped_embedding_dim)
        self.relation_embedding_imaginary = nn.Embedding(num_relation, mapped_embedding_dim)
        self.device = device

    def forward(self, x):
        """
        accepts list which is bigger than 3 (the tuple of knowledge base)

        Parameters
        ----------
        x: List[str]
            the triple of example
            consists of (sentence1, relation, sentence2)

        Returns
        -------
        out_sentence1:G torch.FloatTensor
            the embedding of sentence1
        out_relation: torch.FloatTensor
            the embedding of relation
        out_sentence2: torch.FloatTensor
            the embedding of sentence2
        """
        sentence1 = x[0]
        sentence2 = x[2]
        relation = x[1]
        encoded_relation = torch.LongTensor(self.relation_encoder.encode(relation)).to(self.device)
        h_sentence1 = self._forward_sentence(sentence1, encoded_relation)
        h_sentence2 = self._forward_sentence(sentence2, encoded_relation)
        h_relation = self._forward_relation(relation)
        h1 = torch.sum(h_sentence1[0] * h_sentence2[0] * h_relation[0], dim=1, keepdim=True)
        h2 = torch.sum(h_sentence1[1] * h_sentence2[1] * h_relation[0], dim=1, keepdim=True)
        h3 = torch.sum(h_sentence1[0] * h_sentence2[1] * h_relation[1], dim=1, keepdim=True)
        h4 = torch.sum(h_sentence1[1] * h_sentence2[0] * h_relation[1], dim=1, keepdim=True)
        out = h1 + h2 + h3 - h4
        return out

    def _forward_sentence(self, x, encoded_relation):
        """
        forward function for sentence
        :param x:
        consists of str
        :return: h_real, h_imaginary
        the encoded-then-mapped sentence
        """
        encoded = torch.FloatTensor(self.sentence_encoder.encode(x)).to(self.device)
        out_real = self.sentence_w_real(encoded)
        out_imaginary = self.sentence_w_imaginary(encoded)
        return out_real, out_imaginary

    def _forward_relation(self, x):
        """
        forward function for relation
        :param x:
        consists of str
        :return: out
        the embedding
        """
        encoded = torch.LongTensor(self.relation_encoder.encode(x)).to(self.device)
        out_real = self.relation_embedding_real(encoded)
        out_imaginary = self.relation_embedding_imaginary(encoded)
        return out_real, out_imaginary

    def get_sentence_embedding(self, sentence):
        """
        Gets the learned embedding of sentence
        This is for trained model
        :param sentence: str
        :return: embedding
        """
        return self._forward_sentence(sentence)

    def get_relation_embedding(self, relation):
        """
        Gets the learned embedding of relation
        This is for trained model
        :param relation: str
        :return: embedding
        """
        if relation in self.relation_encoder.get_relations():
            return self._forward_relation(relation)
        return None, None


class ComplexSentenceLayer(nn.Module):
    def __init__(self, real, imaginary):
        super(ComplexSentenceLayer, self).__init__()
        self.real = real
        self.imaginary = imaginary
    
    def forward(self, x):
        h1 = self.real(x)
        h2 = self.imaginary(x)
        out = torch.cat([h1, h2], dim=1)
        return out