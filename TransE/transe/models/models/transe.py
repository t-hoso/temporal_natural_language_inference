import torch
import torch.nn as nn
import numpy as np


class SentenceTransE(nn.Module):
    """
    The TransE model for knowledge bases whose tuples include sentence elements
    """
    def __init__(self, sentence_encoder, relation_encoder, sentence_embedding_dim,
                 num_relation, mapped_embedding_dim, device=None):
        """
        Parameters
        ----------
        sentence_encoder:
            requires to be in the similar form to SentenceBertEncoder
        relation_encoder:
            requires to be in the similar form to RelationEncoder
        sentence_embedding_dim: int
            the dimension of sentence embedding that are acquired by sentence encoder
        num_relation: int
            the number of relation
        mapped_embedding_dim: int
            the dimension of  mapped embeddings
        device: torch.device
        """
        super(SentenceTransE, self).__init__()
        self.sentence_encoder = sentence_encoder
        self.relation_encoder = relation_encoder
        self.sentence_w = nn.Linear(sentence_embedding_dim, mapped_embedding_dim)
        self.relation_embedding = nn.Embedding(num_relation, mapped_embedding_dim)
        self.device = device

    def forward(self, x):
        """
        accepts list which is bigger than 3 (the tuple of knowledge base)
        
        Parameters
        ----------
        x: list
            each element should be str
        
        Returns
        -------
        out_sentence1, out_relation, out_sentence2
            the embeddings of each entities
        """
        sentence1 = x[0]
        sentence2 = x[2]
        relation = x[1]
        out_sentence1 = self._forward_sentence(sentence1)
        out_sentence2 = self._forward_sentence(sentence2)
        out_relation = self._forward_relation(relation)
        return out_sentence1, out_relation, out_sentence2

    def _forward_sentence(self, x):
        """
        forward function for sentence
        
        Parameters
        ----------
        x:
            consists of str
        
        Returns
        -------
        out
            the encoded-then-mapped sentence
        """
        encoded = torch.FloatTensor(self.sentence_encoder.encode(x)).to(self.device)
        out = self.sentence_w(encoded)
        return out

    def _forward_relation(self, x):
        """
        forward function for relation
        
        Parameters
        ----------
        x:
            consists of str
        
        Returns
        -------
        out
            the embedding
        """
        encoded = torch.LongTensor(self.relation_encoder.encode(x)).to(self.device)
        out = self.relation_embedding(encoded)
        return out

    def get_sentence_embedding(self, sentence):
        """
        Gets the learned embedding of sentence
        This is for trained model
        
        Parameters
        ----------
            sentence: str
        
        Returns
        -------
            embedding
        """
        return self._forward_sentence(sentence)

    def get_relation_embedding(self, relation):
        """
        Gets the learned embedding of relation
        This is for trained model
        
        Parameters
        ----------
        relation: str
        
        Returns
        -------
        embedding
        """
        if relation in self.relation_encoder.get_relations():
            return self._forward_relation(relation)
        return None

    def get_similarity(self, head, relation, tail):
        """
        Calculates the similarity
        
        Parameters
        ----------
        head:
        relation:
        tail:
        
        Returns
        -------
        Tuple
        """
        embedding_head = self.get_sentence_embedding(head)
        embedding_relation = self.get_relation_embedding(relation)
        embedding_tail = self.get_sentence_embedding(tail)

        return np.dot(embedding_head+embedding_relation, embedding_tail) / \
               (np.linalg.norm(embedding_head+embedding_relation) * np.linalg.norm(embedding_tail))

