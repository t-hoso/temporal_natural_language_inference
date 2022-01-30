import torch
import torch.nn as nn
from Self_Explaining_Structures_Improve_NLP_Models.explain.model import ExplainableModel

from .layers.explainable_base import ExplainableBase
from .layers.identity import Identity


class SelfExplainingTransE(nn.Module):
    """
    Self-Explaining TransE model

    Attributes
    ----------
    explain_base: ExplainableBase
        Self-Explaining Model without output layer
    relation_embedding: nn.Embedding
        Embedding for relation
    """
    def __init__(
        self,
        explain_base: ExplainableBase,
        num_relation: int,
        device=None
    ):
        """
        Parameters
        ----------
        explain_base: ExplainableBase
        num_relation: int
            the number of relations in the dataset
        """
        super(SelfExplainingTransE, self).__init__()
        self.explain_base = explain_base

        explain_base.model.bert_config.hidden_size
        self.relation_embedding = nn.Embedding(
            num_relation, explain_base.model.bert_config.hidden_size
        )
        self.device = device
    
    def forward(self, x):
        """
        accepts list which is bigger than 3 (the tuple of knowledge base)
        
        Parameters
        ----------
        x: Tuple
            batch = (input_for_sentence1, input_for_sentence2)
            input_for_sentence1 is:
                input_ids, labels, lengths, start_indices, end_indices, span_masks
        
        Returns
        -------
        out_sentence1: torch.FloatTensor
            encoded sentence1 
        out_relation: torch.FloatTensor
            encoded relation 
        out_sentence2: torch.FloatTensor
            encoded sentence2
        """
        relation = x[0][1].to(self.device)
        out_sentence1 = self.__forward_sentence(x[0])
        out_sentence2 = self.__forward_sentence(x[1])
        out_relation = self.__forward_relation(relation)
        return out_sentence1, out_relation, out_sentence2

    def __forward_sentence(self, x):
        """
        forward function for sentence

        Parameters
        ----------
        x: torch.LongTensor
            inputs for the model

        Returns
        -------
        torch.FloatTensor
            the encoded sentence
        """
        input_ids, _, _, start_indices, end_indices, span_masks = x
        return self.explain_base(
            input_ids.to(self.device), 
            start_indices.to(self.device), 
            end_indices.to(self.device), 
            span_masks.to(self.device))

    def __forward_relation(self, x):
        """
        forward function for relation
        
        Parameters
        ----------
        x: torch.LongTensor
            the relation labels
        
        Returns
        -------
        torch.FloatTensor
            the embedding
        """
        return self.relation_embedding(x)



class TransEExplainableModel(nn.Module):
    """
    This Model is Self-Explaining Model with TransE-pre-trained weights.

    Attributes
    ----------
    explainable_model: torch.nn.Module
        the explainable model
    """
    def __init__(self, bert_dir, transe_dir, num_labels):
        """
        Parameters
        ----------
        bert_dir: str
            the dir of pre-trained bert
        num_labels: int
            the number of labels
        transe_dir: str
            the dir of Explainable TransE pre-traind directory path
        """
        super(TransEExplainableModel, self).__init__()
        bert_config = ExplainableModel(bert_dir).bert_config
        transe = SelfExplainingTransE(ExplainableBase(bert_dir), 23, None)
        transe.load_state_dict(torch.load(transe_dir))
        explainable_base = transe.explain_base
        explainable_base.model.output = nn.Linear(bert_config.hidden_size, num_labels)
        self.explainable_model = explainable_base.model
    
    def forward(self, input_ids, start_indices, end_indices, span_masks):
        """
        Parameters
        ----------
        input_ids: torch.LongTensor
            the input ids of encoded sentences
        start_indices: torch.LongTensor
            indices of start spans
        end_indices: torch.LongTensor
            indices of end spans
        span_masks: torch.LongTensor
            mask for spans

        Returns
        -------
        torch.FloatTensor
            the prediction
        """
        return self.explainable_model(
            input_ids, start_indices, end_indices, span_masks
        )
