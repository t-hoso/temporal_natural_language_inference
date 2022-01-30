import sys

import torch
import torch.nn as nn
from Self_Explaining_Structures_Improve_NLP_Models.explain.model import ExplainableModel

sys.path.append(".")
from .identity import Identity
from .roberta import Roberta


class ExplainableBase(nn.Module):
    """
    This Model is Self-Explaining Model without output layer.
    Thus, this outputs the sentence embeddings.

    Attributes
    ----------
    model: ExplainableModel
        the explainable model
        output layer is changed to indentity layer
    """
    def __init__(self, bert_dir, bert: torch.nn.Module=None):
        """
        Parameters
        ----------
        bert_dir: str
            pre-trained bert dir
        """
        super(ExplainableBase, self).__init__()
        model = ExplainableModel(bert_dir)
        model.intermediate = bert if bert else Roberta(model.intermediate)
        model.output = Identity()
        self.model = model
    
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
            the encoded sentence embedding
        """
        out, _ = self.model(input_ids, start_indices, end_indices, span_masks)
        return out