import sys

import torch
import torch.nn as nn
from Self_Explaining_Structures_Improve_NLP_Models.explain.model import ExplainableModel

sys.path.append(".")
from .roberta import Roberta


class ExplainableOutput(nn.Module):
    """
    This Model is Self-Explaining Model without output layer.
    Thus, this outputs the sentence embeddings.

    Attributes
    ----------
    model: torch.nn.Module
        the explainable model's output layer
    """
    def __init__(self, bert_dir, num_labels):
        """
        Parameters
        ----------
        bert_dir: str
            pre-trained bert dir
        """
        super(ExplainableOutput, self).__init__()
        model = ExplainableModel(bert_dir, num_labels)
        self.model = model.output
    
    def forward(self, x):
        """
        Parameters
        ----------
        x: torch.FloatTensor
            the output of explainable base model

        Returns
        -------
        torch.FloatTensor
            the prediction
        """
        return self.model(x)