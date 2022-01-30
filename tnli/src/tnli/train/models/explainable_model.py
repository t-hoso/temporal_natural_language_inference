import torch.nn as nn

from .layers.explainable_base import ExplainableBase


class ExplainableBertsModel(nn.Module):
    """
    This Model is Self-Explaining Model with TransE-pre-trained weights.

    Attributes
    ----------
    explainable_model: torch.nn.Module
        the explainable model
    """
    def __init__(self, explain_base, output):
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
        super(ExplainableBertsModel, self).__init__()
        explain_base.model.output = output
        self.model = explain_base.model
    
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
        return self.model(
            input_ids, start_indices, end_indices, span_masks
        )
