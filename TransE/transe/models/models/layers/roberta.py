import torch.nn as nn

from transformers.models.roberta.modeling_roberta import RobertaModel


class Roberta(nn.Module):
    """
    The wrapper for Roberta of Transformers

    Attributes
    ----------
    model: RobertaModel
        RoBERTa
    """
    def __init__(self, roberta: RobertaModel):
        """
        Parameters
        ----------
        roberta: RobertaModel
        """
        super(Roberta, self).__init__()
        self.model = roberta
    
    def forward(self, x, attention_mask=None):
        """
        x: Any
            input for roberta
        attention_mask
            attention mask for roberta
        """
        outputs = self.model(x, attention_mask=attention_mask)
        return outputs.last_hidden_state, outputs.pooler_output
    
