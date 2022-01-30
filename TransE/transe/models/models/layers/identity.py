import torch
import torch.nn as nn


class Identity(nn.Module):
    """
    Identity Layer
    Outputs are same as inputs
    """
    def __init__(self):
        super(Identity, self).__init__()
    
    def forward(self, x):
        """
        Returns the input

        Parameters
        ----------
        x: Any

        Returns
        -------
        x: Any
            same as input
        """
        return x