from typing import List

import numpy as np
import torch
from Self_Explaining_Structures_Improve_NLP_Models.datasets.collate_functions import collate_to_max_length



def collate_to_max_length_for_knowledge(batch: List[List[torch.Tensor]], max_len: int = None, fill_values: List[float] = None) -> \
    List[torch.Tensor]:
    """
    """
    # [batch, num_fields]

    batch_related = [sample[:-2] for sample in batch]
    output = collate_to_max_length(batch_related)
    str1s = []
    str2s = []
    for b in batch:
        str1 = b[-2]
        str2 = b[-1]
        str1s.append(str1)
        str2s.append(str2)
    output.append(str1s)
    output.append(str2s)
    return output