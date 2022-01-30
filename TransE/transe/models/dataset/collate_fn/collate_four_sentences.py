from typing import Tuple, List

import torch
from Self_Explaining_Structures_Improve_NLP_Models.datasets.collate_functions \
    import collate_to_max_length


def collate_four_sentences(
    batch: Tuple[Tuple[torch.Tensor]], 
    max_len: int=None, 
    fill_values: List[float]=None) -> Tuple[Tuple[torch.Tensor]]:
    """
    wrapper for collate_to_max_length
    To apply this to ExplainDataset
    
    Parameters
    ----------
    batch: Tuple[Tuple[torch.Tensor]]
        one batch from ExplainDataset
        format is
            (correct_example, corrupt_example)
            and each example is:
                (sentence1_input_ids, label, length1, sentence2_input_ids, label, length2)
    max_len: int
        the max length for padding
    fill_values: List[float]
        padding numbers

    Returns
    -------
    Tuple[Tuple[torch.Tensor]]
        each examples that are applied collate_to_max_length
        that is
            (correct_example, corrupt_example)
        each example is:
            (a complete input for ExplainTransE, a complete input for ExplainTransE)
        and each input is:
            (input_ids, labels, length, start_indices, end_indices, span_masks)
    """
    correct_out = [sample[0] for sample in batch]
    corrupt_out = [sample[1] for sample in batch]

    sentence1_correct = [sample[:len(sample)//2] for sample in correct_out]
    sentence2_correct = [sample[len(sample)//2:] for sample in correct_out]
    sentence1_corrupt = [sample[:len(sample)//2] for sample in corrupt_out]
    sentence2_corrupt = [sample[len(sample)//2:] for sample in corrupt_out]

    return (
        collate_to_max_length(sentence1_correct, max_len, fill_values),
        collate_to_max_length(sentence2_correct, max_len, fill_values)
    ), (
        collate_to_max_length(sentence1_corrupt, max_len, fill_values),
        collate_to_max_length(sentence2_corrupt, max_len, fill_values)
    )
