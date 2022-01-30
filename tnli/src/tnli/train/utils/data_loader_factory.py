from functools import partial
import sys

import torch
import torch.utils.data
from Self_Explaining_Structures_Improve_NLP_Models.datasets.collate_functions import collate_to_max_length

sys.path.append(".")
from .data_factory import DatasetFactory
from .mode import Mode
from .dataset_type import DatasetType
from .collate_for_knowledge import collate_to_max_length_for_knowledge


class DataLoaderFactory:
    @staticmethod
    def create_instance(kind, fold, mode, batch_size, ):
        shuffle = mode == Mode.TRAIN
        if ((kind != DatasetType.EXPLAIN_MNLI or kind != DatasetType.GLOVE_MNLI) 
            and (mode == Mode.MATCHED or mode == Mode.MISMATCHED)): 
            raise ValueError("There is no mismatched or matched for  the dataset")
        if (kind == DatasetType.EXPLAIN_MNLI 
            or kind == DatasetType.EXPLAIN_SNLI
            or kind == DatasetType.EXPLAIN_FOLD
            or kind == DatasetType.EXPLAIN_BERT):
            return torch.utils.data.DataLoader(
                dataset=DatasetFactory.create_instance(kind, fold, mode),
                batch_size=batch_size,
                collate_fn=partial(collate_to_max_length, fill_values=[1, 0, 0]),
                drop_last=False,
                shuffle=shuffle
            )
        elif (kind == DatasetType.EXPLAIN_KNOWLEDGE
            or kind == DatasetType.KNOWLEDGE_EXPLAIN_BERT):
            return torch.utils.data.DataLoader(
                dataset=DatasetFactory.create_instance(kind, fold, mode),
                batch_size=batch_size,
                collate_fn=partial(
                    collate_to_max_length_for_knowledge, 
                    fill_values=[1, 0, 0]),
                drop_last=False,
                shuffle=shuffle
            )
        else:
            return  torch.utils.data.DataLoader(DatasetFactory.create_instance(kind, fold, mode),
                                                batch_size=batch_size, 
                                                shuffle=shuffle)
