import os
import random
import copy
import csv
import pathlib
from collections import defaultdict

import numpy as np
import torch

ATOMIC_DIRECTORY = os.environ.get("ATOMIC_DIRECTORY")
_LABEL_PREFIX = "_label.tsv"
_SEED_SENTENCE_PREFIX = "_seed_sentence.tsv"
_SENTENCE_PREFIX = '_sentence.tsv'


class Atomic2020Dataset(torch.utils.data.Dataset):
    """
    Atomic2020Dataset class

    This class is dataset of Atomic2020
    The raw Atomic data are consists of tuples (head_entity, relation, tail_entity).
    This class holds these data and returns (head_entity, relation, tail_entity, head_entity', relation, tail_entity').
    """
    def __init__(self, mode='train', transform=None, relation=["isAfter", "isBefore"]):
        """
        Read dataset from specified directory
        
        Parameters
        ----------
        mode: str
            supposed to be {train, test, dev}.
        transform: transform
            the transform object of pytorch
        """
        self.transform = transform
        self.data = load_atomic_data(os.path.join(ATOMIC_DIRECTORY, (mode + '.tsv')), relation=relation)
        self.nodes = self._set_nodes()
        self.datanum = len(self.data)

    def __len__(self):
        return self.datanum

    def _set_nodes(self):
        """
        returns all entities as list
        
        Returns
        -------
        nodes: list
            the all nodes
        """
        nodes = set()
        for data in self.data:
            nodes.add(data[0])
            nodes.add(data[2])
        return list(nodes)

    def _corrupt_sample(self):
        """
        sample random head or tail
        
        Returns
        -------

        """
        return random.choice(self.nodes)

    def _corrupt(self, idx):
        """
        returns corrupted tuple
        
        Parameters
        ----------
        idx:

        Returns
        -------
        corrupted_data
        """
        element = self._corrupt_sample()
        head_or_tail = random.randint(0, 1)
        replaced_idx = head_or_tail if head_or_tail == 0 else 2
        data = self.data[idx]
        corrupted_data = copy.deepcopy(data)
        corrupted_data[replaced_idx] = element
        return corrupted_data

    def __getitem__(self, idx):
        """
        Returns data
        This also includes "corrupted" data
        Paramters
        ---------
        idx: int
            the id of data
        
        Returns
        -------
        data
            (head_sentence, relation, tail_sentence, head_sentence', relation, tail_sentence')
        """
        out_data = self.data[idx]
        out_corrupted = self._corrupt(idx)

        if self.transform:
            out_data = self.transform(out_data)
            out_corrupted = self.transform(out_corrupted)

        return (out_data[0], out_data[1], out_data[2]), \
            (out_corrupted[0], out_corrupted[1], out_corrupted[2])


def load_atomic_data(filepath, relation=["isAfter", "isBefore"]):
    """
    This function loads tsv file.
    Is supposed to load atomic2020 dataset
    
    Parameters
    ----------
    filepath: pathlib.Path
        the path of the input file
    
    Returns
    -------
    data: list
        the data that are read
        This is in the form of list of list
    """
    with open(filepath, mode='r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        data = [d for d in reader]
    if len(relation) != 0:
        data_tmp = [d for d in data if d[1] in relation]
        data = data_tmp
    return data

