import os
import sys
import random
import copy
from typing import List, Tuple

import torch
from transformers import RobertaTokenizer

sys.path.append(".")
from .dataset import load_atomic_data


ATOMIC_DIRECTORY = os.environ.get("ATOMIC_DIRECTORY")
START_TOKEN = [0]
END_TOKEN = [2]


class ExplainDataset(torch.utils.data.Dataset):
    """
    The dataset for ExplainTransE

    Attributes
    ----------
    transform
        the transform function
    max_length: int
        the maximum length of tokens
    relation_to_number: dict
        mapping relation to number
    data
    nodes: List[Tuple]
        sentence1 nodes and sentence2 nodes
    datanum: int
        the number of data
    tokenizer: transformers.RobertaTokenizer
        tokenizer for each sentence
    """
    def __init__(
        self, 
        mode: str='train', 
        transform=None, 
        relation: List[str]=["isAfter", "isBefore"], 
        bert_path: str="roberta-base", 
        max_length: int = 32
    ):
        """
        Parameters
        ----------
        mode: str
            the mode {"train", "dev", "test"}
        transform: Any
            transform function
        relation: List[str]
            the relations
        bert_path: str
            the path for RobertaTokenizer
        max_length: int
            the maximum length of tokens
        """
        super().__init__()
        self.transform = transform
        self.max_length = max_length
        self.relation_to_number = {relation:i for i, relation in enumerate(relation)}
        self.data = load_atomic_data(os.path.join(ATOMIC_DIRECTORY, (mode + '.tsv')), relation=relation)
        self.nodes = self.__set_nodes()
        self.datanum = len(self.data)
        self.tokenizer = RobertaTokenizer.from_pretrained(bert_path)
    
    def __set_nodes(self):
        """
        returns all entities as list

        Returns
        -------
        nodes: List[List]
            the all nodes
        """
        nodes = set()
        for data in self.data:
            nodes.add(data[0])
            nodes.add(data[2])
        return list(nodes)
    
    def __corrupt_sample(self):
        """
        sample random head or tail
        to replace one of them
        
        Returns
        -------
        str
            chosen node
        """
        return random.choice(self.nodes)

    def __corrupt(self, idx):
        """
        returns corrupted tuple

        Parameters
        ----------
        idx: int
            current target index of data
        
        Returns
        -------
        corrupted_data: Tuple
            the correputed version of target data
        """
        element = self.__corrupt_sample()
        head_or_tail = random.randint(0, 1)
        replaced_idx = head_or_tail if head_or_tail == 0 else 2
        data = self.data[idx]
        corrupted_data = copy.deepcopy(data)
        corrupted_data[replaced_idx] = element
        return corrupted_data
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Parameters
        ----------
        idx: int

        Returns
        -------
        Tuple
            the positive example and negative example
        """
        # TODO: implement transform
        out_data = self.data[idx]
        out_corrupted = self.__corrupt(idx)
        return self.__process(out_data), self.__process(out_corrupted)

    def __process(self, data: Tuple):
        """
        process the data
        convert the sentence into the input ids
        count its lengths
        and convert label to id

        Parameters
        ----------
        data: Tuple
            the example
        
        Returns
        -------
        sentence1_input_ids: torch.LongTensor
        label: torch.LongTensor
        length1: torch.LongTensor
        sentence2_input_ids: torch.LongTensor
        label: torch.LongTensor
        length2: torch.LongTensor
        """
        sentence1, relation, sentence2 = data
        if sentence1.endswith("."):
            sentence1 = sentence1[:-1]
        if sentence2.endswith("."):
            sentence2 = sentence2[:-1]
        sentence1_input_ids = self.tokenizer.encode(sentence1, add_special_tokens=False)
        sentence2_input_ids = self.tokenizer.encode(sentence2, add_special_tokens=False)
        sentence1_input_ids = torch.LongTensor(
            START_TOKEN + sentence1_input_ids + END_TOKEN 
        )
        sentence2_input_ids = torch.LongTensor(
            START_TOKEN + sentence2_input_ids + END_TOKEN
        )
        length1 = torch.LongTensor([len(sentence1_input_ids)])
        length2 = torch.LongTensor([len(sentence2_input_ids)])
        label = torch.LongTensor([self.relation_to_number[relation]])
        return sentence1_input_ids, label, length1, sentence2_input_ids, label, length2