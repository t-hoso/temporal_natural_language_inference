import sys
from pathlib import Path

import torch
import numpy as np


def load_glove_nli_dataset(dir_name: Path):
    dataset = load_glove_nli_data(dir_name)
    return \
        torch.utils.data.TensorDataset(torch.LongTensor(dataset[0]), \
            torch.LongTensor(dataset[1]), \
                torch.LongTensor(dataset[2]))


def load_glove_nli_data(dir_name: Path):
    """
    load vectorized data that are pre-saved.
    labels are {0, 1, 2}
    data are processed for self-explaining model
    :param model_name: str
    :return:
    """
    filepath = dir_name / 'labels.csv'
    train_labels = np.loadtxt(filepath, delimiter=',')

    filepath = dir_name /  'sentence1.csv'
    with open(filepath, encoding='utf-8', mode='r') as f:
        train_input_ids = []
        line = f.readline()
        while line:
            if line != '\n':
                train_input_ids.append(list(map(int, line.replace("\n", "").split(','))))
            line = f.readline()
    sentence1 = np.array(train_input_ids)

    filepath = dir_name / 'sentence2.csv'
    with open(filepath, encoding='utf-8', mode='r') as f:
        train_input_ids = []
        line = f.readline()
        while line:
            if line != '\n':
                train_input_ids.append(list(map(int, line.replace("\n", "").split(','))))
            line = f.readline()
    sentence2 = np.array(train_input_ids)

    return (sentence1, sentence2, train_labels)