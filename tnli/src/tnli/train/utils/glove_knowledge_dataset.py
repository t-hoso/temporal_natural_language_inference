import sys

import torch

sys.path.append(".")
from .dataset.glove_dataset import load_glove_split_data
from .load_data import load_fold_sentences, train_test_fold_split
from .mode import Mode


class GloveKnowledgeDataset(torch.utils.data.Dataset):
    def __init__(self, mode, fold, name, dim, val=True, data_dir=None):
        super().__init__()
        sentence1_list, sentence2_list = load_fold_sentences()
        if val:
            train_data, val_data, test_data = load_glove_split_data(fold, True, data_dir)
            train_str1, val_str1, test_str1 = train_test_fold_split(sentence1_list, fold-1, True)
            train_str2, val_str2, test_str2 = train_test_fold_split(sentence2_list, fold-1, True)
        else:
            train_data, test_data = load_glove_split_data(fold, True, data_dir)
            train_str1, test_str1 = train_test_fold_split(sentence1_list, fold - 1, False)
            train_str2, test_str2 = train_test_fold_split(sentence2_list, fold - 1, False)
        self.result = None
        if mode == Mode.TRAIN:
            self.result = [train_data[0], train_data[1], train_data[2], train_str1, train_str2]
        elif mode == Mode.TEST:
            self.result = [test_data[0], test_data[1], test_data[2], test_str1, test_str2]
        elif mode == Mode.VALIDATION:
            self.result = [val_data[0], val_data[1], val_data[2], val_str1, val_str2]

    def __getitem__(self, item):
        return torch.LongTensor(self.result[0][item]), torch.LongTensor(self.result[1][item]),\
               torch.LongTensor([self.result[2][item]]), self.result[3][item], self.result[4][item]

    def __len__(self):
        return len(self.result[0])
