import numpy as np
import torch
from .split import train_test_fold_split

class ExplainDataset(torch.utils.data.Dataset):
    def __init__(self, mode, fold, val, data_dir):
        super().__init__()
        if val:
            train_data, val_data, test_data = load_explain_split_data(fold, True, data_dir)
        else:
            train_data, test_data = load_explain_split_data(fold, False, data_dir)
        self.result = None
        if mode == 'train':
            self.result = list(train_data)
        elif mode == 'test':
            self.result = list(test_data)
        elif mode == 'validation':
            self.result = list(val_data)

    def __getitem__(self, item):
        return torch.LongTensor(self.result[0][item]), torch.LongTensor([self.result[2][item]]),\
               torch.LongTensor([self.result[1][item]])

    def __len__(self):
        return len(self.result[0])


def load_explain_split_data(fold: int, val: bool, data_dir):
    """
    load vectorized data that are pre-saved.
    labels are {0, 1, 2},
    data are processed for explaining model

    :param fold: the ordinal number of fold, {1, 2, 3, 4, 5}
    :param val: bool, if validation is necessary
    :param model_name: str, the name of model (explaining model)
    :return:
    """
    folds = 5
    label_list = []
    for i in range(1, folds+1):
        filepath = data_dir.parent / 'paraphrase-distilroberta-base-v1' / ('label_' + str(i) + '.csv')
        label_list.append(np.loadtxt(filepath))

    input_ids = []
    for i in range(1, folds+1):
        filepath = data_dir / ('ids_' + str(i) + '.csv')
        with open(filepath, encoding='utf-8', mode='r') as f:
            input_id = []
            line = f.readline()
            while line:
                if line != '\n':
                    input_id.append(list(map(int, line.replace("\n", "").split(','))))
                line = f.readline()
        input_ids.append(np.array(input_id))
    lengths = []
    for i in range(1, folds+1):
        filepath = data_dir / ('length_' + str(i) + '.csv')
        lengths.append(np.loadtxt(filepath, delimiter=','))

    if val:
        train_labels, val_labels, test_labels = train_test_fold_split(label_list, fold - 1, val)
        train_input_ids, val_input_ids, test_input_ids = train_test_fold_split(input_ids, fold - 1, val)
        train_length, val_length, test_length = train_test_fold_split(lengths, fold - 1, val)
        return (train_input_ids, train_length, train_labels), \
               (val_input_ids, val_length, val_labels), \
               (test_input_ids, test_length, test_labels)

    else:
        train_labels, test_labels = train_test_fold_split(label_list, fold-1)
        train_input_ids, test_input_ids = train_test_fold_split(input_ids, fold - 1, val)
        train_length, test_length = train_test_fold_split(lengths, fold - 1, val)
        return (train_input_ids, train_length, train_labels), (test_input_ids, test_length, test_labels)