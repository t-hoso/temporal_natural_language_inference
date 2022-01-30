import numpy as np
import torch
from .split import train_test_fold_split

class GloveDataset(torch.utils.data.Dataset):
    def __init__(self, mode, fold, val, data_dir):
        super().__init__()
        if val:
            train_data, val_data, test_data = load_glove_split_data(fold, True, data_dir)
        else:
            train_data, test_data = load_glove_split_data(fold, False, data_dir)
        self.result = None
        if mode == 'train':
            self.result = list(train_data)
        elif mode == 'test':
            self.result = list(test_data)
        elif mode == 'validation':
            self.result = list(val_data)

    def __getitem__(self, item):
        return torch.LongTensor(self.result[0][item]), torch.LongTensor(self.result[1][item]),\
               torch.LongTensor([self.result[2][item]])

    def __len__(self):
        return len(self.result[0])


def load_glove_split_data(fold: int, val: bool, data_dir):
    """
    load vectorized data that are pre-saved.
    labels are {0, 1, 2},
    data are processed by glove

    Parameters
    ----------
    fold: int
        the ordinal number of fold, {1, 2, 3, 4, 5}
    val: bool
        if validation is necessary
    name: str
        the glove model name
    dim: str
        the glove model dim
    Returns
    Tuple[Tuple[np.ndarray]]
        (train_data, test_data) or (train_data, validation_data, test_data)
    """
    folds = 5
    label_list = []
    for i in range(1, folds+1):
        filepath = data_dir.parent / 'paraphrase-distilroberta-base-v1' / ('label_' + str(i) + '.csv')
        label_list.append(np.loadtxt(filepath))

    sentence1 = []
    for i in range(1, folds+1):
        filepath = data_dir / ('sentence1_' + str(i) + '.csv')
        sentence1.append(np.loadtxt(filepath, delimiter=','))
    sentence2 = []
    for i in range(1, folds+1):
        filepath = data_dir / ('sentence2_' + str(i) + '.csv')
        sentence2.append(np.loadtxt(filepath, delimiter=','))

    if val:
        train_labels, val_labels, test_labels = train_test_fold_split(label_list, fold - 1, val)
        train_sentence1, val_sentence1, test_sentence1 = train_test_fold_split(sentence1, fold - 1, val)
        train_sentence2, val_sentence2, test_sentence2 = train_test_fold_split(sentence2, fold - 1, val)
        return (train_sentence1, train_sentence2, train_labels),\
               (val_sentence1, val_sentence2, val_labels),\
               (test_sentence1, test_sentence2, test_labels)

    else:
        train_labels, test_labels = train_test_fold_split(label_list, fold-1)
        train_sentence1, test_sentence1 = train_test_fold_split(sentence1, fold - 1, val)
        train_sentence2, test_sentence2 = train_test_fold_split(sentence2, fold - 1, val)
        return (train_sentence1, train_sentence2, train_labels),\
               (test_sentence1, test_sentence2, test_labels)







def load_glove_split_tensor_dataset(fold: int, val: bool, name: str = '840B', dim: int = 300):
    """
    load vectorized dataset that is pre-saved.
    labels are {0, 1, 2},
    data are processed by glove
    :param fold: int, the number of fold to be treated as test set
    :param val: bool, if it is
    :param model_name:
    :return: tuple of torch.utils.data.TensorDataset
    """
    if val:
        train_data, val_data, test_data = load_glove_split_data(fold, val, name, dim)
    else:
        train_data, test_data = load_vectorized_split_data(fold, val, name, dim)

    train_dataset = torch.utils.data.TensorDataset(torch.tensor(train_data[0]).long(),
                                                   torch.tensor(train_data[1]).long(), torch.tensor(train_data[2]).long())
    test_dataset = torch.utils.data.TensorDataset(torch.tensor(test_data[0]).long(),
                                                  torch.tensor(test_data[1]).long(), torch.tensor(test_data[2]).long())

    if val:
        val_dataset = torch.utils.data.TensorDataset(torch.tensor(val_data[0]).long(),
                                                     torch.tensor(val_data[1]).long(), torch.tensor(val_data[2]).long())
        return train_dataset, val_dataset, test_dataset
    else:
        return train_dataset, test_dataset





    @staticmethod
    def load_tnli_glove_dataset_folds(fold, name='840B', dim=300):
        """
        load raw sentence and corresponding dataset
        this datasets are for 5-fold cv
        :param fold:
        :return:
        """
        train_dataset, val_dataset, test_dataset = load_glove_split_tensor_dataset(fold, val=True, name=name, dim=dim)
        sentence1_list, sentence2_list = load_fold_sentences()
        train_sentence1, val_sentence1, test_sentence1 = train_test_fold_split(sentence1_list, fold-1, True)
        train_sentence2, val_sentence2, test_sentence2 = train_test_fold_split(sentence2_list, fold-1, True)
        return train_sentence1, train_sentence2, train_dataset, val_sentence1, val_sentence2, val_dataset,\
            np.array(test_sentence1), np.array(test_sentence2), test_dataset
