import numpy as np


def train_test_fold_split(data: list, fold: int, val: bool=False):
    """
    split data into training/(development)/test
    :param data: list of np.ndarray, the data to be split
    :param fold: int, the ordinal number of fold
    :param val: if data should be divided to developement set
    :return: split data

    >>> data_list = [np.array([1, 2, 3]), np.array([4, 5, 6]), np.array([7, 8, 9]), np.array([10, 11, 12])]
    >>> train_test_fold_split(data_list, fold=3, val=True)
    (array([ 1,  2,  3, 10, 11, 12]), array([4, 5, 6]), array([7, 8, 9]))
    >>> data_list = [np.array([1, 2, 3]), np.array([4, 5, 6]), np.array([7, 8, 9])]
    >>> train_test_fold_split(data_list, fold=1, val=True)
    (array([4, 5, 6]), array([7, 8, 9]), array([1, 2, 3]))
    >>> data_list = [np.array([1, 2, 3]), np.array([4, 5, 6]), np.array([7, 8, 9]), np.array([10, 11, 12])]
    >>> train_test_fold_split(data_list, fold=3, val=False)
    (array([ 1,  2,  3,  4,  5,  6, 10, 11, 12]), array([7, 8, 9]))
    """
    def fold_split(data_list, test_index):
        if len(data_list) == 1 or len(data_list) <= test_index:
            return None, None
        if len(data_list) == 2:
            return data_list[(3 - test_index)%2], data_list[test_index]

        train_list = []
        test_list = data_list[test_index]
        if test_index == 0:
            train_list = data_list[1:]
        elif test_index == len(data_list)-1:
            train_list = data_list[:-1]
        else:
            train_list.extend(data_list[:test_index])
            train_list = train_list + data_list[test_index+1:]
        return train_list, test_list

    if val and len(data) == 2:
        return None, None, None

    train_data, test_data = fold_split(data, fold-1)
    if val:
        if len(train_data) >= 3:
            train_data, val_data = fold_split(train_data, fold-2)
            train_data = np.concatenate(train_data)
        else:
            train_data, val_data = fold_split(train_data, fold-2)
        return train_data, val_data, test_data
    else:
        return np.concatenate(train_data), test_data