from pathlib import Path

import numpy as np
import torch
import torch.utils.data


def load_vectorized_data_all(model_name: str='paraphrase-distilroberta-base-v1'):
    """
    load vectorized data that are pre-saved.
    labels are {0, 1, 2}
    data are processed by sentence-transformer
    
    Parameters
    ----------
    model_name: str
        the name of model that is corresponding to the sentence-transformer
    
    Returns
    -------
    Tuple[np.ndarray],
        labels and sentences
    """
    folds = 5
    path = get_vectorized_data_path()
    label_list = []
    for i in range(1, folds+1):
        filepath = path / ('label_' + str(i) + '.csv')
        label_list.append(np.loadtxt(filepath))
    labels = np.concatenate(label_list)

    sentence1 = []
    for i in range(1, folds+1):
        filepath = path / ('sentence1_' + str(i) + '.csv')
        sentence1.append(np.loadtxt(filepath, delimiter=','))
    sentence2 = []
    for i in range(1, folds+1):
        filepath = path / ('sentence2_' + str(i) + '.csv')
        sentence2.append(np.loadtxt(filepath, delimiter=','))
    sentence1 = np.concatenate(sentence1)
    sentence2 = np.concatenate(sentence2)

    return labels, np.concatenate([sentence1, sentence2], 1)


def load_sentences():
    """
    Load all sentences
    Returns
    -------
    Tuple[list]
        sentence1 and sentence2
    """
    data_dir = Path(__file__).parent.parent.parent.parent.parent / 'data' / 'folds'
    sentence1 = []
    sentence2 = []
    for i in range(1, 6):
        sent_path = data_dir / ('dataset'+str(i)+'.tsv')
        with open(sent_path, encoding='utf-8', mode='r') as f:
            line = f.readline()
            line = f.readline()
            while line:
                split_line = line.split("\t")
                sentence1.append(split_line[1])
                sentence2.append(split_line[2].replace("\n", ""))
                line = f.readline()
    return sentence1, sentence2


def load_fold_sentences():
    """
    load all sentences in folds

    Returns
    -------
        list_sentence1, list_sentence2
    """
    import os
    data_dir = Path(os.environ.get("DATA_DIR")) / 'folds'
    list_sentence1 = []
    list_sentence2 = []
    for i in range(1, 6):
        sent_path = data_dir / ('dataset'+str(i)+'.tsv')
        sentence1 = []
        sentence2 = []
        with open(sent_path, encoding='utf-8', mode='r') as f:
            line = f.readline()
            line = f.readline()
            while line:
                split_line = line.split("\t")
                sentence1.append(split_line[1])
                sentence2.append(split_line[2].replace("\n", ""))
                line = f.readline()
        list_sentence1.append(sentence1)
        list_sentence2.append(sentence2)

    return list_sentence1, list_sentence2


def load_fold_sentences_and_labels():
    """
    load sentences and corresponding encoded labels
    """
    folds = 5
    path = get_vectorized_data_path()
    label_list = []
    for i in range(1, folds+1):
        filepath = path / ('label_' + str(i) + '.csv')
        label_list.append(np.loadtxt(filepath))
    labels = np.concatenate(label_list)
    list_sentence1, list_sentence2 = load_fold_sentences()
    print(len(labels), len(list_sentence1))
    return labels, list_sentence1, list_sentence2


def load_split_data(fold: int, val: bool):
    sentence1, sentence2 = load_fold_sentences()
    if val:
        t, v, test = load_vectorized_split_data(fold, val)
        train_labels = t[1]
        val_labels = v[1]
        test_labels = test[1]
        train_sentence1, val_sentence1, test_sentence1 = train_test_fold_sentence_split(sentence1, fold - 1, val)
        train_sentence2, val_sentence2, test_sentence2 = train_test_fold_sentence_split(sentence2, fold - 1, val)
        return (train_sentence1, train_sentence2, train_labels),\
               (val_sentence1, val_sentence2, val_labels),\
               (test_sentence1, test_sentence2, test_labels)

    else:
        train_labels, test_labels = train_test_fold_split(label_list, fold-1)
        train_sentence1, test_sentence1 = train_test_fold_split(sentence1, fold - 1, val)
        train_sentence2, test_sentence2 = train_test_fold_split(sentence2, fold - 1, val)
        return (train_sentence1, train_sentence2, train_labels),\
               (test_sentence1, test_sentence2, test_labels)
    

def load_raw_sentence_dataset(fold, val):
    if val:
        train, _val, test = load_split_data(fold, val)
        return RawSentenceDataset(train), RawSentenceDataset(_val), RawSentenceDataset(test)


class RawSentenceDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        super().__init__()
        self.data = data

    def __getitem__(self, item):
        return self.data[0][item], self.data[1][item],\
               torch.LongTensor([self.data[2][item]])

    def __len__(self):
        return len(self.data[0])



def load_glove_data(name='840B', dim=300):
    """
    load glove data from data directory
    Parameters
    ----------
    name: str
    dim: int
    
    Returns
    -------
    labels
    sentence1
    sentence2
    """
    folds = 5
    path = get_vectorized_data_path()
    label_list = []
    for i in range(1, folds+1):
        filepath = path / ('label_' + str(i) + '.csv')
        label_list.append(np.loadtxt(filepath))
    labels = np.concatenate(label_list)

    sentence1 = []
    for i in range(1, folds+1):
        filepath = path / ('sentence1_' + str(i) + '.csv')
        sentence1.append(np.loadtxt(filepath, delimiter=','))
    sentence2 = []
    for i in range(1, folds+1):
        filepath = path / ('sentence2_' + str(i) + '.csv')
        sentence2.append(np.loadtxt(filepath, delimiter=','))
    sentence1 = np.concatenate(sentence1)
    sentence2 = np.concatenate(sentence2)

    return labels, sentence1, sentence2


def load_glove_split_tensor_dataset(fold: int, val: bool, name: str = '840B', dim: int = 300):
    """
    load vectorized dataset that is pre-saved.
    labels are {0, 1, 2},
    data are processed by glove
    Parameters
    ----------
    fold: int
        the number of fold to be treated as test set
    val: bool
        if it is
    model_name: str
    
    Returns
    -------
    Tuple[torch.utils.data.TensorDataset]
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


def load_glove_split_data(fold: int, val: bool, name: str = '840B', dim: int = 300):
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
    -------
    Tuple
        (train_data, test_data) or (train_data, validation_data, test_data)
    """
    folds = 5
    path = get_vectorized_data_path() / "glove"
    label_list = []
    for i in range(1, folds+1):
        filepath = path / ('label_' + str(i) + '.csv')
        label_list.append(np.loadtxt(filepath))

    sentence1 = []
    for i in range(1, folds+1):
        filepath = path / ('sentence1_' + str(i) + '.csv')
        sentence1.append(np.loadtxt(filepath, delimiter=','))
    sentence2 = []
    for i in range(1, folds+1):
        filepath = path / ('sentence2_' + str(i) + '.csv')
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


def load_glove_nli_data(train_dir: str = 'glove_snli_train',):
    """
    load vectorized data that are pre-saved.
    labels are {0, 1, 2}
    data are processed for self-explaining model
    Parameters
    ----------
    train_dir: str
    
    Returns
    -------
    None
    """
    base_path = get_vectorized_data_path()
    train_dir_path = base_path / train_dir

    filepath = get_filename(train_dir_path, 'label')
    train_labels = np.loadtxt(filepath, delimiter=',')

    filepath = get_filename(train_dir_path, 'sentence1')
    with open(filepath, encoding='utf-8', mode='r') as f:
        train_input_ids = []
        line = f.readline()
        while line:
            if line != '\n':
                train_input_ids.append(list(map(int, line.replace("\n", "").split(','))))
            line = f.readline()
    sentence1 = np.array(train_input_ids)

    filepath = get_filename(train_dir_path, 'sentence2')
    with open(filepath, encoding='utf-8', mode='r') as f:
        train_input_ids = []
        line = f.readline()
        while line:
            if line != '\n':
                train_input_ids.append(list(map(int, line.replace("\n", "").split(','))))
            line = f.readline()
    sentence2 = np.array(train_input_ids)

    return (sentence1, sentence2, train_labels)


def load_vectorized_tensor_dataloader(model_name: str='paraphrase-distilroberta-base-v1'):
    """
    load vectorized data that are pre-saved.
    labels are {0, 1, 2},
    data are processed by sentence-transformer
    Parameters
    ----------
    model_name: str
        the name of model that is corresponding to the sentence-transformer
    
    Returns
    -------
    Tuple
    """
    labels, sentences = load_vectorized_data_all(model_name)
    labels = torch.from_numpy(labels)  # not hard copy, it is not necessary to access the original data
    sentences = torch.from_numpy(sentences)
    return labels, sentences


def load_vectorized_split_tensor_dataset(fold: int, val: bool, model_name: str= 'paraphrase-distilroberta-base-v1'):
    """
    load vectorized dataloader are pre-saved.
    labels are {0, 1, 2},
    data are processed by sentence-transformer
    Parameters
    ----------
    fold: int
        the number of fold to be treated as test set
    val: bool
        if it is
    model_name: str
    
    Returns
    -------
    Tuple[torch.utils.data.TensorDataset]
    """
    if val:
        train_data, val_data, test_data = load_vectorized_split_data(fold, val, model_name)
    else:
        train_data, test_data = load_vectorized_split_data(fold, val, model_name)

    train_dataset = torch.utils.data.TensorDataset(torch.tensor(train_data[0]).float(), torch.tensor(train_data[1]).long())
    test_dataset = torch.utils.data.TensorDataset(torch.tensor(test_data[0]).float(), torch.tensor(test_data[1]).long())

    if val:
        val_dataset = torch.utils.data.TensorDataset(torch.tensor(val_data[0]).float(), torch.tensor(val_data[1]).long())
        return train_dataset, val_dataset, test_dataset
    else:
        return train_dataset, test_dataset


def load_vectorized_split_data(fold: int, val: bool, model_name: str= 'paraphrase-distilroberta-base-v1'):
    """
    load vectorized data that are pre-saved.
    labels are {0, 1, 2},
    data are processed by sentence-transformer

    Parameters
    ----------
    fold: int
        the ordinal number of fold, {1, 2, 3, 4, 5}
    val: bool
        if validation is necessary
    model_name: str
        the name of model that is corresponding to the sentence-transformer
    
    Returns
    -------
    Tuple
    """
    folds = 5
    path = get_vectorized_data_path()
    label_list = []
    for i in range(1, folds+1):
        filepath = path / ('label_' + str(i) + '.csv')
        label_list.append(np.loadtxt(filepath))

    sentence1 = []
    for i in range(1, folds+1):
        filepath = path / ('sentence1_' + str(i) + '.csv')
        sentence1.append(np.loadtxt(filepath, delimiter=','))
    sentence2 = []
    for i in range(1, folds+1):
        filepath = path / ('sentence2_' + str(i) + '.csv')
        sentence2.append(np.loadtxt(filepath, delimiter=','))

    if val:
        train_labels, val_labels, test_labels = train_test_fold_split(label_list, fold - 1, val)
        train_sentence1, val_sentence1, test_sentence1 = train_test_fold_split(sentence1, fold - 1, val)
        train_sentence2, val_sentence2, test_sentence2 = train_test_fold_split(sentence2, fold - 1, val)
        train_sentence = np.concatenate([train_sentence1, train_sentence2], 1)
        val_sentence = np.concatenate([val_sentence1, val_sentence2], 1)
        test_sentence = np.concatenate([test_sentence1, test_sentence2], 1)
        return (train_sentence, train_labels), (val_sentence, val_labels), (test_sentence, test_labels)

    else:
        train_labels, test_labels = train_test_fold_split(label_list, fold-1)
        train_sentence1, test_sentence1 = train_test_fold_split(sentence1, fold - 1, val)
        train_sentence2, test_sentence2 = train_test_fold_split(sentence2, fold - 1, val)
        train_sentence = np.concatenate([train_sentence1, train_sentence2], 1)
        test_sentence = np.concatenate([test_sentence1, test_sentence2], 1)
        return (train_sentence, train_labels), (test_sentence, test_labels)


def load_bert_data_all(model_name: str='bert-base'):
    """
    load vectorized data that are pre-saved.
    labels are {0, 1, 2}
    data are processed by sentence-transformer

    Parameters
    ----------
    model_name: str
        the name of model that is corresponding to the sentence-transformer
    
    Returns
    -------
    Tuple[np.ndarray]
        labels and sentences
    """
    folds = 5
    path = get_vectorized_data_path(model_name)
    label_list = []
    for i in range(1, folds+1):
        filepath = path.parent / 'paraphrase-distilroberta-base-v1' / ('label_' + str(i) + '.csv')
        label_list.append(np.loadtxt(filepath))
    labels = np.concatenate(label_list)

    input_ids = []
    for i in range(1, folds+1):
        filepath = path / ('input_ids_' + str(i) + '.csv')
        input_ids.append(np.loadtxt(filepath, delimiter=','))
    masks = []
    for i in range(1, folds+1):
        filepath = path / ('attention_mask_' + str(i) + '.csv')
        masks.append(np.loadtxt(filepath, delimiter=','))

    return labels, input_ids, masks


def load_bert_split_data(fold: int, val: bool, model_name: str = 'bert-base-uncased'):
    """
    load vectorized data that are pre-saved.
    labels are {0, 1, 2},
    data are processed by berttokenizer

    Parameters
    ----------
    fold: int
        the ordinal number of fold, {1, 2, 3, 4, 5}
    val: bool
        if validation is necessary
    model_name: str
        the name of model that is corresponding to berttokenizer
    
    Returns
    Tuple
    """
    folds = 5
    path = get_vectorized_data_path(model_name)
    label_list = []
    for i in range(1, folds+1):
        filepath = path.parent/ 'paraphrase-distilroberta-base-v1' / ('label_' + str(i) + '.csv')
        label_list.append(np.loadtxt(filepath))

    input_ids = []
    for i in range(1, folds+1):
        filepath = path / ('input_ids_' + str(i) + '.csv')
        input_ids.append(np.loadtxt(filepath, delimiter=','))
    masks = []
    for i in range(1, folds+1):
        filepath = path / ('attention_mask_' + str(i) + '.csv')
        masks.append(np.loadtxt(filepath, delimiter=','))

    if val:
        train_labels, val_labels, test_labels = train_test_fold_split(label_list, fold - 1, val)
        train_input_ids, val_input_ids, test_input_ids = train_test_fold_split(input_ids, fold - 1, val)
        train_masks, val_masks, test_masks = train_test_fold_split(masks, fold - 1, val)
        val_sentence = np.concatenate([val_input_ids, val_masks], 1)
        test_sentence = np.concatenate([test_input_ids, test_masks], 1)
        return (train_input_ids, train_masks, train_labels), \
               (val_input_ids, val_masks, val_labels), \
               (test_input_ids, test_masks, test_labels)

    else:
        train_labels, test_labels = train_test_fold_split(label_list, fold-1)
        train_input_ids, test_input_ids = train_test_fold_split(input_ids, fold - 1, val)
        train_masks, test_masks = train_test_fold_split(masks, fold - 1, val)
        return (train_input_ids, train_masks, train_labels), (test_input_ids, test_masks, test_labels)


def load_bart_split_data(fold: int, val: bool, model_name: str = 'facebook/bart-large-mnli'):
    """
    load vectorized data that are pre-saved.
    labels are {0, 1, 2},
    data are processed by barttokenizer

    Parameters
    ----------
    fold: int
        the ordinal number of fold, {1, 2, 3, 4, 5}
    val: bool
        if validation is necessary
    model_name: str
        the name of model that is corresponding to barttokenizer
    
    Returns
    Tuple
    """
    folds = 5
    path = get_vectorized_data_path(model_name)
    label_list = []
    for i in range(1, folds+1):
        filepath = get_vectorized_data_path('paraphrase-distilroberta-base-v1') / ('label_' + str(i) + '.csv')
        label_list.append(np.loadtxt(filepath))

    input_ids = []
    for i in range(1, folds+1):
        filepath = path / ('input_ids_' + str(i) + '.csv')
        input_ids.append(np.loadtxt(filepath, delimiter=','))
    masks = []
    for i in range(1, folds+1):
        filepath = path / ('attention_mask_' + str(i) + '.csv')
        masks.append(np.loadtxt(filepath, delimiter=','))

    if val:
        train_labels, val_labels, test_labels = train_test_fold_split(label_list, fold - 1, val)
        train_input_ids, val_input_ids, test_input_ids = train_test_fold_split(input_ids, fold - 1, val)
        train_masks, val_masks, test_masks = train_test_fold_split(masks, fold - 1, val)
        val_sentence = np.concatenate([val_input_ids, val_masks], 1)
        test_sentence = np.concatenate([test_input_ids, test_masks], 1)
        return (train_input_ids, train_masks, train_labels), \
               (val_input_ids, val_masks, val_labels), \
               (test_input_ids, test_masks, test_labels)

    else:
        train_labels, test_labels = train_test_fold_split(label_list, fold-1)
        train_input_ids, test_input_ids = train_test_fold_split(input_ids, fold - 1, val)
        train_masks, test_masks = train_test_fold_split(masks, fold - 1, val)
        return (train_input_ids, train_masks, train_labels), (test_input_ids, test_masks, test_labels)


def load_bert_split_tensor_dataset(fold: int, val: bool, model_name: str= 'bert-base-uncased'):
    """
    load dataloader that are pre-saved and encoded by bert tokenizer.
    labels are {0, 1, 2},
    data are processed by berttokenizer
    
    Parameters
    ----------
    fold: int
        the number of fold to be treated as test set
    val: bool
        if it is
    model_name: str
    
    Returns
    -------
    Tuple[torch.utils.data.TensorDataset]
    """
    if val:
        train_data, val_data, test_data = load_bert_split_data(fold, val, model_name)
    else:
        train_data, test_data = load_bert_split_data(fold, val, model_name)

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


def load_bart_split_tensor_dataset(fold: int, val: bool, model_name: str= 'facebook/bart-large-mnli'):
    """
    load dataloader that are pre-saved and encoded by bart tokenizer.
    labels are {0, 1, 2},
    data are processed by barttokenizer
    
    Parameters
    ----------
    fold: int
        the number of fold to be treated as test set
    val: bool
        if it is
    model_name: str
    
    Returns
    -------
    Tuple[torch.utils.data.TensorDataset]
    """
    if val:
        train_data, val_data, test_data = load_bart_split_data(fold, val, model_name)
    else:
        train_data, test_data = load_bart_split_data(fold, val, model_name)

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


def load_explain_data_all(model_name: str = 'explain'):
    """
    load vectorized data that are pre-saved.
    labels are {0, 1, 2}
    data are processed for self-explain model

    Parameters
    ----------
    model_name: str
        the name of model that is corresponding to explaining model
    
    Returns
    -------
    Tuple[np.ndarray]
        labels and sentences
    """
    folds = 5
    path = get_vectorized_data_path(model_name)
    label_list = []
    for i in range(1, folds+1):
        filepath = path.parent / 'paraphrase-distilroberta-base-v1' / ('label_' + str(i) + '.csv')
        label_list.append(np.loadtxt(filepath))
    labels = np.concatenate(label_list)

    input_ids = []
    for i in range(1, folds+1):
        filepath = path / ('ids_' + str(i) + '.csv')
        input_ids.append(np.loadtxt(filepath, delimiter=','))
    lengths = []
    for i in range(1, folds+1):
        filepath = path / ('length_' + str(i) + '.csv')
        lengths.append(np.loadtxt(filepath, delimiter=','))

    return labels, input_ids, lengths


def get_filename(dir_path, kind: str = 'id'):
    """
    find the name of file
    
    Parameters
    ----------
    dir_path: pathlib.Path
    kind: str
        {'id', 'length', 'label', attention_mask}
    
    Returns
    target_filename: str
    """
    target_filename = None
    for filename in dir_path.iterdir():
        if kind in str(filename):
            target_filename = filename
            break
    return target_filename


def load_explain_data(model_name: str = 'explain',
                      train_dir: str = 'explain_snli',
                      validation_dir: str = 'explain_snli_validation',
                      test_dir: str = 'explain_snli_test'):
    """
    load vectorized data that are pre-saved.
    labels are {0, 1, 2}
    data are processed for self-explaining model
    
    Parameters
    ----------
    model_name: str
    
    Returns
    -------
    Tuple
    
    Examples
    --------
    >>> load_explain_data()
    """
    base_path = get_vectorized_data_path(model_name)
    base_path = base_path.parent
    train_dir_path = base_path / train_dir
    validation_dir_path = base_path / validation_dir
    test_dir_path = base_path / test_dir

    path = get_vectorized_data_path(model_name)
    filepath = get_filename(train_dir_path, 'label')
    train_labels = np.loadtxt(filepath, delimiter=',')
    filepath = get_filename(test_dir_path, 'label')
    test_labels = np.loadtxt(filepath, delimiter=',')
    filepath = get_filename(validation_dir_path, 'label')
    validation_labels = np.loadtxt(filepath, delimiter=',')

    filepath = get_filename(train_dir_path, 'ids')
    with open(filepath, encoding='utf-8', mode='r') as f:
        train_input_ids = []
        line = f.readline()
        while line:
            if line != '\n':
                train_input_ids.append(list(map(int, line.replace("\n", "").split(','))))
            line = f.readline()
    train_input_ids = np.array(train_input_ids)
    filepath = get_filename(test_dir_path, 'id')
    with open(filepath, encoding='utf-8', mode='r') as f:
        test_input_ids = []
        line = f.readline()
        while line:
            if line != '\n':
                test_input_ids.append(list(map(int, line.replace("\n", "").split(','))))
            line = f.readline()
    filepath = get_filename(validation_dir_path, 'ids')
    with open(filepath, encoding='utf-8', mode='r') as f:
        validation_input_ids = []
        line = f.readline()
        while line:
            if line != '\n':
                validation_input_ids.append(list(map(int, line.replace("\n", "").split(','))))
            line = f.readline()
    validation_input_ids = np.array(validation_input_ids)
    filepath = get_filename(train_dir_path, 'length')
    train_lengths = np.loadtxt(filepath, delimiter=',')
    filepath = get_filename(test_dir_path, 'length')
    test_lengths = np.loadtxt(filepath, delimiter=',')
    filepath = get_filename(validation_dir_path, 'length')
    validation_lengths = np.loadtxt(filepath, delimiter=',')

    return (train_input_ids, train_lengths, train_labels), \
           (validation_input_ids, validation_lengths, validation_labels), \
           (test_input_ids, test_lengths, test_labels)



def load_explain_snli_data_all(model_name: str = 'explain'):
    """
    load vectorized data that are pre-saved.
    labels are {0, 1, 2}
    data are processed for self-explain model

    Parameters
    ----------
    model_name: str
        the name of model that is corresponding to explaining model
    
    Returns
    -------
    Tuple[np.ndarray]
        labels and sentences

    Examples
    --------
     >>> i = load_explain_snli_data_all()[1][0][0]
     >>> from transformers import RobertaTokenizer
     >>> tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
     >>> tokenizer.convert_ids_to_tokens(i)
    """
    path = get_vectorized_data_path(model_name)
    filepath = path / 'explain_snli_train' / ('labels_' + str(550152) + '.csv')
    train_labels = np.loadtxt(filepath, delimiter=',')
    filepath = path / 'explain_snli_test' / ('labels_' + str(10000) + '.csv')
    test_labels = np.loadtxt(filepath, delimiter=',')
    filepath = path / 'explain_snli_validation' / ('labels_' + str(10000) + '.csv')
    validation_labels = np.loadtxt(filepath, delimiter=',')

    filepath = path.parent / 'explain_snli_train' / ('ids_' + str(550152) + '.csv')
    with open(filepath, encoding='utf-8', mode='r') as f:
        train_input_ids = []
        line = f.readline()
        while line:
            if line != '\n':
                train_input_ids.append(list(map(int, line.replace("\n", "").split(','))))
            line = f.readline()
    train_input_ids = np.array(train_input_ids)
    filepath = path.parent / 'explain_snli_test' / ('ids_10000.csv')
    with open(filepath, encoding='utf-8', mode='r') as f:
        test_input_ids = []
        line = f.readline()
        while line:
            if line != '\n':
                test_input_ids.append(list(map(int, line.replace("\n", "").split(','))))
            line = f.readline()
    filepath = path.parent / 'explain_snli_dev' / ('ids_10000.csv')
    with open(filepath, encoding='utf-8', mode='r') as f:
        validation_input_ids = []
        line = f.readline()
        while line:
            if line != '\n':
                validation_input_ids.append(list(map(int, line.replace("\n", "").split(','))))
            line = f.readline()
    validation_input_ids = np.array(validation_input_ids)
    filepath = path.parent / 'explain_snli_train' / ('length_' + str(550152) + '.csv')
    train_lengths = np.loadtxt(filepath, delimiter=',')
    filepath = path.parent / 'explain_snli_test' / ('length_10000.csv')
    test_lengths = np.loadtxt(filepath, delimiter=',')
    filepath = path.parent / 'explain_snli_dev' / ('length_10000.csv')
    validation_lengths = np.loadtxt(filepath, delimiter=',')

    return (train_input_ids, train_lengths, train_labels), \
           (validation_input_ids, validation_lengths, validation_labels), \
           (test_input_ids, test_lengths, test_labels)


def load_explain_nli_dataset(dir_name: str):
    return ExplainNLIDataset(dir_name)


class ExplainNLIDataset(torch.utils.data.Dataset):
    def __init__(self, dir_name: str):
        super().__init__()
        train_data, validation_data, test_data = load_explain_data(
            model_name='explain',
            train_dir=dir_name,
            validation_dir=dir_name,
            test_dir=dir_name)
        self.result = train_data

    def __getitem__(self, item):
        return torch.LongTensor(self.result[0][item]), torch.LongTensor([self.result[2][item]]),\
               torch.LongTensor([self.result[1][item]])

    def __len__(self):
        return len(self.result[0])



class ExplainSNLIDataset(torch.utils.data.Dataset):
    def __init__(self, mode):
        super().__init__()
        train_data, validation_data, test_data = load_explain_snli_data_all('explain')
        self.result = None
        if mode == 'train':
            self.result = list(train_data)
        elif mode == 'test':
            self.result = list(test_data)
        elif mode == 'validation':
            self.result = list(validation_data)
        else:
            self.result = []

    def __getitem__(self, item):
        return torch.LongTensor(self.result[0][item]), torch.LongTensor([self.result[2][item]]),\
               torch.LongTensor([self.result[1][item]])

    def __len__(self):
        return len(self.result[0])


def load_explain_snli_tensor_dataset(model_name: str= 'explain'):
    """
    load dataset that are pre-saved and encoded by roberta tokenizer.
    labels are {0, 1, 2},
    data are processed by roberta tokenizer
    
    Parameters
    ----------
    model_name: str

    Returns
    Tuple[ExplainSNLIDataset]
    """
    train_dataset = ExplainSNLIDataset('train')
    test_dataset = ExplainSNLIDataset('test')
    val_dataset = ExplainSNLIDataset('validation')
    return train_dataset, val_dataset, test_dataset



class ExplainDataset(torch.utils.data.Dataset):
    def __init__(self, mode, fold, val):
        super().__init__()
        if val:
            train_data, val_data, test_data = load_explain_split_data(fold, True, 'explain')
        else:
            train_data, test_data = load_explain_split_data(fold, False, 'explain')
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


def load_explain_split_data(fold: int, val: bool, model_name: str = 'explain'):
    """
    load vectorized data that are pre-saved.
    labels are {0, 1, 2},
    data are processed for explaining model

    
    Parameters
    ----------
    fold: int
        the ordinal number of fold, {1, 2, 3, 4, 5}
    val: bool
        if validation is necessary
    model_name: str
        the name of model (explaining model)
    
    Returns
    -------
    Tuple
    """
    folds = 5
    path = get_vectorized_data_path(model_name)
    label_list = []
    for i in range(1, folds+1):
        filepath = path.parent / 'paraphrase-distilroberta-base-v1' / ('label_' + str(i) + '.csv')
        label_list.append(np.loadtxt(filepath))

    input_ids = []
    for i in range(1, folds+1):
        filepath = path / ('ids_' + str(i) + '.csv')
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
        filepath = path / ('length_' + str(i) + '.csv')
        lengths.append(np.loadtxt(filepath, delimiter=','))

    if val:
        train_labels, val_labels, test_labels = train_test_fold_split(label_list, fold - 1, val)
        train_input_ids, val_input_ids, test_input_ids = train_test_fold_split(input_ids, fold - 1, val)
        train_length, val_length, test_length = train_test_fold_split(lengths, fold - 1, val)
        #val_sentence = np.concatenate([val_input_ids, val_length], 1)
        #test_sentence = np.concatenate([test_input_ids, test_length], 1)
        return (train_input_ids, train_length, train_labels), \
               (val_input_ids, val_length, val_labels), \
               (test_input_ids, test_length, test_labels)

    else:
        train_labels, test_labels = train_test_fold_split(label_list, fold-1)
        train_input_ids, test_input_ids = train_test_fold_split(input_ids, fold - 1, val)
        train_length, test_length = train_test_fold_split(lengths, fold - 1, val)
        return (train_input_ids, train_length, train_labels), (test_input_ids, test_length, test_labels)


def load_explain_split_tensor_dataset(fold: int, val: bool, model_name: str= 'explain'):
    """
    load dataloader that are pre-saved and encoded by bert tokenizer.
    labels are {0, 1, 2},
    data are processed by berttokenizer
    
    Parameters
    ----------
    fold: int
        the number of fold to be treated as test set
    val: bool
        if it is
    model_name:
    
    Returns
    -------
    Tuple[torch.utils.data.TensorDataset]
    """
    train_dataset = ExplainDataset('train', fold, val)
    test_dataset = ExplainDataset('test', fold, val)
    if val:
        val_dataset = ExplainDataset('validation', fold, val)
        return train_dataset, val_dataset, test_dataset
    else:
        return train_dataset, test_dataset
    if val:
        train_data, val_data, test_data = load_explain_split_data(fold, val, model_name)
    else:
        train_data, test_data = load_explain_split_data(fold, val, model_name)

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


def train_test_fold_split(data: list, fold: int, val: bool=False):
    """
    split data into training/(development)/test
    
    Parameters
    ----------
    data: List[np.ndarray]
        the data to be split
    fold: int
        the ordinal number of fold
    val: bool
        if data should be divided to developement set
    
    Returns
    -------
    split data

    Examples
    --------
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


def train_test_fold_sentence_split(data, fold, val):
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
            train_list.extend(data_list[test_index+1:]) # train_list = train_list + data_list[test_index+1:]
        return train_list, test_list

    if val and len(data) == 2:
        return None, None, None

    train_data, test_data = fold_split(data, fold-1)
    if val:
        if len(train_data) >= 3:
            train_data, val_data = fold_split(train_data, fold-2)
        else:
            train_data, val_data = fold_split(train_data, fold-2)
        train_list = []
        for d in train_data:
            train_list.extend(d)
        train_data = train_list
        return train_data, val_data, test_data
    else:
        return train_data, test_data


def get_split_data_path():
    """
    return the pathlib.Path of split data path
    which is TemporalAware/data/folds
    
    Returns
    -------
    pathlib.Path
        the path of split data
    """
    file_directory = Path(__file__).parent
    data_directory = file_directory.parent.parent.parent.parent / 'data' / 'folds'
    return data_directory

def get_vectorized_data_path(model_name: str='paraphrase-distilroberta-base-v1'):
    """
    return the pathlib.Path of split data preprocessed by sentence-transformers
    
    Parameters
    ----------
    model_name: str
        model name that is used by sentence-transformers
    
    Returns
    -------
    pathlib.Path
        the path of split preprocessed data
    """
    import os
    os.environ.get("DATA_DIR")
    data_directory = Path(os.environ.get("DATA_DIR"))
    return data_directory/model_name

def test():
    pass

if __name__ == '__main__':
    train_data, test_data = load_bert_split_data(fold=1, val=False, model_name='bert-base-uncased')
    sent_id, attn, label = test_data
    for s, a, l in zip(sent_id, attn, label):
        print(s,a,l)