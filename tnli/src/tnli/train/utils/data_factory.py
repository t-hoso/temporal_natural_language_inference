from pathlib import Path
import os

import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.utils.data

from .mode import Mode
from .dataset_type import DatasetType
from .load_data import(
    load_vectorized_data_all, 
    load_sentences, 
    load_vectorized_split_tensor_dataset,
    load_fold_sentences, 
    load_bert_split_tensor_dataset, 
    load_explain_split_tensor_dataset,
    load_explain_snli_tensor_dataset, 
    load_raw_sentence_dataset,
    load_bart_split_tensor_dataset
)
from .explain_knowledge_dataset import ExplainKnowledgeDataset
from .glove_knowledge_dataset import GloveKnowledgeDataset
from .dataset import ExplainDataset
from .dataset import (
    KnowledgeDataset,
    ExplainNLIDataset,
    GloveDataset,
    load_glove_nli_dataset
)
from .dataset.split import train_test_fold_split


DATA_DIR = os.environ.get("DATA_DIR")
BERT = "explain_bert"


class DatasetFactory():
    """
    Dataset Factory
    This creates dataset of any dataset
    """
    def __init__(self):
        self.data_dir = DATA_DIR

    @classmethod
    def create_instance(cls, kind:DatasetType, fold:int, mode:Mode):
        if kind == DatasetType.SBERT_RANDOM:
            _, _, train_dataset, _, _, val_dataset,\
            _, _, test_dataset = cls.load_tnli_sbert_dataset_random()
        elif kind == DatasetType.SBERT_FOLD:
            _, _, train_dataset, _, _, val_dataset,\
            _, _, test_dataset = cls.load_tnli_dataset_folds(fold)
        elif kind == DatasetType.BERT_FOLD:
            _, _, train_dataset, _, _, val_dataset,\
            _, _, test_dataset = cls.load_tnli_bert_dataset_folds(fold)
        elif kind == DatasetType.EXPLAIN_FOLD:
            _, _, train_dataset, _, _, val_dataset,\
            _, _, test_dataset = cls.load_tnli_explain_dataset_folds(fold)
        elif kind == DatasetType.EXPLAIN_SNLI:
            if mode == Mode.TRAIN:
                return cls.load_nli_explain_dataset(
                    Path(DATA_DIR) / "explain_snli_train"
                )
            elif mode == Mode.VALIDATION:
                return cls.load_nli_explain_dataset(
                    Path(DATA_DIR) / "explain_snli_dev"
                )
            elif mode == Mode.TEST:
                return cls.load_nli_explain_dataset(
                    Path(DATA_DIR) / "explain_snli_test"
                )
            # _, _, train_dataset, _, _, val_dataset,\
            # _, _, test_dataset = cls.load_nli_explain_dataset(
            #     Path(DATA_DIR) / "explain_snli_train"
            # )
        elif kind ==  DatasetType.EXPLAIN_MNLI:
            if mode == Mode.TRAIN:
                return cls.load_nli_explain_dataset(
                    Path(DATA_DIR) / "explain_mnli_train"
                )
            elif mode == Mode.MATCHED:
                return cls.load_nli_explain_dataset(
                    Path(DATA_DIR) / "explain_mnli_dev_matched"
                )
            elif mode == Mode.MISMATCHED:
                return cls.load_nli_explain_dataset(
                    Path(DATA_DIR) / "explain_mnli_dev_mismatched"
                )
        elif kind == DatasetType.GLOVE_FOLD:
            train_dataset, val_dataset, test_dataset = \
                cls.load_tnli_glove_dataset_folds(fold)
        elif kind == DatasetType.GLOVE_SNLI:
            if mode == Mode.TRAIN:
                return cls.load_glove_nli_dataset(
                    Path(DATA_DIR) / "glove_snli_train"
                )
            if mode == Mode.VALIDATION:
                return cls.load_glove_nli_dataset(
                    Path(DATA_DIR) / "glove_snli_dev"
                )
            if mode == Mode.TEST:
                return cls.load_glove_nli_dataset(
                    Path(DATA_DIR) / "glove_snli_test"
                )
                # return cls.load_glove_nli_dataset('glove_snli_test')
        elif kind == DatasetType.GLOVE_MNLI:
            if mode == Mode.TRAIN:
                return cls.load_glove_nli_dataset(
                    Path(DATA_DIR) / "glove_mnli_train"
                )
            if mode == Mode.MATCHED:
                return cls.load_glove_nli_dataset(
                    Path(DATA_DIR) / "glove_mnli_dev_matched"
                )
            if mode == Mode.MISMATCHED:
                return cls.load_glove_nli_dataset(
                    Path(DATA_DIR) / "glove_mnli_dev_mismatched"
                )
        elif kind == DatasetType.RAW_SENTENCE:
            train_dataset, val_dataset, test_dataset = cls.load_raw_data(fold)
        elif kind == DatasetType.EXPLAIN_KNOWLEDGE:
            train_dataset, val_dataset, test_dataset = \
                cls.load_explain_knowledge_dataset(fold)
        elif kind == DatasetType.GLOVE_KNOWLEDGE:
            train_dataset, val_dataset, test_dataset = \
                cls.load_glove_knowledge_dataset(fold, name="6B", dim=300)
        elif kind == DatasetType.EXPLAIN_BERT:
            train_dataset, val_dataset, test_dataset = \
                cls.load_explain_dataset(fold, True, Path(DATA_DIR)/BERT)
        elif kind == DatasetType.KNOWLEDGE_EXPLAIN_BERT:
            train_dataset, val_dataset, test_dataset = \
                cls.load_knowledge_dataset(fold, True, Path(DATA_DIR)/BERT)
        elif kind == DatasetType.BART_MNLI:
            _, _, train_dataset, _, _, val_dataset,\
            _, _, test_dataset = cls.load_tnli_bart_mnli_dataset_folds(fold)

        if mode == Mode.TRAIN:
            return train_dataset
        elif mode == Mode.TEST:
            return test_dataset
        elif mode == Mode.VALIDATION:
            return val_dataset         

    @staticmethod
    def load_tnli_sbert_dataset_random():
        """
        load raw sentence and corresponding dataset(labels, sentence vectors)
        this datasets are for hold-out
        
        Examples
        --------
        >>> factory = DatasetFactory()
        >>> train_sentence1, _, _, _, _, _, _, _, _ = factory.load_tnli_sbert_dataset_random()
        >>> train_sentence1_2, _, _, _, _, _, _, _, _ = factory.load_tnli_sbert_dataset_random()
        >>> (train_sentence1 == train_sentence1_2).sum() != len(train_sentence1)
        True
        """
        labels, sentence_vectors = load_vectorized_data_all()
        sentence1, sentence2 = load_sentences()
        train_labels, test_labels, train_sentences, test_sentences,\
            train_sentence1, test_sentence1, train_sentence2, test_sentence2 = \
            train_test_split_with_sentence(labels, sentence_vectors, sentence1, sentence2, test_size=0.3)
        train_labels, val_labels, train_sentences, val_sentences,\
            train_sentence1, val_sentence1, train_sentence2, val_sentence2 = \
            train_test_split_with_sentence(train_labels, train_sentences, train_sentence1, train_sentence2, test_size=0.1)

        train_labels = torch.tensor(train_labels)
        val_labels = torch.tensor(val_labels)
        test_labels = torch.tensor(test_labels)
        train_sentences = torch.tensor(train_sentences)
        val_sentences = torch.tensor(val_sentences)
        test_sentences = torch.tensor(test_sentences)
        train_dataset = torch.utils.data.TensorDataset(train_sentences.float(), train_labels.long())
        val_dataset = torch.utils.data.TensorDataset(val_sentences.float(), val_labels.long())
        test_dataset = torch.utils.data.TensorDataset(test_sentences.float(), test_labels.long())

        return train_sentence1, train_sentence2, train_dataset, val_sentence1, val_sentence2, val_dataset,\
            test_sentence1, test_sentence2, test_dataset

    @staticmethod
    def load_tnli_dataset_folds(fold):
        """
        load raw sentence and corresponding dataset
        this datasets are for 5-fold cv
        Parameters
        ----------
        fold: int
        
        Returns
        -------
        Tuple
        """
        train_dataset, val_dataset, test_dataset = load_vectorized_split_tensor_dataset(fold, val=True)
        sentence1_list, sentence2_list = load_fold_sentences()
        train_sentence1, val_sentence1, test_sentence1 = train_test_fold_split(sentence1_list, fold-1, True)
        train_sentence2, val_sentence2, test_sentence2 = train_test_fold_split(sentence2_list, fold-1, True)
        return train_sentence1, train_sentence2, train_dataset, val_sentence1, val_sentence2, val_dataset,\
            np.array(test_sentence1), np.array(test_sentence2), test_dataset

    @staticmethod
    def load_tnli_bert_dataset_folds(fold):
        """
        load raw sentence and corresponding dataset
        this datasets are for 5-fold cv for bert model
        Parameters
        ----------
        fold: int
        
        Returns
        -------
        Tuple
        """
        train_dataset, val_dataset, test_dataset = load_bert_split_tensor_dataset(fold, val=True)
        sentence1_list, sentence2_list = load_fold_sentences()
        train_sentence1, val_sentence1, test_sentence1 = train_test_fold_split(sentence1_list, fold-1, True)
        train_sentence2, val_sentence2, test_sentence2 = train_test_fold_split(sentence2_list, fold-1, True)
        return train_sentence1, train_sentence2, train_dataset, val_sentence1, val_sentence2, val_dataset,\
            np.array(test_sentence1), np.array(test_sentence2), test_dataset

    @staticmethod
    def load_tnli_bart_mnli_dataset_folds(fold):
        """
        load raw sentence and corresponding dataset
        this datasets are for 5-fold cv for bart-large-mnli model
        Parameters
        ----------
        fold: int
        
        Returns
        -------
        Tuple
        """
        train_dataset, val_dataset, test_dataset = load_bart_split_tensor_dataset(fold, val=True)
        sentence1_list, sentence2_list = load_fold_sentences()
        train_sentence1, val_sentence1, test_sentence1 = train_test_fold_split(sentence1_list, fold-1, True)
        train_sentence2, val_sentence2, test_sentence2 = train_test_fold_split(sentence2_list, fold-1, True)
        return train_sentence1, train_sentence2, train_dataset, val_sentence1, val_sentence2, val_dataset,\
            np.array(test_sentence1), np.array(test_sentence2), test_dataset

    @staticmethod
    def load_tnli_explain_dataset_folds(fold):
        """
        load raw sentence and corresponding dataset
        this datasets are for 5-fold cv for explaining model
        Parameters
        ----------
        fold: int
        
        Returns
        -------
        Tuple
        """
        train_dataset, val_dataset, test_dataset = load_explain_split_tensor_dataset(fold, val=True)
        sentence1_list, sentence2_list = load_fold_sentences()
        train_sentence1, val_sentence1, test_sentence1 = train_test_fold_split(sentence1_list, fold-1, True)
        train_sentence2, val_sentence2, test_sentence2 = train_test_fold_split(sentence2_list, fold-1, True)
        return train_sentence1, train_sentence2, train_dataset, val_sentence1, val_sentence2, val_dataset,\
            np.array(test_sentence1), np.array(test_sentence2), test_dataset

    @staticmethod
    def load_snli_explain_dataset():
        """
        load explain snli dataset
        this dataset is for explaining model
        
        Examples
        --------
        >>> factory = DatasetFactory()
        >>> factory.create('explain_snli')[8][0]
        """
        train_dataset, val_dataset, test_dataset = load_explain_snli_tensor_dataset()
        train_sentence1 = None
        train_sentence2 = None
        val_sentence1 = None
        val_sentence2 = None
        test_sentence1 = None
        test_sentence2 = None
        return train_sentence1, train_sentence2, train_dataset, val_sentence1, val_sentence2, val_dataset,\
            test_sentence1, test_sentence2, test_dataset

    @staticmethod
    def load_nli_explain_dataset(dir_name):
        dataset = ExplainNLIDataset(dir_name)
        return dataset

    @staticmethod
    def load_tnli_glove_dataset_folds(fold: int):
        """
        load raw sentence and corresponding dataset
        this datasets are for 5-fold cv
        
        Parameters
        ----------
        fold: int
        
        Returns
        -------
        Tuple[torch.utils.data.Dataset]
        """
        return GloveDataset(
            "train", fold, True, Path(DATA_DIR) / "glove"
        ), GloveDataset(
            "validation", fold, True, Path(DATA_DIR) / "glove"
        ), GloveDataset(
            "test", fold, True, Path(DATA_DIR) / "glove"
        )
        # train_dataset, val_dataset, test_dataset = load_glove_split_tensor_dataset(fold, val=True, name=name, dim=dim)
        # sentence1_list, sentence2_list = load_fold_sentences()
        # train_sentence1, val_sentence1, test_sentence1 = train_test_fold_split(sentence1_list, fold-1, True)
        # train_sentence2, val_sentence2, test_sentence2 = train_test_fold_split(sentence2_list, fold-1, True)
        # return train_sentence1, train_sentence2, train_dataset, val_sentence1, val_sentence2, val_dataset,\
        #     np.array(test_sentence1), np.array(test_sentence2), test_dataset

    @staticmethod
    def load_glove_nli_dataset(dir_name):
        return load_glove_nli_dataset(dir_name)
    
    @staticmethod
    def load_raw_data(fold: int):
        train_data, val_data, test_data = load_raw_sentence_dataset(fold, True)
        return train_data, val_data, test_data

    @staticmethod
    def load_explain_knowledge_dataset(fold: int):
        return ExplainKnowledgeDataset(
            Mode.TRAIN, fold
        ), ExplainKnowledgeDataset(
            Mode.VALIDATION, fold
        ), ExplainKnowledgeDataset(
            Mode.TEST, fold
        )

    @staticmethod
    def load_glove_knowledge_dataset(fold: int, name: str, dim: int):
        return GloveKnowledgeDataset(
            Mode.TRAIN, fold, name, dim, data_dir=Path(DATA_DIR)/"glove"
        ), GloveKnowledgeDataset(
            Mode.VALIDATION, fold, name, dim, data_dir=Path(DATA_DIR)/"glove"
        ), GloveKnowledgeDataset(
            Mode.TEST, fold, name, dim, data_dir=Path(DATA_DIR)/"glove"
        )
    
    @staticmethod
    def load_explain_dataset(fold: int, val: bool, data_dir: Path):
        """
        load dataloader that are pre-saved and encoded by bert tokenizer.
        labels are {0, 1, 2},
        data are processed by berttokenizer

        Parameters
        ----------
        fold: int, the number of fold to be treated as test set
        val: bool, if it is
        data_dir: Path

        Returns
        -------
        Tuple[torch.utils.data.TensorDataset]
        """
        train_dataset = ExplainDataset('train', fold, val, data_dir)
        test_dataset = ExplainDataset('test', fold, val, data_dir)
        if val:
            val_dataset = ExplainDataset('validation', fold, val, data_dir)
            return train_dataset, val_dataset, test_dataset
        else:
            return train_dataset, test_dataset
    
    @staticmethod
    def load_knowledge_dataset(fold: int, val: bool, data_dir: Path):
        """
        load dataset that are pre-saved and encoded by bert tokenizer.
        labels are {0, 1, 2},
        data are processed by berttokenizer

        Parameters
        ----------
        fold: int, the number of fold to be treated as test set
        val: bool, if it is
        data_dir: Path

        Returns
        -------
        Tuple[torch.utils.data.TensorDataset]
        """
        train_dataset = KnowledgeDataset(Mode.TRAIN, fold, val, data_dir)
        test_dataset = KnowledgeDataset(Mode.TEST, fold, val, data_dir)
        if val:
            val_dataset = KnowledgeDataset(Mode.VALIDATION, fold, val, data_dir)
            return train_dataset, val_dataset, test_dataset
        else:
            return train_dataset, test_dataset
    

def train_test_split_with_sentence(labels, sentence_vectors, sentence1, sentence2, test_size=None, random_state=None, shuffle=True, stratify=None):
    """
    Split labels and sentences or matrices into random train and test subsets
    Parameters
    ----------
    labels: np.ndarray
    sentence_vectors: np.ndarray
    sentence1: list
    sentence2: list
    test_size: int
    random_state: int
    shuffle: bool
    stratify:

    Returns
    -------
    Tuple
    """
    list_index = list(range(len(labels)//3))
    train_index, test_index = train_test_split(list_index, test_size=test_size, random_state=random_state, shuffle=shuffle, stratify=stratify)
    train_index = [3*idx+i for idx in train_index for i in range(3)]
    test_index = [3*idx+i for idx in test_index for i in range(3)]
    train_labels = np.array([labels[idx] for idx in train_index])
    train_vectors = np.array([sentence_vectors[idx] for idx in train_index])
    train_sentence1 = np.array([sentence1[idx] for idx in train_index])
    train_sentence2 = np.array([sentence2[idx] for idx in train_index])
    test_labels = np.array([labels[idx] for idx in test_index])
    test_vectors = np.array([sentence_vectors[idx] for idx in test_index])
    test_sentence1 = np.array([sentence1[idx] for idx in test_index])
    test_sentence2 = np.array([sentence2[idx] for idx in test_index])

    return train_labels, test_labels, train_vectors, test_vectors,\
           train_sentence1, test_sentence1, train_sentence2, test_sentence2
