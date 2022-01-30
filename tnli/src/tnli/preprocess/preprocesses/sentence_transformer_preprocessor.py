import csv
from pathlib import Path
from typing import Any

import numpy as np

from .preprocessor import Preprocessor
from .encoders import SentenceTransformersEncoder


INDEX_DATA_SENTENCE1 = 1
INDEX_DATA_SENTENCE2 = 2
INDEX_DATA_LABEL = 0


class SentenceTransformerPreprocessor(Preprocessor):
    """
    Preprocessor for Sentence Transformer setting
    """
    @classmethod
    def write(
        cls,
        data: tuple,
        output_dir_name: Path, 
        model_name: str,
        count: int):
        """
        write and save
        """
        print(str(output_dir_name/('label_'+str(count)))+".csv")
        encoded_labels, encoded_sentence1s, encoded_sentence2s = data

        with open(
            str(output_dir_name/('label_'+str(count)))+".csv", 
            encoding='utf-8', mode='w') as wf:
            writer = csv.writer(wf)
            for label in encoded_labels:
                writer.writerow([label])
        with open(
            str(output_dir_name/('sentence1_'+str(count)))+".csv", 
            encoding='utf-8', mode='w') as wf:
            writer = csv.writer(wf)
            for sentence in encoded_sentence1s:
                writer.writerow(sentence)
        with open(
            str(output_dir_name/('sentence2_'+str(count)))+".csv", 
            encoding='utf-8', mode='w') as wf:
            writer = csv.writer(wf)
            for sentence in encoded_sentence2s:
                writer.writerow(sentence)

    @classmethod
    def encode(
        cls,
        data: tuple,
        encoder: Any, 
        max_len: int
    ):
        """
        encode one set of data

        Parameters
        ----------
        data: tuple
            sentence1, sentence2, and labels
        encoder: Any
            the encoder such as BertTokenizer
        max_len: int
            the maximum length of input
        """
        return (cls.__encode_labels(data[INDEX_DATA_LABEL]),
            cls.__encode_sentence_transformer(
                data[INDEX_DATA_SENTENCE1], encoder),
            cls.__encode_sentence_transformer(
                data[INDEX_DATA_SENTENCE2], encoder)
        )

    @staticmethod
    def __encode_sentence_transformer(
        sentences: np.ndarray, encoder: SentenceTransformersEncoder):
        """
        encode sentences

        Parameters
        ----------
        sentences: np.ndarray
            array of str, sentences
        encoder: SentenceTransformersEncoder
            encoder
        
        Returns
        -------
            encoded sentences
        
        Examples
        --------
        >>> encoder = SentenceTransformersEncoder()
        >>> sentences = np.array(["A white dog running through the snow", "A white dog sleeping on the snow", "The sun feels very hot on the girls' heads." ])
        >>> encode_sentence_transformer(sentences, encoder)
        array([[-0.03229558,  0.16357218,  0.11522663, ...,  0.08778763,
                -0.21562025,  0.38087332],
            [-0.0068887 ,  0.2275389 ,  0.2428784 , ...,  0.06617424,
                -0.06620031,  0.36235854],
            [ 0.32937512,  0.41474926,  0.07971326, ...,  0.03852906,
                -0.10188176, -0.35676098]], dtype=float32)
        """
        return encoder.encode_many(sentences)

    @staticmethod
    def __encode_labels(labels: np.ndarray):
        """
        encode labels{'Invalidate', 'Support', 'Neutral'} to 0, 1, 2, respectively

        Parameters
        ----------
            labels: np.ndarray of str
        
        Returns
        -------
        np.ndarray
            encoded version of labels

        Examples
        --------
        >>> labels = np.array(['Invalidate', 'Support', 'Neutral', 'Neutral'])
        >>> encode_labels(labels)
        array([0, 1, 2, 2], dtype=object)
        """
        # encode one label
        label2number = {'Invalidate': 0, 'Support': 1, 'Neutral': 2}
        def encode_label(label):
            return label2number[label]
        u_encode_label = np.frompyfunc(encode_label, 1, 1)

        return u_encode_label(labels)
