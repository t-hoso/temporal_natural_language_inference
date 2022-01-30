from enum import Enum


class DatasetType(Enum):
    SBERT_FOLD = 1
    BERT_FOLD = 2
    EXPLAIN_FOLD = 3
    EXPLAIN_SNLI = 4
    EXPLAIN_MNLI = 5
    GLOVE_FOLD = 6
    GLOVE_SNLI = 7
    GLOVE_MNLI = 8
    RAW_SENTENCE = 10
    EXPLAIN_KNOWLEDGE = 11
    GLOVE_KNOWLEDGE = 12
    EXPLAIN_BERT = 13
    KNOWLEDGE_EXPLAIN_BERT = 14

    SBERT_RANDOM = 9