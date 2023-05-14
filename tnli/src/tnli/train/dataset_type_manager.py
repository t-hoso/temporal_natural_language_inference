import sys
sys.path.append(".")

from .utils import DatasetType

class DatasetTypeManager:
    @staticmethod
    def get_dataset_type_from_string(dataset_type: str):
        if dataset_type == "sbert_fold":
            return DatasetType.SBERT_FOLD
        elif dataset_type == "bert_fold":
            return DatasetType.BERT_FOLD
        elif dataset_type == "explain_fold":
            return DatasetType.EXPLAIN_FOLD
        elif dataset_type == "explain_snli":
            return DatasetType.EXPLAIN_SNLI
        elif dataset_type == "explain_MNLI":
            return DatasetType.EXPLAIN_MNLI
        elif dataset_type == "glove_fold":
            return DatasetType.GLOVE_FOLD
        elif dataset_type == "glove_snli":
            return DatasetType.GLOVE_SNLI
        elif dataset_type == "glove_mnli":
            return DatasetType.GLOVE_MNLI
        elif dataset_type == "sbert_random":
            return DatasetType.SBERT_RANDOM
        elif dataset_type == "raw_sentence":
            return DatasetType.RAW_SENTENCE
        elif dataset_type == "explain_knowledge":
            return DatasetType.EXPLAIN_KNOWLEDGE
        elif dataset_type == "glove_knowledge":
            return DatasetType.GLOVE_KNOWLEDGE
        elif dataset_type == "explain_bert":
            return DatasetType.EXPLAIN_BERT
        elif dataset_type == "knowledge_explain_bert":
            return DatasetType.KNOWLEDGE_EXPLAIN_BERT
        elif dataset_type == "bart_mnli":
            return DatasetType.BART_MNLI
