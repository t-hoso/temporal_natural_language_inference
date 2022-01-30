import sys
import torch

sys.path.append(".")
from .sentence_transformer_encoder import SentenceTransformerEncoder
from .relation_encoder import RelationEncoder
from .transe import SentenceTransE
from .transh import SentenceTransH
from .complex import SentenceComplEx
from .self_explaining_transe import SelfExplainingTransE
from .layers import ExplainableBase


SBERT_NAME = "paraphrase-distilroberta-base-v1"
SENTENCE_EMBEDDING_DIM = 768
TRANSE = "transe"
TRANSH = "transh"
COMPLEX = "complex"
EXPLAIN_TRANSE = "explain_transe"


class ModelFactory:
    """
    The model factory
    """
    @staticmethod
    def create_instance(model_name: str, mapped_embedding_dim: int = 256):
        """
        Creates an instance of model

        Parameters
        ----------
        model_name: str
            the name of model
        mapped_embedding_dim: int
            the dimension of the embedding that the model outputs

        Returns
        -------
        torch.nn.Module
        """
        sentence_encoder = SentenceTransformerEncoder(model_name=SBERT_NAME)
        relation = ["isAfter", "isBefore", "HinderedBy",
                    'oEffect', 'oReact', 'oWant',
                    'xNeed', 'xAttr', 'xEffect',
                    'xIntent', 'xWant', 'xReact',
                    'MadeUpOf', 'Causes',
                    'ObjectUse', 'AtLocation', 'HasProperty',
                    'CapableOf', 'Desires', 'NotDesires',
                    'HasSubEvent', 'xReason',
                    'isFilledBy']
        relation_encoder = RelationEncoder(relation)
        sentence_embedding_dim = SENTENCE_EMBEDDING_DIM
        num_relation = len(relation)
        mapped_embedding_dim = mapped_embedding_dim
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if model_name == TRANSE:
            return SentenceTransE(sentence_encoder,
                                relation_encoder,
                                sentence_embedding_dim,
                                num_relation,
                                mapped_embedding_dim,
                                device)            
        elif model_name == TRANSH:
            return SentenceTransH(sentence_encoder,
                                relation_encoder,
                                sentence_embedding_dim,
                                num_relation,
                                mapped_embedding_dim,
                                device)
        elif model_name == COMPLEX:
            return SentenceComplEx(sentence_encoder,
                        relation_encoder,
                        sentence_embedding_dim,
                        num_relation,
                        mapped_embedding_dim,
                        device)
        elif model_name == EXPLAIN_TRANSE:
            return SelfExplainingTransE(
                explain_base=ExplainableBase("roberta-base"),
                num_relation=num_relation,
                device=device
            )