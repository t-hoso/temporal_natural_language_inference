import torch.nn as nn

from transformers.models.roberta.modeling_roberta import RobertaModel
from transformers.models.bert.modeling_bert import BertModel

from .layers.roberta import Roberta
from .layers.identity import Identity
from .layers.explainable_base import ExplainableBase
from .explainable_model import ExplainableBertsModel


BERT_DIR_ROBERTA_BASE = "roberta-base"


class ExplainModelBuilder:
    def __init__(self):
        pass

    def build_instance(self, bert_dir, num_labels, is_base=False):
        bert = None
        if bert_dir[:4] == "bert":
            bert = BertModel.from_pretrained(bert_dir)
        elif bert_dir[:7] == "roberta":
            bert = RobertaModel.from_pretrained(bert_dir)
        else:
            raise ValueError(f"bert_dir {bert_dir} is not correct")
        if is_base:
            return ExplainableBertsModel(
            ExplainableBase(
                BERT_DIR_ROBERTA_BASE,
                Roberta(bert)
            ),
            Identity()
        )
        
        return ExplainableBertsModel(
            ExplainableBase(
                BERT_DIR_ROBERTA_BASE,
                Roberta(bert)
            ),
            nn.Linear(bert.config.hidden_size, num_labels)
        )