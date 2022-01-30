import sys
import os

import torch
import torchtext as text
from transformers import BertForSequenceClassification

from Self_Explaining_Structures_Improve_NLP_Models.explain.model import ExplainableModel
from transe.models.models import(
    SentenceTransE, 
    RelationEncoder, 
    SentenceTransformerEncoder
)

sys.path.append(".")
from .models import(
    FeedForwardNetwork, 
    OneLayerModel, 
    KnowledgeOnlyModel,
    KnowledgeExplainModel,
    KnowledgeOneLayerModel,
    KnowledgeOnlyReluModel,
    KnowledgeOnlySubtractionModel,
    KnowledgeOnlySubtractionReluModel,
    KnowledgeOnlyCombineReluModel,
    TransEExplainableModel,
    ExplainModelBuilder
)
from .settings import Setting


NUM_CLASSES = 3
DIM_SBERT = 768
DIM_FFN_LAYER = 500
BERT_MODEL_NAME = "bert-base-uncased"
ROBERTA_MODEL_NAME = "roberta-base"
SIAMESE_DROPOUT_RATE = 0.5
GLOVE_DIM = 300
GLOVE_NAME = "6B"
SIAMESE_HIDDEN_DIM = 100
TRANSE_DIM = 256
TRANSE_PRETRAINED_FILENAME = os.environ.get("TRANSE_PRETRAINED_FILENAME")
CONVERT_DIM = 128


class ModelManager:
    @classmethod
    def create_model(cls, model_name: str):
        setting = Setting()
        if model_name == setting.MODEL_NAME_FFN:
            return FeedForwardNetwork(DIM_SBERT*2, DIM_FFN_LAYER, NUM_CLASSES)
        elif model_name == setting.MODEL_NAME_BERT:
            return BertForSequenceClassification.from_pretrained(
                BERT_MODEL_NAME, 
                num_labels=NUM_CLASSES, 
                output_attentions = False,
                output_hidden_states = False
            )
        elif model_name == setting.MODEL_NAME_EXPLAIN:
            return ExplainModelBuilder() \
                    .build_instance(
                        ROBERTA_MODEL_NAME, NUM_CLASSES, True)
        elif model_name == setting.MODEL_NAME_SIAMESE:
            return OneLayerModel(GLOVE_DIM, 
                                 SIAMESE_HIDDEN_DIM, 
                                 NUM_CLASSES, 
                                 SIAMESE_DROPOUT_RATE,
                                 text.vocab.GloVe(name=GLOVE_NAME, dim=GLOVE_DIM))
        elif model_name == setting.MODEL_NAME_KNOWLEDGE_ONLY:
            transe = cls.create_transe_model(TRANSE_PRETRAINED_FILENAME)
            return KnowledgeOnlyModel(
                transe, TRANSE_DIM, NUM_CLASSES
            )
        elif model_name == setting.MODEL_NAME_EXPLAIN_KNOWLEDGE:
            transe = cls.create_transe_model(TRANSE_PRETRAINED_FILENAME)
            return KnowledgeExplainModel(
                ExplainModelBuilder()
                    .build_instance(
                        ROBERTA_MODEL_NAME, NUM_CLASSES, True),
                TRANSE_DIM,
                transe, CONVERT_DIM, NUM_CLASSES
            )
        elif model_name == setting.MODEL_NAME_SIAMESE_KNOWLEDGE:
            transe = cls.create_transe_model(TRANSE_PRETRAINED_FILENAME)
            return KnowledgeOneLayerModel(
                GLOVE_DIM,
                SIAMESE_HIDDEN_DIM,
                SIAMESE_DROPOUT_RATE,
                text.vocab.GloVe(name=GLOVE_NAME, dim=GLOVE_DIM),
                transe,
                TRANSE_DIM,
                NUM_CLASSES
            )
        elif model_name == setting.MODEL_NAME_KNOWLEDGE_ONLY_SUBTRACTION:
            transe = cls.create_transe_model(TRANSE_PRETRAINED_FILENAME)
            return KnowledgeOnlySubtractionModel(
                transe, TRANSE_DIM, NUM_CLASSES
            )
        elif model_name == setting.MODEL_NAME_KNOWLEDGE_ONLY_RELU:
            transe = cls.create_transe_model(TRANSE_PRETRAINED_FILENAME)
            return KnowledgeOnlyReluModel(
                transe, TRANSE_DIM, NUM_CLASSES
            )
        elif model_name == setting.MODEL_NAME_kNOWLEDGE_ONLY_SUBTRACTION_RELU:
            transe = cls.create_transe_model(TRANSE_PRETRAINED_FILENAME)
            return KnowledgeOnlySubtractionReluModel(
                transe, TRANSE_DIM, NUM_CLASSES
            )
        elif model_name == setting.MODEL_NAME_KNOWLEDGE_ONLY_COMBINE_RELU:
            transe = cls.create_transe_model(TRANSE_PRETRAINED_FILENAME)
            return KnowledgeOnlyCombineReluModel(
                transe, TRANSE_DIM, NUM_CLASSES
            )     
        elif model_name == setting.MODEL_NAME_TRANSE_EXPLAIN:
            return TransEExplainableModel(
                ROBERTA_MODEL_NAME,
                TRANSE_PRETRAINED_FILENAME,
                NUM_CLASSES
            )
        elif model_name == setting.MODEL_NAME_EXPLAIN_BERT:
            return ExplainModelBuilder().build_instance(BERT_MODEL_NAME, NUM_CLASSES)
        elif model_name == setting.MODEL_NAME_KNOWLEDGE_EXPLAIN_BERT:
            transe = cls.create_transe_model(TRANSE_PRETRAINED_FILENAME)
            return KnowledgeExplainModel(
                ExplainModelBuilder()
                    .build_instance(
                        BERT_MODEL_NAME, NUM_CLASSES, True),
                TRANSE_DIM,
                transe, CONVERT_DIM, NUM_CLASSES
            )

    @staticmethod
    def create_model_name_setting(model_name: str):
        setting = Setting()
        if model_name == "ffn":
            return setting.MODEL_NAME_FFN
        elif model_name == "bert":
            return setting.MODEL_NAME_BERT
        elif model_name == "explain":
            return setting.MODEL_NAME_EXPLAIN
        elif model_name == "siamese":
            return setting.MODEL_NAME_SIAMESE
        elif model_name == "knowledge_only":
            return setting.MODEL_NAME_KNOWLEDGE_ONLY
        elif model_name == "explain_knowledge":
            return setting.MODEL_NAME_EXPLAIN_KNOWLEDGE
        elif model_name == "siamese_knowledge":
            return setting.MODEL_NAME_SIAMESE_KNOWLEDGE
        elif model_name == "knowledge_only_subtraction":
            return setting.MODEL_NAME_KNOWLEDGE_ONLY_SUBTRACTION
        elif model_name == "knowledge_only_relu":
            return setting.MODEL_NAME_KNOWLEDGE_ONLY_RELU
        elif model_name == "knowledge_only_subtraction_relu":
            return setting.MODEL_NAME_kNOWLEDGE_ONLY_SUBTRACTION_RELU
        elif model_name == "knowledge_only_combine_relu":
            return setting.MODEL_NAME_KNOWLEDGE_ONLY_COMBINE_RELU
        elif model_name == "transe_explain":
            return setting.MODEL_NAME_TRANSE_EXPLAIN
        elif model_name == "explain_bert":
            return setting.MODEL_NAME_EXPLAIN_BERT
        elif model_name == "knowledge_explain_bert":
            return setting.MODEL_NAME_KNOWLEDGE_EXPLAIN_BERT
    
    @staticmethod
    def create_transe_model(pretrained_path):
        sentence_encoder = SentenceTransformerEncoder(model_name='paraphrase-distilroberta-base-v1')
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
        sentence_embedding_dim = 768
        num_relation = len(relation)
        mapped_embedding_dim = 256
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        sentence_transe = SentenceTransE(sentence_encoder,
                            relation_encoder,
                            sentence_embedding_dim,
                            num_relation,
                            mapped_embedding_dim,
                            device)
        if TRANSE_PRETRAINED_FILENAME:
            sentence_transe.load_state_dict(torch.load(pretrained_path))
        return sentence_transe
