import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AdamW, get_linear_schedule_with_warmup

from .trainer import Trainer
from .ffn_trainer import FFNTrainer
from .bert_trainer import BertTrainer
from .explain_trainer import ExplainTrainer
from .explain_mnli_trainer import ExplainMNLITrainer
from .siamese_trainer import SiameseTrainer
from .knowledge_only_trainer import KnowledgeOnlyTrainer
from .explain_knowledge_trainer import ExplainKnowledgeTrainer
from .siamese_knowledge_trainer import SiameseKnowledgeTrainer
from ..settings import Setting
from ..utils import DatasetFactory, DatasetType, Mode
from ..loss import SelfExplainLoss


class TrainerFactory(object):
    def __setup_optimizer(batch_size,
                        model_name,
                        num_warmup_steps,
                        epochs,
                        lr,
                        eps,
                        betas,
                        weight_decay,
                        dataset_kind,
                        fold,
                        tensorboard_path,
                        lamb,
                        loss_fn_name,
                        optimizer_name,
                        model):
        setting = Setting()
        if optimizer_name == setting.OPTIMIZER_ADAM:
            optimizer = optim.Adam(model.parameters(), lr=lr)
        elif optimizer_name == setting.OPTIMIZER_ADAMW:
            optimizer = AdamW(
            model.parameters(),
            betas=betas,
            lr=lr,  # 5e-5, 3e-5, 2e-5
            eps=eps,  #
            weight_decay=weight_decay  # 0.01
        )
        return optimizer

    @classmethod
    def create_instance(cls,
                        batch_size,
                        model_name,
                        num_warmup_steps,
                        epochs,
                        lr,
                        eps,
                        betas,
                        weight_decay,
                        dataset_kind,
                        fold,
                        tensorboard_path,
                        lamb,
                        loss_fn_name,
                        optimizer_name,
                        model) -> Trainer:
        setting = Setting()

        if loss_fn_name == setting.LOSS_FN_CROSS_ENTROPY_LOSS:
            loss_fn = nn.CrossEntropyLoss()
        elif loss_fn_name == setting.LOSS_FN_SELF_EXPLAIN_LOSS:
            loss_fn = SelfExplainLoss(nn.CrossEntropyLoss(), lamb)
        else:
            raise ValueError(loss_fn_name + "is not yet acceptable")

        optimizer = cls.__setup_optimizer(batch_size,
                        model_name,
                        num_warmup_steps,
                        epochs,
                        lr,
                        eps,
                        betas,
                        weight_decay,
                        dataset_kind,
                        fold,
                        tensorboard_path,
                        lamb,
                        loss_fn_name,
                        optimizer_name,
                        model)
        total_steps = len(DatasetFactory.create_instance(
            dataset_kind, fold, Mode.TRAIN
        )) * epochs

        if optimizer_name == setting.OPTIMIZER_ADAMW:
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=num_warmup_steps,  # 10000
                num_training_steps=total_steps
            )
        else:
            scheduler = None
        
        if model_name == setting.MODEL_NAME_FFN:
            return FFNTrainer(
                 model_name,
                 epochs,
                 tensorboard_path,
                 loss_fn, 
                 optimizer,
                 scheduler
            )
        elif model_name == setting.MODEL_NAME_BERT:
            return BertTrainer(
                 model_name,
                 epochs,
                 tensorboard_path,
                 loss_fn, 
                 optimizer,
                 scheduler
            )
        elif model_name == setting.MODEL_NAME_EXPLAIN and dataset_kind == DatasetType.EXPLAIN_MNLI:
            return ExplainMNLITrainer(
                 model_name,
                 epochs,
                 tensorboard_path, 
                 loss_fn, 
                 optimizer,
                 scheduler                
            )
        elif model_name in (
            setting.MODEL_NAME_EXPLAIN,
            setting.MODEL_NAME_TRANSE_EXPLAIN,
            setting.MODEL_NAME_EXPLAIN_BERT
        ):
            return ExplainTrainer(
                 model_name,
                 epochs,
                 tensorboard_path, 
                 loss_fn, 
                 optimizer,
                 scheduler
            )
        elif model_name == setting.MODEL_NAME_SIAMESE and dataset_kind == DatasetType.GLOVE_MNLI:
            return SiameseTrainer(
                 model_name,
                 epochs,
                 tensorboard_path, 
                 loss_fn, 
                 optimizer,
                 scheduler
            )        
        elif model_name == setting.MODEL_NAME_SIAMESE:
            return SiameseTrainer(
                 model_name,
                 epochs,
                 tensorboard_path, 
                 loss_fn, 
                 optimizer,
                 scheduler
            )
        elif model_name in (
            setting.MODEL_NAME_KNOWLEDGE_ONLY,
            setting.MODEL_NAME_KNOWLEDGE_ONLY_SUBTRACTION,
            setting.MODEL_NAME_KNOWLEDGE_ONLY_RELU,
            setting.MODEL_NAME_kNOWLEDGE_ONLY_SUBTRACTION_RELU,
            setting.MODEL_NAME_KNOWLEDGE_ONLY_COMBINE_RELU
            ):
            return KnowledgeOnlyTrainer(
                model_name,
                epochs,
                tensorboard_path,
                loss_fn,
                optimizer,
                scheduler
            )
        elif model_name in (
            setting.MODEL_NAME_SIAMESE_KNOWLEDGE
        ):
            return SiameseKnowledgeTrainer(
                model_name,
                epochs,
                tensorboard_path,
                loss_fn,
                optimizer,
                scheduler
            )
        elif model_name in (
            setting.MODEL_NAME_KNOWLEDGE_EXPLAIN_BERT,
            setting.MODEL_NAME_EXPLAIN_KNOWLEDGE
        ):
            return ExplainKnowledgeTrainer(
                model_name,
                epochs,
                tensorboard_path,
                loss_fn,
                optimizer,
                scheduler
            )
