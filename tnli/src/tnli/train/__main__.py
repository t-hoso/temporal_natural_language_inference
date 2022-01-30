import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import click

from .dataset_type_manager import DatasetTypeManager
from .trainers import TrainerFactory
from .model_manager import ModelManager
from .utils import DataLoaderFactory, Mode, DatasetType

BATCH_SIZE = 16


@click.command()
@click.option("--dataset_type", type=str)
@click.option("--batch_size", default=16, type=int)
@click.option("--num_warmup_steps", default=0, type=int)
@click.option("--epochs", default=40, type=int)
@click.option("--lr", type=float, default=0.00001)
@click.option("--eps", type=float)
@click.option("--betas", type=float, multiple=True)
@click.option("--weight_decay", type=float)
@click.option("--fold", type=int)
@click.option("--tensorboard_path", type=click.Path(exists=False))
@click.option("--lamb", type=float)
@click.option("--loss_fn_name", type=str)
@click.option("--optimizer_name", type=str)
@click.option("--model_name", type=str)
@click.option("--model_save_path", type=click.Path(exists=False))
@click.option("--model_load_path", default="")
def run(dataset_type,
        batch_size,
        num_warmup_steps,
        epochs,
        lr,
        eps,
        betas,
        weight_decay,
        fold,
        tensorboard_path,
        lamb,
        loss_fn_name,
        optimizer_name,
        model_name,
        model_save_path,
        model_load_path):
    random.seed(0)
    torch.manual_seed(0)

    dataset_kind = DatasetTypeManager.get_dataset_type_from_string(dataset_type)
    model_name = ModelManager.create_model_name_setting(model_name)
    model = ModelManager.create_model(model_name)
    if model_load_path:
        model.load_state_dict(torch.load(model_load_path))
    trainer = TrainerFactory.create_instance(
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
        model
    )
    model.train()
    if (dataset_kind == DatasetType.EXPLAIN_MNLI
        or dataset_kind == DatasetType.GLOVE_MNLI):
        loss_train, accuracy_train, loss_val, accuracy_val, epoch, model = trainer.fit(model,
          DataLoaderFactory.create_instance(dataset_kind, fold, Mode.TRAIN, BATCH_SIZE),
          [DataLoaderFactory.create_instance(dataset_kind, fold, Mode.MATCHED, BATCH_SIZE),
          DataLoaderFactory.create_instance(dataset_kind, fold, Mode.MISMATCHED, BATCH_SIZE)]
        )

    else:
          loss_train, accuracy_train, loss_val, accuracy_val, epoch, model = trainer.fit(model,
          DataLoaderFactory.create_instance(dataset_kind, fold, Mode.TRAIN, BATCH_SIZE),
          DataLoaderFactory.create_instance(dataset_kind, fold, Mode.VALIDATION, BATCH_SIZE)
          )
          model.eval()
          loss_test, accuracy, cm, precision, recall, f1 = \
            trainer.test(
                model,
                DataLoaderFactory.create_instance(dataset_kind, fold, Mode.TEST, 1)
            )
          print(loss_test, accuracy, cm, precision, recall, f1)
    torch.save(model.to("cpu").state_dict(), model_save_path)
    print(loss_train, accuracy_train, loss_val, accuracy_val)

run()










