from typing import List
import sys
import functools
import pathlib
from typing import Union

import torch
import torch.optim as optim

from transe.models.models.model_factory import EXPLAIN_TRANSE

sys.path.append("...")
from ...models.dataset import Atomic2020Dataset, ExplainDataset
from ...models.dataset.collate_fn import collate_four_sentences
from ...service.model import ModelService
from ...models.models import ModelFactory
from ...models.loss_function import (
    MarginBasedRankingLoss,
    l2_distance,
    NegativeLikelihoodLossNegativeSampling
)

COMPLEX = "complex"
MODEL_NAME_EXPLAIN_TRANSE = "explain_transe"
MODEL_NAME_ROBERTA = "roberta-base"
BATCH_SIZE = 16

class Training:
    """
    Class for training
    """
    # TODO: limit some of parameters that are not necessary for some models
    @staticmethod
    def train(model_name: str, 
              mapped_embedding_dim: int,
              relation: List[str],
              tensorboard_path: str,
              lr: float,
              n_epochs: int=30,
              model_save_path: Union[str, pathlib.Path]="",
              margin: float=0.5, 
              epsilon: float=1e-10, 
              constraint_weight: float=0.25):
        """
        actually trains the model

        Parameters
        ----------
        model_name: str
            model name
        mapped_embedding_dim: int
            the mapped embedding dim
        relation: List[str]
            the relations to be used
        tensorboard_path: str
            the path for tensorboard
        lr: float
            learning rate
        n_epochs: int
            the number of epocsh
        model_save_path: Union[str, pathlib.Path]
            path for saving model
        margin: float
            margin for the margin based ranking loss
        epsilon: float
            epsilon value for optimizers 
        constraint_weight: float
            constraint weight for constraint version of marign ranking loss
        """
        # TODO: Construct dataset instances out this function
        if model_name == EXPLAIN_TRANSE:
            train_dataset = ExplainDataset(
                mode="train", 
                relation=relation, 
                bert_path=MODEL_NAME_ROBERTA
            )
            val_dataset = ExplainDataset(
                mode="dev", 
                relation=relation, 
                bert_path=MODEL_NAME_ROBERTA
            )
            train_dataloader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=BATCH_SIZE,
                collate_fn=functools.partial(collate_four_sentences, fill_values=[1, 0, 0]),
                drop_last=False
            )
            val_dataloader = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=BATCH_SIZE,
                collate_fn=functools.partial(collate_four_sentences, fill_values=[1, 0, 0]),
                drop_last=False
            )
        else:
            train_dataset = Atomic2020Dataset(mode="train", relation=relation)
            val_dataset = Atomic2020Dataset(mode="dev", relation=relation)
            train_dataloader = torch.utils.data.DataLoader(train_dataset, shuffle=False, batch_size=BATCH_SIZE)
            val_dataloader = torch.utils.data.DataLoader(val_dataset, shuffle=False, batch_size=BATCH_SIZE)
        model = ModelFactory.create_instance(model_name, mapped_embedding_dim)
        parameters = {
        "tensorboard_path": tensorboard_path,
        "n_epochs": n_epochs
        }
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        # TODO: Construct optimizer out of this function
        if model_name == COMPLEX:
            criterion = NegativeLikelihoodLossNegativeSampling()
            optimizer = optim.Adagrad(model.parameters(), lr=lr)
        else:
            criterion = MarginBasedRankingLoss(margin=margin, distance_function=l2_distance)
            optimizer = optim.SGD(model.parameters(), lr=lr)
        model_service = ModelService(parameters, device)

        model = model_service.fit(model, train_dataloader, val_dataloader, criterion, optimizer)
        torch.save(model.to('cpu').state_dict(), model_save_path)
