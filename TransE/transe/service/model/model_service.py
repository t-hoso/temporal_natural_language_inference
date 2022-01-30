import pathlib
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

sys.path.append("...")
from ...models.models import SentenceTransE, SentenceTransH, SentenceTransformerEncoder, RelationEncoder, SentenceComplEx
from ...models.loss_function import l2_distance
from ...models.dataset import Atomic2020Dataset


# TODO: split files
def density_estimator(model, data_tuple):
    """
    This function works as density estimator
    This is merely working as a l2 distance
    
    Parameters
    ----------
    model: torch.nn.Module
    data_tuple: the input tuple for the model
    
    Returns
    -------
    the distance between sentence1 + relation and sentence2
    """
    data = model(data_tuple)
    return torch.dist(data[0] + data[1], data[2]).detach().cpu().numpy()


def fixed_density_estimator(model, head, relation, tail):
    """
    This function works as density estimator
    This is merely working as a l2 distance
    When we calculate ranking, we do same transformation for the same head+relation or tail+relation pair.
    Thus we do not need to give head or tail to given model

    Parameters
    ----------
    model: torch.nn.Module
    head: list or torch.Tensor
    relation: torch.Tensor
    tail: list or torch.Tensor

    Returns
    -------
    dense
        the estimated density
    """
    dense = None
    if type(head) == list:
        head = model._forward_sentence(head)
        dense = torch.cdist(head + relation, tail).detach().cpu().numpy().reshape(-1)

    if type(tail) == list:
        tail = model._forward_sentence(tail)
        dense = torch.cdist(head + relation, tail).detach().cpu().numpy().reshape(-1)
    return dense


def density_estimator_many(model, data_triples):
    """
    This function calculates density of many examples

    Parameters
    ----------
    model: torch.nn.Module
    data_triples
        the data
    
    Returns
    -------
    dist
        the distance
    """
    data = model(data_triples)
    return torch.cdist(data[0] + data[1], data[2])[0].detach().cpu().numpy()


def mean_ranks(ranking):
    """
    This function calculates the mean rank
    This works as a evaluation function

    Parameters
    ----------
    ranking: List[int]
        the ranks.
    
    Returns
    -------
    mean
        the mean of them
    """
    return np.mean(np.array(ranking))


def hits_at(k, ranking):
    """
    This function calculates hits@k

    Parameters
    ----------
    k: int
    ranking: list

    Returns
    -------
    hits
        the number of entries that are less or equals to k
    """
    np_ranking = np.array(ranking)
    return np.sum(np_ranking <= k) / len(ranking)


class ModelService:
    """
    Trainer class
    Trains the SentenceTransE model

    Attributes
    ----------
    parameters: dict
    device: torch.device
    """
    def __init__(self, parameters, device):
        """
        Parameters
        ----------
        parameters: dict
            requires: {'n_epochs': int, 'tensorboard_path': str or pathlib.Path}
        device: torch.device
        """
        self.parameters = parameters
        self.device = device

    def fit(self, model, train_dataloader, val_dataloader, criterion, optimizer):
        """
        Actually trains model using data in dataloader.

        Parameters
        ----------
        model: torch.nn.Module
            the model
        train_dataloader: torch.utils.data.DataLoader
            the data loader of training set
        val_dataloader: torch.utils.data.DataLoader
            the data loader of vallidation set
        criterion: torch._Loss
            the criterion of the training
        optimizer: torch.optim.Optimizer
            the optimizer
        
        Returns
        -------
        model: torch.Module
            trained model
        """
        device = self.device
        n_epochs = self.parameters['n_epochs']
        writer = SummaryWriter(log_dir=self.parameters['tensorboard_path'])

        model.train()
        model.to(device)
        target = torch.FloatTensor([-1]).to(device)
        criterion.to(device)

        for epoch in range(1, n_epochs + 1):
            loss_train = 0.0
            loss_val = 0.0

            for batch in train_dataloader:
                positive_example, negative_example = batch
                out = model(positive_example)
                corrupted_out = model(negative_example)
                loss = criterion(out, corrupted_out, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_train += loss.item()

            with torch.no_grad():
                for batch in val_dataloader:
                    positive_example, negative_example = batch
                    out = model(positive_example)
                    corrupted_out = model(negative_example)
                    loss = criterion(out, corrupted_out, target)
                    loss_val += loss.item()

            if epoch % 1 == 0:
                print("epoch: {}, training_loss: {}, val_loss: {}".format(epoch, loss_train/len(train_dataloader),
                                                                          loss_val/len(val_dataloader)))
            writer.add_scalars("loss", {'train': loss_train/len(train_dataloader), 'val': loss_val/len(val_dataloader)}, epoch)
        writer.close()
        return model

    def test(self, model, train_dataset, test_dataset, k):
        """
        Tests the performance of model using mean ranks and hits@k
        All entities are retrieved from training/test sets
        
        Parameters
        ----------
        model: torch.nn.Module
        train_dataset: torch.utils.data.Dataset
            train dataset
        test_dataset: torch.utils.data.Dataset
            test dataset
        k: int
            k for hits@k
        
        Returns
        -------
        mean, hits: float, float
            the test result
        """
        model.eval()
        all_nodes = set(train_dataset.nodes) | set(test_dataset.nodes)
        replace_indices = [0, 2]
        ranking = []
        with torch.no_grad():
            for edge_replace_idx in replace_indices:
                ranking = ranking + self._ranking(model, all_nodes, test_dataset, edge_replace_idx)
        return mean_ranks(ranking), hits_at(k, ranking), ranking

    def _ranking(self, model, all_nodes, test_dataset, edge_replace_idx):
        """
        calculates the ranking of all test dataset entities
        replace either head or tail depending of edge_replace_idx
        
        Parameters
        ----------
        model: torch.nn.Module
        all_nodes: list
            the nodes
        test_dataset: torch.utils.dataDataset
        edge_replace_idx: int
            the index to be replaced. either 0 or 2
        
        Returns
        -------
        ranking_list: list
            the ranking of each entities
        """
        import time
        t = time.time()
        num_nodes = 1000
        ranking_list = []
        print("s")
        for data in test_dataset:
            density_list = []
            idx = 0
            data_edge = data[edge_replace_idx]
            relation = model._forward_relation([data[1]])
            head = data[0]
            tail = data[2]
            if edge_replace_idx == 2:
                head = model._forward_sentence([data[0]])
                tail = []
            else:
                head = []
                tail = model._forward_sentence([data[2]])
            i_s = []
            cnt = 0
            for i, node in enumerate(list(all_nodes)):
                if node == data_edge:
                    idx = i
                if edge_replace_idx == 2:
                    tail.append(node)
                else:
                    head.append(node)

                cnt += 1
                if cnt == num_nodes:
                    densities = fixed_density_estimator(model, head, relation, tail)
                    density_list.append(densities)
                    cnt = 0
                    if edge_replace_idx == 2:
                        tail = []
                    else:
                        head = []

            replaced_data = head if edge_replace_idx == 0 else tail
            if len(replaced_data) >= 1:
                densities = fixed_density_estimator(model, head, relation, tail)
                density_list.append(densities)
            density_list = np.concatenate(density_list)
            ranking = self._find_rank(density_list, idx)
            ranking_list.append(ranking + ranking)

        return ranking_list

    def save(self, model, filename):
        #TODO
        pass
