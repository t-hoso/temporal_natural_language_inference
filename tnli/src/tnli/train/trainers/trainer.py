import datetime
import copy
from abc import ABC, abstractmethod
import sys

import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
import torch

sys.path.append(".")
from .early_stopping import EarlyStopping

class Trainer(ABC):
    """
    this class is responsible for one model and training
    """
    def __init__(self,
                 model_name,
                 epochs,
                 tensorboard_path, 
                 loss_fn, 
                 optimizer,
                 scheduler):
        self.model_name = model_name
        self.epochs = epochs
        self.tensorboard_path = tensorboard_path
        self.writer = None
        self.device = None
        self.early_stopping = None
        self.loss_fn = None

        self.loss_fn = loss_fn
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.early_stopping = EarlyStopping(patience=5)
        self.optimizer = optimizer
        self.scheduler = scheduler

    def configure_optimizer(self):
        return self.optimizer

    def fit(self, model, train_dataloader, val_dataloader):
        return self.fit_epochs(model, self.epochs, train_dataloader, val_dataloader)

    def fit_early_stopping(self, model, train_dataloader, val_dataloader):
        """
        actually train the model with early stopping
        :return:
        """
        optimizer = self.configure_optimizer
        max_epochs = self.epochs
        self.writer = SummaryWriter(log_dir=self.tensorboard_path)
        es = self.early_stopping
        epoch = 1
        model.to(self.device)
        while epoch <= max_epochs:
            loss_train = 0.0
            accuracy_train = 0.0
            model.train()
            for batch in train_dataloader:
                loss, acc = self.training_step(model, batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_train += loss.item()
                accuracy_train += acc
            loss_train = loss_train / len(train_dataloader)
            accuracy_train = accuracy_train / len(train_dataloader)

            if epoch == 1 or epoch % 10 == 0:
                print('{} Epoch {}, Training loss {}, Training acc {}'.format(
                    datetime.datetime.now(), epoch,
                    loss_train, accuracy_train))

            model.eval()
            loss_val = 0.0
            accuracy_val = 0.0
            for batch in val_dataloader:
                loss, acc = self.validation_step(model, batch)
                loss_val += loss.item()
                accuracy_val += acc
            loss_val = loss_val / len(val_dataloader)
            accuracy_val = accuracy_val / len(val_dataloader)
            self.writer.add_scalars("loss", {'train': loss_train, 'validation': loss_val}, epoch)
            self.writer.add_scalars("accuracy", {'train': accuracy_train, 'validation': accuracy_val}, epoch)

            steps_left = self.early_stopping(loss_val)
            if steps_left == -1:
                break
            elif steps_left == 0:
                model_weights = copy.deepcopy(model.state_dict())

            epoch += 1
        model.load_state_dict(model_weights)
        self.writer.close()
        return loss_train, accuracy_train, loss_val, accuracy_val, epoch, model

    def fit_epochs(self, model, epochs, train_dataloader, val_dataloader):
        """
        actually train the model with early stopping
        :return:
        """
        optimizer = self.optimizer
        self.writer = SummaryWriter(log_dir=self.tensorboard_path)

        model.to(self.device)
        for epoch in range(1, epochs+1):
            loss_train = 0.0
            accuracy_train = 0.0
            model.train()
            for batch in train_dataloader:
                loss, acc = self.training_step(model, batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_train += loss.item()
                accuracy_train += acc
            loss_train = loss_train / len(train_dataloader)
            accuracy_train = accuracy_train / len(train_dataloader)

            if epoch == 1 or epoch % 10 == 0:
                print('{} Epoch {}, Training loss {}, Training acc {}'.format(
                    datetime.datetime.now(), epoch,
                    loss_train, accuracy_train))

            model.eval()
            loss_val = 0.0
            accuracy_val = 0.0
            for batch in val_dataloader:
                loss, acc = self.validation_step(model, batch)
                loss_val += loss.item()
                accuracy_val += acc
            loss_val = loss_val / len(val_dataloader)
            accuracy_val = accuracy_val / len(val_dataloader)
            self.writer.add_scalars("loss", {'train': loss_train, 'validation': loss_val}, epoch)
            self.writer.add_scalars("accuracy", {'train': accuracy_train, 'validation': accuracy_val}, epoch)

        self.writer.close()
        return loss_train, accuracy_train, loss_val, accuracy_val, epoch, model

    @abstractmethod
    def compute_loss_and_acc(self, model, batch):
        pass

    def training_step(self, model, batch):
        loss, acc = self.compute_loss_and_acc(model, batch)
        return loss, acc

    def validation_step(self, model, batch):
        loss, acc = self.compute_loss_and_acc(model, batch)
        return loss, acc

    def test(self, model, test_loader):
        model.to(self.device)
        loss_test = 0.0

        list_predicted = []
        list_labels = []
        model.eval()
        for batch in test_loader:
            outputs, loss, labels = self.test_step(model, batch)
            list_labels.append(labels.to('cpu').detach().numpy().copy())
            loss_test += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            list_predicted.append(predicted.to('cpu').detach().numpy().copy())
        loss_test = loss_test / len(test_loader)

        labels = np.concatenate(list_labels)
        predicted = np.concatenate(list_predicted)

        accuracy = accuracy_score(labels, predicted)
        cm = confusion_matrix(labels, predicted)
        precision = precision_score(labels, predicted, average='macro')
        recall = recall_score(labels, predicted, average='macro')
        f1 = f1_score(labels, predicted, average='macro')

        model.to('cpu')
        return loss_test, accuracy, cm, precision, recall, f1

    @abstractmethod
    def test_step(self, model, batch):
        pass