import datetime
import numpy as np
import torch
from sklearn.metrics import accuracy_score
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
import torch
from Self_Explaining_Structures_Improve_NLP_Models.datasets.collate_functions import collate_to_max_length

from .trainer import Trainer

class ExplainTrainer(Trainer):
    def compute_loss_and_acc(self, model, batch):
        input_ids, labels, length, start_indices, end_indices, span_masks = batch
        input_ids = input_ids.to(self.device)
        labels = labels.view(-1)
        labels = labels.to(self.device)
        start_indices = start_indices.to(self.device)
        end_indices = end_indices.to(self.device)
        span_masks = span_masks.to(self.device)
        outputs, a_ij = model(input_ids, start_indices, end_indices, span_masks)
        loss  = self.loss_fn([outputs, a_ij], labels.reshape(-1))
        _, predicted = torch.max(outputs.data, 1)
        del start_indices
        del end_indices
        del span_masks
        del input_ids
        labels = labels.to('cpu')
        predicted = predicted.to('cpu')
        acc = accuracy_score(labels.detach().numpy(), predicted.detach().numpy())
        return loss, acc

    def fit_epochs(self, model, epochs, train_dataloader, val_dataloader):
        """
        actually train the model with early stopping
        :return:
        """
        optimizer = self.optimizer
        scheduler = self.scheduler
        self.writer = SummaryWriter(log_dir=self.tensorboard_path)

        model.to(self.device)
        loss_train, accuracy_train, loss_val, accuracy_val, epoch = None, None, None, None, None
        for epoch in range(1, epochs+1):
            loss_train = 0.0
            accuracy_train = 0.0
            model.train()
            for batch in train_dataloader:
                loss, acc = self.training_step(model, batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
                loss_train += float(loss.to('cpu').detach())
                del loss
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
            with torch.no_grad():
                for batch in val_dataloader:
                    loss, acc = self.validation_step(model, batch)
                    loss_val += loss.to('cpu').item()
                    accuracy_val += acc
            loss_val = loss_val / len(val_dataloader)
            accuracy_val = accuracy_val / len(val_dataloader)
            self.writer.add_scalars("loss", {'train': loss_train, 'validation': loss_val}, epoch)
            self.writer.add_scalars("accuracy", {'train': accuracy_train, 'validation': accuracy_val}, epoch)

        self.writer.close()
        return loss_train, accuracy_train, loss_val, accuracy_val, epoch, model

    def fit(self, model, train_dataloader, val_dataloader):
        return self.fit_epochs(model, self.epochs, train_dataloader, val_dataloader)

    def test_step(self, model, batch):
        input_ids, labels, length, start_indices, end_indices, span_masks = batch
        input_ids = input_ids.to(self.device)
        labels = labels.view(-1)
        labels = labels.to(self.device)
        length = length.to(self.device)
        start_indices = start_indices.to(self.device)
        end_indices = end_indices.to(self.device)
        span_masks = span_masks.to(self.device)
        outputs, a_ij = model(input_ids, start_indices, end_indices, span_masks)
        loss = self.loss_fn([outputs, a_ij], labels.reshape(-1))
        _, predicted = torch.max(outputs.data, 1)
        return outputs, loss, labels, a_ij

    def test(self, model, test_loader):
        model.to(self.device)
        loss_test = 0.0

        list_predicted = []
        list_labels = []
        list_ids = []
        list_aij = []
        model.eval()
        for batch in test_loader:
            outputs, loss, labels, a_ij = self.test_step(model, batch)
            list_labels.append(labels.to('cpu').detach().numpy().copy())
            list_ids.append(batch[0].to('cpu').detach().numpy().copy())
            loss_test += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            list_predicted.append(predicted.to('cpu').detach().numpy().copy())
            list_aij.append(a_ij.to('cpu').detach().numpy())
        loss_test = loss_test / len(test_loader)

        labels = np.concatenate(list_labels)
        print(list_ids[0].shape, list_ids[1].shape)
        ids = []
        for i in list_ids:
            ids.extend(i.tolist())
        ids = np.array(ids)
        print(ids.shape)
        a_ij = []
        for i in list_aij:
            a_ij.extend(i.tolist())
        a_ij = np.array(a_ij)
        predicted = np.concatenate(list_predicted)
        accuracy = accuracy_score(labels, predicted)
        cm = confusion_matrix(labels, predicted)
        precision = precision_score(labels, predicted, average='macro')
        recall = recall_score(labels, predicted, average='macro')
        f1 = f1_score(labels, predicted, average='macro')

        model.to('cpu')
        return loss_test, accuracy, cm, precision, recall, f1