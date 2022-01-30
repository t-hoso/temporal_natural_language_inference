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

class ExplainMNLITrainer(Trainer):
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
        start_indices.to('cpu')
        end_indices.to('cpu')
        span_masks.to('cpu')
        input_ids.to('cpu')
        labels = labels.to('cpu')
        predicted = predicted.to('cpu')
        acc = accuracy_score(labels.detach().numpy(), predicted.detach().numpy())
        return loss, acc

    def fit_epochs(self, model, epochs, train_dataloader, val_dataloader):
        """
        actually train the model with early stopping
        :return:
        """
        optimizer, scheduler = self.optimizer
        self.writer = SummaryWriter(log_dir=self.config_dict['tensorboard_path'])
        val_matched_dataloader = val_dataloader[0]
        val_mismatched_dataloader = val_dataloader[1]

        model.to(self.device)
        loss_train, accuracy_train, loss_matched, accuracy_matched, epoch = None, None, None, None, None
        loss_mismatched, accuracy_mismatched = None, None
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
                loss_train += float(loss.to('cpu'))
                accuracy_train += acc
            loss_train = loss_train / len(train_dataloader)
            accuracy_train = accuracy_train / len(train_dataloader)

            if epoch == 1 or epoch % 10 == 0:
                print('{} Epoch {}, Training loss {}, Training acc {}'.format(
                    datetime.datetime.now(), epoch,
                    loss_train, accuracy_train))

            model.eval()
            loss_matched = 0.0
            accuracy_matched = 0.0
            with torch.no_grad():
                for batch in val_matched_dataloader:
                    loss, acc = self.validation_step(model, batch)
                    loss_matched += loss.to('cpu').item()
                    accuracy_matched += acc
            loss_matched = loss_matched / len(val_matched_dataloader)
            accuracy_matched = accuracy_matched / len(val_matched_dataloader)
            loss_mismatched = 0.0
            accuracy_mismatched = 0.0
            with torch.no_grad():
                for batch in val_mismatched_dataloader:
                    loss, acc = self.validation_step(model, batch)
                    loss_mismatched += loss.to('cpu').item()
                    accuracy_mismatched += acc
            loss_mismatched = loss_mismatched / len(val_mismatched_dataloader)
            accuracy_mismatched = accuracy_mismatched / len(val_mismatched_dataloader)
            self.writer.add_scalars("loss", {'train': loss_train, 'matched': loss_matched,
                                             'mimatched': loss_mismatched}, epoch)
            self.writer.add_scalars("accuracy", {'train': accuracy_train, 'matched': accuracy_matched,
                                                 'mismatched': accuracy_mismatched}, epoch)

        self.writer.close()
        return loss_train, accuracy_train, loss_matched, accuracy_matched,\
               loss_mismatched, accuracy_mismatched, epoch, model

    def fit(self, model, train_dataloader, val_dataloader):
        return self.fit_epochs(model, self.epochs, train_dataloader, val_dataloader)

    def training_step(self, model, batch):
        loss, acc = self.compute_loss_and_acc(model, batch)
        return loss, acc

    def validation_step(self, model, batch):
        loss, acc = self.compute_loss_and_acc(model, batch)
        return loss, acc

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
        ce_loss = self.loss_fn_base(outputs, labels)
        loss = self.loss_fn(ce_loss, a_ij)
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
        ids = []
        for i in list_ids:
            ids.extend(i.tolist())
        ids = np.array(ids)
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

        wrong_sentence1 = None
        wrong_sentence2 = None
        original_label = None
        predicted_label = None

        model.to('cpu')
        return loss_test, accuracy, cm, precision, recall, f1,\
               wrong_sentence1, wrong_sentence2, original_label, predicted_label
