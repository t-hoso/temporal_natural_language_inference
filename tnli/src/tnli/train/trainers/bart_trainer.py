import datetime
import torch
from sklearn.metrics import accuracy_score
from torch.utils.tensorboard import SummaryWriter

import sys
sys.path.append(".")

from .trainer import Trainer


class BartTrainer(Trainer):
    def compute_loss_and_acc(self, model, batch):
        """
        :param model:
        :param batch:
        :return:
        """
        print(batch)
        input_ids, masks, labels = batch
        input_ids = input_ids.to(self.device)
        masks = input_ids.to(self.device)
        labels = labels.to(self.device)

        outputs = model(input_ids,
                              attention_mask=masks,
                              labels=labels)
        loss = outputs.loss
        logits = outputs.logits
        _, predicted = torch.max(logits.data, 1)
        acc = accuracy_score(labels.to('cpu').detach().numpy(), predicted.to('cpu').detach().numpy())
        return loss, acc

    def fit_epochs(self, model, epochs, train_dataloader, val_dataloader):
    #     """
    #     actually train the model with early stopping
    #     :return:
    #     """
    #     optimizer, scheduler = self.configure_optimizer()
    #     self.writer = SummaryWriter(log_dir=self.config_dict['tensorboard_path'])

    #     model.to(self.device)
    #     loss_train, accuracy_train, loss_val, accuracy_val, epoch = None, None, None, None, None
    #     for epoch in range(1, epochs+1):
    #         loss_train = 0.0
    #         accuracy_train = 0.0
    #         model.train()
    #         for batch in train_dataloader:
    #             loss, acc = self.training_step(model, batch)
    #             optimizer.zero_grad()
    #             loss.backward()
    #             optimizer.step()
    #             scheduler.step()
    #             loss_train += loss.item()
    #             accuracy_train += acc
    #         loss_train = loss_train / len(train_dataloader)
    #         accuracy_train = accuracy_train / len(train_dataloader)

    #         if epoch == 1 or epoch % 10 == 0:
    #             print('{} Epoch {}, Training loss {}, Training acc {}'.format(
    #                 datetime.datetime.now(), epoch,
    #                 loss_train, accuracy_train))

    #         model.eval()
    #         loss_val = 0.0
    #         accuracy_val = 0.0
    #         for batch in val_dataloader:
    #             loss, acc = self.validation_step(model, batch)
    #             loss_val += loss.item()
    #             accuracy_val += acc
    #         loss_val = loss_val / len(val_dataloader)
    #         accuracy_val = accuracy_val / len(val_dataloader)
    #         self.writer.add_scalars("loss", {'train': loss_train, 'validation': loss_val}, epoch)
    #         self.writer.add_scalars("accuracy", {'train': accuracy_train, 'validation': accuracy_val}, epoch)

    #     self.writer.close()
    #     return loss_train, accuracy_train, loss_val, accuracy_val, epoch, model
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

    def fit(self, model, train_dataloader, val_dataloader):
        return self.fit_epochs(model, self.epochs, train_dataloader, val_dataloader)

    def test_step(self, model, batch):
        input_ids, masks, labels = batch
        input_ids = input_ids.to(self.device)
        masks = input_ids.to(self.device)
        labels = labels.to(self.device)

        outputs = model(input_ids,
                              attention_mask=masks,
                              labels=labels)
        return outputs.logits, outputs.loss, labels
