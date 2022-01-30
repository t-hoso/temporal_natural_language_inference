import datetime
import torch
from sklearn.metrics import accuracy_score
from torch.utils.tensorboard import SummaryWriter

from .trainer import Trainer


class SiameseMNLITrainer(Trainer):
    def compute_loss_and_acc(self, model, batch):
        sentence1, sentence2, labels = batch
        sentence1 = sentence1.to(self.device)
        sentence2 = sentence2.to(self.device)
        labels = labels.to(self.device)

        outputs = model([sentence1, sentence2])
        loss = self.loss_fn(outputs, labels)
        _, predicted = torch.max(outputs.data, 1)
        acc = accuracy_score(labels.to('cpu').detach().numpy(), predicted.to('cpu').detach().numpy())
        return loss, acc

    def fit_epochs(self, model, epochs, train_dataloader, val_dataloader):
        """
        actually train the model with early stopping
        :return:
        """
        optimizer = self.optimizer
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
                                             'mismatched': loss_mismatched}, epoch)
            self.writer.add_scalars("accuracy", {'train': accuracy_train, 'matched': accuracy_matched,
                                                 'mismatched': accuracy_mismatched}, epoch)

        self.writer.close()
        return loss_train, accuracy_train, loss_matched, accuracy_matched,\
               loss_mismatched, accuracy_mismatched, epoch, model

    def fit(self, model, train_dataloader, val_dataloader):
        return self.fit_epochs(model, self.config_dict['epochs'], train_dataloader, val_dataloader)

    def test_step(self, model, batch):
        sentence1, sentence2, labels = batch
        sentence1 = sentence1.to(self.device)
        sentence2 = sentence2.to(self.device)
        labels = labels.to(self.device)

        outputs = model([sentence1, sentence2])
        loss = self.loss_fn(outputs, labels)
        return outputs, loss, labels