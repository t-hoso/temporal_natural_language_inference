import torch
from sklearn.metrics import accuracy_score

from .trainer import Trainer


class FFNTrainer(Trainer):
    def compute_loss_and_acc(self, model, batch):
        sentences, labels = batch
        sentences = sentences.to(self.device)
        labels = labels.to(self.device)

        outputs = model(sentences)
        loss = self.loss_fn(outputs, labels)
        _, predicted = torch.max(outputs.data, 1)
        acc = accuracy_score(labels.to('cpu').detach().numpy(), predicted.to('cpu').detach().numpy())
        return loss, acc

    def fit(self, model, train_dataloader, val_dataloader):
        return self.fit_epochs(model, self.epochs, train_dataloader, val_dataloader)

    def test_step(self, model, batch):
        sentences, labels = batch
        sentences = sentences.to(self.device)
        labels = labels.to(self.device)

        outputs = model(sentences)
        loss = self.loss_fn(outputs, labels)
        return outputs, loss, labels