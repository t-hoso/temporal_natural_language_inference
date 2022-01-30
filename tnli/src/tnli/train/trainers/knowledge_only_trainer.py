import torch
from sklearn.metrics import accuracy_score

from .trainer import Trainer


class KnowledgeOnlyTrainer(Trainer):
    def compute_loss_and_acc(self, model, batch):
        sentence1, sentence2, labels = batch
        labels = labels.to(self.device)

        outputs = model([sentence1, sentence2])
        loss = self.loss_fn(outputs, labels.reshape(-1))
        _, predicted = torch.max(outputs.data, 1)
        acc = accuracy_score(labels.to('cpu').detach().numpy(), predicted.to('cpu').detach().numpy())
        return loss, acc

    def test_step(self, model, batch):
        sentence1, sentence2, labels = batch
        labels = labels.to(self.device)

        outputs = model([sentence1, sentence2])
        loss = self.loss_fn(outputs, labels.reshape(-1))
        return outputs, loss, labels

