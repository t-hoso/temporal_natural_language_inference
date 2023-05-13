import torch
from sklearn.metrics import accuracy_score

from .trainer import Trainer

class SiameseKnowledgeTrainer(Trainer):
    def compute_loss_and_acc(
        self, model, batch):
        ids1, ids2, labels, str1, str2  = batch
        ids1 = ids1.to(self.device)
        labels = labels.view(-1)
        labels = labels.to(self.device)
        ids2 = ids2.to(self.device)
        outputs = model([
            (ids1, ids2),
            (str1, str2)
        ])
        loss = self.loss_fn(outputs, labels)
        _, predicted = torch.max(outputs.data, 1)
        acc = accuracy_score(labels.to('cpu').detach().numpy(), predicted.to('cpu').detach().numpy())
        return loss, acc

    def test_step(self, model, batch):
        ids1, ids2, labels, str1, str2  = batch
        ids1 = ids1.to(self.device)
        labels = labels.view(-1)
        labels = labels.to(self.device)
        ids2 = ids2.to(self.device)
        outputs = model([
            (ids1, ids2),
            (str1, str2)
        ])
        loss = self.loss_fn(outputs, labels)
        return outputs, loss, labels