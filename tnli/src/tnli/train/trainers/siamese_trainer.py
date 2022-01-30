import torch
from sklearn.metrics import accuracy_score

from .trainer import Trainer


class SiameseTrainer(Trainer):
    def compute_loss_and_acc(self, model, batch):
        sentence1, sentence2, labels = batch
        print(sentence1.shape, sentence2.shape, labels.shape)

        sentence1 = sentence1.to(self.device)
        sentence2 = sentence2.to(self.device)
        labels = labels.to(self.device).reshape(-1)

        outputs = model([sentence1, sentence2])
        loss = self.loss_fn(outputs, labels)
        _, predicted = torch.max(outputs.data, 1)
        acc = accuracy_score(labels.to('cpu').detach().numpy(), predicted.to('cpu').detach().numpy())
        return loss, acc

    def test_step(self, model, batch):
        sentence1, sentence2, labels = batch
        sentence1 = sentence1.to(self.device)
        sentence2 = sentence2.to(self.device)
        labels = labels.to(self.device)

        outputs = model([sentence1, sentence2])
        loss = self.loss_fn(outputs, labels)
        return outputs, loss, labels

