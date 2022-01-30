import torch
from sklearn.metrics import accuracy_score

from .explain_trainer import ExplainTrainer


class ExplainKnowledgeTrainer(ExplainTrainer):
    def compute_loss_and_acc(self, model, batch):
        input_ids, labels, length, start_indices, \
            end_indices, span_masks, str1, str2 = batch
        input_ids = input_ids.to(self.device)
        labels = labels.view(-1)
        labels = labels.to(self.device)
        start_indices = start_indices.to(self.device)
        end_indices = end_indices.to(self.device)
        span_masks = span_masks.to(self.device)
        outputs, a_ij = model([
            (input_ids, start_indices, end_indices, span_masks),
            (str1, str2)])
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
    
    def test_step(self, model, batch):
        input_ids, labels, length, start_indices, \
            end_indices, span_masks, str1, str2 = batch
        input_ids = input_ids.to(self.device)
        labels = labels.view(-1)
        labels = labels.to(self.device)
        length = length.to(self.device)
        start_indices = start_indices.to(self.device)
        end_indices = end_indices.to(self.device)
        span_masks = span_masks.to(self.device)
        outputs, a_ij = model([
            (input_ids, start_indices, end_indices, span_masks),
            (str1, str2)])
        loss = self.loss_fn([outputs, a_ij], labels.reshape(-1))
        _, predicted = torch.max(outputs.data, 1)
        return outputs, loss, labels, a_ij