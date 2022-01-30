import torch
import torch.nn.modules.loss as loss
import torch.nn.functional as F


class MarginBasedRankingLoss(loss._Loss):
    def __init__(self, margin, distance_function, size_average=None, reduce=None, reduction: str = 'mean'):
        super(MarginBasedRankingLoss, self).__init__(size_average, reduce, reduction)
        self.margin = margin
        self.distance_function = distance_function

    def forward(self, positive_triple, negative_triple, target):
        distance_of_positive = self.distance_function(positive_triple[0], positive_triple[1], positive_triple[2])
        distance_of_negative = self.distance_function(negative_triple[0], negative_triple[1], negative_triple[2])
        return F.margin_ranking_loss(distance_of_positive, distance_of_negative, target,
                                                 margin=self.margin, reduction=self.reduction)
