import torch
import torch.autograd as autograd
import torch.nn.modules.loss as loss
import torch.nn.functional as F


class ConstrainedMarginRankingLoss(loss._Loss):
    def __init__(self, margin, distance_function, epsilon, constraint_weight, size_average=None, reduce=None, reduction: str = 'mean'):
        super(ConstrainedMarginRankingLoss, self).__init__(size_average, reduce, reduction)
        self.margin = margin
        self.distance_function = distance_function
        self.c = constraint_weight
        self.epsilon = epsilon

    def forward(self, positive, negative, target):
        one = autograd.Variable(torch.FloatTensor([1.0]))
        epsilon = torch.FloatTensor([self.epsilon])
        if positive[0].is_cuda:
            one = autograd.Variable(torch.cuda.FloatTensor([1.0]))
            epsilon = torch.cuda.FloatTensor([self.epsilon])
        distance_of_positive = self.distance_function(positive[0], positive[1], positive[2])
        distance_of_negative = self.distance_function(negative[0], negative[1], negative[2])
        distance_ranking = F.margin_ranking_loss(distance_of_positive, distance_of_negative, target,
                                                 margin=self.margin, reduction=self.reduction)
        embedding_norm = torch.linalg.norm(torch.cat([positive[0], positive[2],
                                                      negative[0], negative[2]])).reshape(1)
        e_loss = F.margin_ranking_loss(embedding_norm, one, target, margin=0, reduction=self.reduction)
        wd = torch.sum(torch.cat([positive[2], negative[2]]) * torch.cat([positive[0], negative[0]]),
                       dim=1, keepdim=True) ** 2
        wd = wd / (torch.linalg.norm(torch.cat([positive[1], negative[1]])).reshape(1) + 1e-10)
        wd = F.margin_ranking_loss(wd, epsilon, target, margin=0, reduction=self.reduction)
        return distance_ranking + self.c * (e_loss + wd)

    