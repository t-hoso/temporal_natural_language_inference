import unittest

import torch

from transe.models.loss_function import ConstrainedMarginRankingLoss, l2_distance


class TestConstrainedMarginLoss(unittest.TestCase):
    def setUp(self) -> None:
        self.criterion = ConstrainedMarginRankingLoss(margin=1, epsilon=0, distance_function=l2_distance, constraint_weight=0)

    def tearDown(self) -> None:
        del self.criterion

    def test_forward(self):
        test_cases = [
            [[torch.FloatTensor([[1, 1]]), torch.FloatTensor([[1, 1]]), torch.FloatTensor([[2, 2]]), torch.FloatTensor([[1, 1]])],
            [torch.FloatTensor([[1, 1]]), torch.FloatTensor([[0, 0]]), torch.FloatTensor([[1, 1]]), torch.FloatTensor([[1, 1]])]],
            [[torch.FloatTensor([[1, 1]]), torch.FloatTensor([[1, 1]]), torch.FloatTensor([[2, 2]]), torch.FloatTensor([[1, 1]])],
            [torch.FloatTensor([[0, 0]]), torch.FloatTensor([[0, 0]]), torch.FloatTensor([[1, 0]]), torch.FloatTensor([[1, 1]])]],
            [[torch.FloatTensor([[1, 1]]), torch.FloatTensor([[0, 0]]), torch.FloatTensor([[2, 1]]), torch.FloatTensor([[1, 1]])],
             [torch.FloatTensor([[0, 0]]), torch.FloatTensor([[0, 0]]), torch.FloatTensor([[2, 0]]), torch.FloatTensor([[1, 1]])]]
        ]
        answers = [
            torch.FloatTensor([1]), torch.FloatTensor([0]), torch.FloatTensor([0]),
        ]
        target = torch.FloatTensor([-1])
        for test_case, answer in zip(test_cases, answers):
            triple_positive_array, triple_negative_array = test_case[0], test_case[1]
            loss = self.criterion(triple_positive_array, triple_negative_array, target)
            print(loss, loss.shape)
            self.assertEqual(answer, loss)

