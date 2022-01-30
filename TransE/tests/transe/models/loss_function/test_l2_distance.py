import unittest

import torch

from transe.models.loss_function import l2_distance


class TestDissimilarityFunction(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_dissimilarity_function(self):
        test_cases = [
            [torch.FloatTensor([[1, 1]]), torch.FloatTensor([[1, 1]]), torch.FloatTensor([[2, 2]])],
            [torch.FloatTensor([[1, 1]]), torch.FloatTensor([0, 0]), torch.FloatTensor([[1, 1]])],
            [torch.FloatTensor([[1, 1]]), torch.FloatTensor([[0, 0]]), torch.FloatTensor([[2, 1]])],
            [torch.FloatTensor([[0, 0]]), torch.FloatTensor([[0, 0]]), torch.FloatTensor([[1, 0]])],
            [torch.FloatTensor([[0, 0]]), torch.FloatTensor([[0, 0]]), torch.FloatTensor([[2, 0]])],
        ]
        answers = [
            0, 0, 1, 1, 2
        ]
        for test_case, answer in zip(test_cases, answers):
            self.assertEqual(l2_distance(*test_case), answer)