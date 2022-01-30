import unittest

import torch
from Self_Explaining_Structures_Improve_NLP_Models.datasets.collate_functions import collate_to_max_length

from transe.models.models import SelfExplainingTransE
from transe.models.models.layers.explainable_base import ExplainableBase


class TestSentenceTransE(unittest.TestCase):
    def setUp(self):
        self.model = SelfExplainingTransE(
            ExplainableBase("roberta-base"),
            num_relation=3,
        )

    def tearDown(self):
        del self.model

    def test_forward(self):
        input_id1 = torch.LongTensor([0, 3, 2, 4, 5, 2])
        input_id2 = torch.LongTensor([0, 3, 2, 4, 2])
        input_id3 = torch.LongTensor([0, 3, 2])
        batch = [(input_id1, torch.LongTensor([1]), torch.LongTensor([6])),
            (input_id2, torch.LongTensor([1]), torch.LongTensor([5])),
            (input_id3, torch.LongTensor([1]), torch.LongTensor([3]))]


        processed_batch = collate_to_max_length(batch, fill_values=[1, 0, 0])
        output = self.model((processed_batch, processed_batch))
        self.assertEqual(
            len(output),
            3
        )
        self.assertEqual(
            len(output[0]),
            3
        )
        print(output[0])
        self.assertEqual(
            len(output[0][0]),
            768
        )
        self.assertEqual(
            len(output[1][0]),
            768
        )
        self.assertEqual(
            len(output[2][0]),
            768
        )
        self.assertEqual(
            all(output[0][0] == output[2][0])
        )