import unittest

from Self_Explaining_Structures_Improve_NLP_Models.datasets.collate_functions import collate_to_max_length
import torch

from transe.models.models.layers.explainable_base import ExplainableBase


class TestExplainableBase(unittest.TestCase):
    def setUp(self):
        self.model = ExplainableBase("roberta-base")

    def tearDown(self):
        del self.model

    def test_forward(self):
        input_id1 = torch.LongTensor([0, 3, 2, 4, 5, 2])
        input_id2 = torch.LongTensor([0, 3, 2, 4, 2])
        input_id3 = torch.LongTensor([0, 3, 2])
        batch = [(input_id1, torch.LongTensor([1]), torch.LongTensor([6])),
            (input_id2, torch.LongTensor([1]), torch.LongTensor([5])),
            (input_id3, torch.LongTensor([1]), torch.LongTensor([3]))]


        input_ids, _, _, start_indices, end_indices, span_masks = collate_to_max_length(batch, fill_values=[1, 0, 0])
        out = self.model(input_ids, start_indices, end_indices, span_masks)
        self.assertEqual(len(out[0]), 768)
