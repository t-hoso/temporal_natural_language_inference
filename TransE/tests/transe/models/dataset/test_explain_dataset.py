import unittest
import os

import torch
from transformers import RobertaTokenizer

from transe.models.dataset import ExplainDataset


ATOMIC_DIRECTORY = os.environ.get("ATOMIC_DIRECTORY")


class TestDataset(unittest.TestCase):
    def setUp(self):
        self.dataset = ExplainDataset()
        self.dataset.data = [[i, i, i] for i in range(6)]
        self.dataset.datanum = 6

    def tearDown(self):
        del self.dataset

    def test_getitem(self):
        tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
        self.dataset.data = [
            ["sentence1", "relation1", "sentence2"],
            ["sent", "relation1", "sent2"]
        ]
        self.dataset.relation_to_number["relation1"] = 20
        answers = [
            (
                torch.LongTensor([0] + tokenizer.encode("sentence1", add_special_tokens=False) + [2]), 
                20, 
                5, 
                torch.LongTensor([0] + tokenizer.encode("sentence2", add_special_tokens=False) + [2]), 
                20, 
                5
            ),
            (
                torch.LongTensor([0] + tokenizer.encode("sent", add_special_tokens=False) + [2]), 
                20, 
                3, 
                torch.LongTensor([0] + tokenizer.encode("sent2", add_special_tokens=False) + [2]), 
                20, 
                4
            )
        ]
        for i, answer in enumerate(answers):
            for d, ans in zip(self.dataset.__getitem__(i)[0], answer):
                print("d", d)
                print("a", ans)
                self.assertTrue(all(d == ans))
