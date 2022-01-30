import unittest
import os

import torch

from transe.models.dataset.collate_fn.collate_four_sentences import collate_four_sentences


ATOMIC_DIRECTORY = os.environ.get("ATOMIC_DIRECTORY")


class TestCollateFourSentences(unittest.TestCase):
    def setUp(self):
        input_id1 = torch.LongTensor([0, 3, 2, 5, 6, 2])
        input_id2 = torch.LongTensor([0, 3, 2, 4, 2])
        input_id3 = torch.LongTensor([0, 3, 2])
        self.batch = (
            (
                (
                    input_id1, torch.LongTensor([1]), torch.LongTensor([6]),
                    input_id1, torch.LongTensor([1]), torch.LongTensor([6])
                ),
                (
                    input_id2, torch.LongTensor([1]), torch.LongTensor([5]),
                    input_id2, torch.LongTensor([1]), torch.LongTensor([5])
                ),
                (
                    input_id3, torch.LongTensor([1]), torch.LongTensor([3]),
                    input_id3, torch.LongTensor([1]), torch.LongTensor([3])
                ),
            ),
            (
                (
                    input_id1, torch.LongTensor([1]), torch.LongTensor([6]),
                    input_id1, torch.LongTensor([1]), torch.LongTensor([6])
                ),
                (
                    input_id2, torch.LongTensor([1]), torch.LongTensor([5]),
                    input_id2, torch.LongTensor([1]), torch.LongTensor([5])
                ),
                (
                    input_id3, torch.LongTensor([1]), torch.LongTensor([3]),
                    input_id3, torch.LongTensor([1]), torch.LongTensor([3])
                ),
            ),
        )
        
    def test_collate_four_sentences(self):
        output = collate_four_sentences(batch=self.batch, fill_values=[1, 0, 0])
        print(output)
        positive, negative = output
        sentence1, sentence2 = positive
        input_ids, labels, lengths, start_indices, end_indices, span_masks = sentence1
        self.assertTrue(
            all(
                input_ids.flatten()
                == torch.LongTensor(
                    [[0, 3, 2, 5, 6, 2],
                    [0, 3, 2, 4, 2, 1],
                    [0, 3, 2, 1, 1, 1]]).flatten()
            )
        )
        self.assertTrue(
            all(
                labels.flatten()
                == torch.LongTensor(
                    [[1],
                    [1],
                    [1]]).flatten()
            )
        )
        self.assertTrue(
            all(
                lengths.flatten()
                == torch.LongTensor(
                    [[6],
                    [5],
                    [3]]).flatten()
            )
        )
        self.assertTrue(
            all(
                start_indices
                == torch.LongTensor([1, 1, 1, 1, 2, 2, 2, 3, 3, 4])
            )
        )
        self.assertTrue(
            all(
                end_indices
                == torch.LongTensor([1, 2, 3, 4, 2, 3, 4, 3, 4, 4])
            )
        )
        self.assertTrue(
            all(
                span_masks.flatten()
                == torch.LongTensor(
                    [[0, 1000000, 1000000, 1000000, 1000000, 1000000, 1000000, 0, 0, 0],
                    [0, 1000000, 1000000, 1000000, 1000000, 1000000, 1000000, 0,1000000, 1000000],
                    [0, 1000000, 1000000, 1000000, 1000000, 1000000, 1000000, 1000000, 1000000, 1000000]]).flatten()
            )
        )
