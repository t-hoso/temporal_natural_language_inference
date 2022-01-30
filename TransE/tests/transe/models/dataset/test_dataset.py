import unittest
import os

from transe.models.dataset import Atomic2020Dataset, load_atomic_data


ATOMIC_DIRECTORY = os.environ.get("ATOMIC_DIRECTORY")


class TestDataset(unittest.TestCase):
    def setUp(self):
        self.dataset = Atomic2020Dataset()
        self.dataset.data = [[i, i, i] for i in range(6)]
        self.dataset.datanum = 6

    def tearDown(self):
        del self.dataset

    def testCorruptSample(self):
        self.dataset.nodes = list(range(10000))
        sampled_1 = self.dataset._corrupt_sample()
        sampled_2 = self.dataset._corrupt_sample()
        self.assertEqual(type(sampled_1), int)
        print("These should be different in high probability", sampled_1, sampled_2)
        cnt = 0
        for _ in range(1000):
            sampled_1 = self.dataset._corrupt_sample()
            sampled_2 = self.dataset._corrupt_sample()
            if sampled_1 == sampled_2:
                cnt += 1
        self.assertNotEqual(1, cnt // 1000)

    def testCorrupt(self):
        self.dataset.nodes = list(range(10000))
        self.dataset.data = [[i, i, i] for i in range(10000)]
        sampled = self.dataset._corrupt(0)
        print("This tuple should be corrupted", sampled)


    def test_getitem(self):
        self.dataset.data = [
            ["sentence1", "relation1", "sentence2"],
            ["sent", "relation1", "sent"]
        ]
        answers = [
            ("sentence1", "relation1", "sentence2"),
            ("sent", "relation1", "sent")
        ]
        for i, answer in enumerate(answers):
            self.assertEqual(self.dataset.__getitem__(i)[:3], answer)


class TestLoad(unittest.TestCase):
    def test_load_atomic_data(self):
        filepath = os.path.join(ATOMIC_DIRECTORY, "train.tsv")
        data = load_atomic_data(filepath)
        self.assertEqual(3, len(data[0]))
        self.assertEqual(str, type(data[0][0]))
        self.assertEqual(str, type(data[0][1]))
        self.assertEqual(str, type(data[0][2]))
        filepath = os.path.join(ATOMIC_DIRECTORY, "test.tsv")
        data = load_atomic_data(filepath)
        self.assertEqual(3, len(data[0]))
        self.assertEqual(str, type(data[0][0]))
        self.assertEqual(str, type(data[0][1]))
        self.assertEqual(str, type(data[0][2]))
        filepath = os.path.join(ATOMIC_DIRECTORY, "dev.tsv")
        data = load_atomic_data(filepath)
        self.assertEqual(3, len(data[0]))
        self.assertEqual(str, type(data[0][0]))
        self.assertEqual(str, type(data[0][1]))
        self.assertEqual(str, type(data[0][2]))

