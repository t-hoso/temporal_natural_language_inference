import unittest

from transe.models.models import SentenceComplEx, SentenceTransformerEncoder, RelationEncoder


class TestComplEx(unittest.TestCase):
    def setUp(self):
        sentence_encoder = SentenceTransformerEncoder(model_name='paraphrase-distilroberta-base-v1')
        relation_encoder = RelationEncoder(["isAfter", "isBefore", "HinderedBy"])
        sentence_embedding_dim = 768
        num_relation = 3
        mapped_embedding_dim = 10
        self.model = SentenceComplEx(sentence_encoder, relation_encoder, sentence_embedding_dim, num_relation, mapped_embedding_dim)

    def tearDown(self):
        del self.model

    def test_forward(self):
        test_cases = [
            [["A person is doing just fine"], ["isBefore"], ["The person gets sick"]],
            ["aaa", "isAfter", "bbb"],
            ["aaa", "HinderedBy", "ccc"],
            [["aaa", "aaa"],
             ["isAfter","HinderedBy"],
             ["bbb", "ccc"]]
        ]
        print(self.model.relation_encoder.label2number.keys())
        for test_case in test_cases[:-1]:
            print(test_case)
            out = self.model(test_case)
            print(out.shape)
            print(out)
            self.assertEqual(out.shape[0], 1)
        self.assertEqual(self.model(test_cases[-1]).shape[0], 2)

