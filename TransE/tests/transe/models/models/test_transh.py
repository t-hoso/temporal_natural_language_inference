import unittest

from transe.models.models import(
    SentenceTransformerEncoder, 
    RelationEncoder,
    HyperplaneProjectionLayer, 
    SentenceTransH,
    relation_encoder
)


class TestSentenceTransH(unittest.TestCase):
    def setUp(self):
        sentence_encoder = SentenceTransformerEncoder(model_name='paraphrase-distilroberta-base-v1')
        relation_encoder = RelationEncoder(["isAfter", "isBefore", "HinderedBy"])
        sentence_embedding_dim = 768
        num_relation = 3
        mapped_embedding_dim = 10
        self.model = SentenceTransH(sentence_encoder, relation_encoder, sentence_embedding_dim, num_relation, mapped_embedding_dim)

    def tearDown(self):
        del self.model

    def test_forward(self):
        test_cases = [
            ["A person is doing just fine", "isBefore", "The person gets sick"],
            ["aaa", "isAfter", "bbb"],
            ["aaa", "HinderedBy", "ccc"]
        ]
        print(self.model.relation_encoder.label2number.keys())
        for test_case in test_cases:
            print(test_case)
            out = self.model(test_case)
            print(out[0] + out[1])
            self.assertEqual(len(out), 4)

    def test_get_sentence_embedding(self):
        test_cases = [
            [["This is what it is", "The world is beautiful"], False],
            [["This is what it is", "This is what it is"], True]
        ]
        for test_case in test_cases:
            s1 = self.model.get_sentence_embedding(test_case[0][0])  # get sentence concept embedding
            s2 = self.model.get_sentence_embedding(test_case[0][1])  # get sentence concept embedding
            self.assertEqual(all(s1 == s2), test_case[1])  # if same, should be True, else False
            self.assertEqual(list(s1.shape), [10])
            self.assertEqual(list(s2.shape), [10])
            print("encoded_sentence", s1)

    def test_get_relation_embedding(self):
        test_cases = [
            "isAfter", "isBefore", "HinderedBy"
        ]
        for test_case in test_cases:
            encoded = self.model.get_relation_embedding(test_case)
            print("encoded_relation", encoded)
            self.assertEqual(list(encoded.shape), [1, 10])
