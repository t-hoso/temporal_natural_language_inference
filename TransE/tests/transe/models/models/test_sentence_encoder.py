import unittest

from transe.models.models import SentenceTransformerEncoder

class TestSentenceEncoder(unittest.TestCase):
    def setUp(self):
        self.encoder = SentenceTransformerEncoder(model_name='paraphrase-distilroberta-base-v1')

    def tearDown(self):
        del self.encoder

    def test_encode(self):
        test_cases = [
            [["This is a pen", "I am building a new stadium"],
             ["I am building a new stadium", "This is a pen"], False],
            [["This is a pen", "I am building a new stadium"],
             ["This is a pen", "I am building a new stadium"], True]
        ]
        for case in test_cases:
            sentence1 = case[0]
            sentence2 = case[1]
            result = case[2]
            encoded_sentence1 = self.encoder.encode(sentence1)
            encoded_sentence2 = self.encoder.encode(sentence2)
            for es1, es2 in zip(encoded_sentence1, encoded_sentence2):
                self.assertEqual(all(es1 == es2), result)

