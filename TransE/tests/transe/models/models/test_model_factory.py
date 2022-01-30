import unittest

from transe.models.models import (
    ModelFactory, 
    SentenceTransE,
    SentenceComplEx,
    SentenceTransH,
    SelfExplainingTransE
)


class TestModelFactory(unittest.TestCase):
    def setUp(self):
        pass
    
    def tearDown(self):
        pass
    
    def test_create_instance(self):
        self.assertTrue(
            isinstance(
                ModelFactory.create_instance(
                    "transe",
                    128,
                ), 
                SentenceTransE
            )
        )
        self.assertTrue(
            isinstance(
                ModelFactory.create_instance(
                    "transh",
                    128,
                ), 
                SentenceTransH
            )
        )
        self.assertTrue(
            isinstance(
                ModelFactory.create_instance(
                    "complex",
                    128,
                ), 
                SentenceComplEx
            )
        )
        self.assertEqual(
            ModelFactory.create_instance(
                "",
                128,
            ), 
            None
        )
        self.assertTrue(
            isinstance(
                ModelFactory.create_instance(
                    "explain_transe",
                    None
                ),
                SelfExplainingTransE
            )
        )

