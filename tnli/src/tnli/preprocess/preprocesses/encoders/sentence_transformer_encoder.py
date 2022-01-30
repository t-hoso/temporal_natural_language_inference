from sentence_transformers import SentenceTransformer

from .encoder import Encoder


class SentenceTransformersEncoder(Encoder):
    def __init__(self, model='paraphrase-distilroberta-base-v1'):
        self.model = SentenceTransformer(model)

    def encode_one(self, sentence):
        sentence_embeddings = self.model.encode([sentence], show_progress_bar=False)
        return sentence_embeddings[0]

    def encode_many(self, sentences):
        """
        encode a list of sentences
        :param sentences: list of str
        :return: embedding

        >>> encoder = SentenceTransformersEncoder()
        >>> sentences = ["A white dog running through the snow", "A white dog sleeping on the snow", "The sun feels very hot on the girls' heads." ]
        >>> print(encoder.encode_many(sentences))
        >>> print(len(encoder.encode_many(sentences)[0]))
        36
        """
        sentence_embeddings = self.model.encode(sentences, show_progress_bar=False)
        return sentence_embeddings

