from sentence_transformers import SentenceTransformer


class SentenceTransformerEncoder:
    """
    Encodes sentence
    This works as an adopter for SentenceTransE
    """
    def __init__(self, model_name):
        self.model = SentenceTransformer(model_name)

    def encode(self, sentences):
        """
        Encodes list of str
        
        Parameters
        ----------
        sentences: list
            list of str
        
        Returns
        -------
        encoded_sentences
        """
        encoded_sentences = self.model.encode(sentences, show_progress_bar=False)
        return encoded_sentences
