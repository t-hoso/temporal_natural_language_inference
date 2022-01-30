class RelationEncoder:
    """
    Encodes relation
    This works as an adopter for relation
    """
    def __init__(self, labels):
        self.label2number = {label: i for i, label in enumerate(labels)}

    def encode(self, labels):
        """
        encodes the labels
        
        Parameters
        ----------
        labels: list
            list of str
        
        Returns
        -------
        encoded_labels
            list of int
        """
        if type(labels) == str:
            labs = [labels]
        else:
            labs = list(labels)
        return [self.label2number[label] for label in labs]

    def get_relations(self):
        return self.label2number.keys()