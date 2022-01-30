class EarlyStopping:
    """
    Early Stopping
    """
    def __init__(self, patience: int=0):
        """
        :param patience: how many times to wait before stop learning
        """
        self.step = 0
        self._loss = float('inf')
        self.patience = patience

    def __call__(self, loss: float):
        """
        Whether to stop the training or not.
        :param loss: the current validation loss
        :return: bool, whether to stop or not

        >>> es = EarlyStopping(patience=0)
        >>> es(0.3)
        0
        >>> es(0.4)
        -1

        >>> es = EarlyStopping()
        >>> es(0.2)
        0
        >>> es(0.1)
        0
        """
        if self._loss < loss:
            self._step += 1
        else:
            self._step = 0
            self._loss = loss
        return self.patience - self._step


