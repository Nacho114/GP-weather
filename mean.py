import numpy as np


class Mean(object):
    """
    Base class for mean functions
    """

    def __init__(self):
        pass

    def eval(self):
        pass

    def update_param(self):
        pass

    def get_param(self):
        pass

class Zero(Mean):

    def __call__(self, X):
        self.X = X
        return self

    def eval(self):
        return np.zeros(self.X.shape[0])

