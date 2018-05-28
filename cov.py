import numpy as np

class Covariance(object):

    def __init__(self):
        pass

    def eval(self):
        pass

    def grad(self):
        pass

    def update_param(self):
        pass

    def get_param(self):
        pass

def euclid_dist_mat(x, x_new=None):
    if x_new is None:
        x_new = x

    x = np.repeat(x, x_new.shape[0], axis=1)
    x_new = np.repeat(x_new, x.shape[0], axis=1)
    return x - x_new.T

class squared_exponential(Covariance):

    def __init__(self, eta, l):
        super(squared_exponential, self).__init__()
        self.eta = eta
        self.l = l

    def __call__(self, X, X_new=None):
        self.X = X
        self.X_new = X_new
        self.pair_dist_x = euclid_dist_mat(X, X_new)

        return self

    def eval(self):
        return (self.eta*self.eta) * \
                    np.exp(- (self.pair_dist_x / self.l) * (self.pair_dist_x / self.l) )

    def grad(self):
        pass 

    def get_param(self):
        return eta, l

    def update_param(self, eta, l):
        self.eta = eta
        self.l = l


