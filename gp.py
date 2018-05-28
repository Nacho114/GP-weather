import numpy as np
from numpy.linalg import cholesky, inv
from scipy.linalg import solve_triangular

def identity_like(x):
    return np.eye(x.shape[0])

def stabilize(K):
    """ adds small diagonal to a covariance matrix """
    return K + 1e-6 * identity_like(K)


# reutrn the likelyhood to find the MAP

# Add stabilize to Kxx?
def fast_cond(X_new, X, f, cov_func, sigma_true=0):
    """ posterior of GP using chilesky factorisation for faster computation"""
    Kxx = cov_func(X).eval()
    Ksx = cov_func(X_new, X).eval()
    Kss = cov_func(X_new).eval()
    L = cholesky(Kxx + sigma_true * np.eye(Kxx.shape[0]))

    beta = solve_triangular(L, f, lower=True)
    alpha = solve_triangular(L.T, beta, lower=False)

    mu = np.dot(Ksx, alpha)

    v = solve_triangular(L, Ksx.T, lower=True)

    cov = Kss - np.dot(v.T, v)

    return mu, cov

def cond(X_new, X, f, cov_func, sigma_true=0):
    """ posterior of GP """
    print('Use fast_cond for better performance.')
    Kxx = cov_func(X).eval()
    Ksx = cov_func(X_new, X).eval()
    Kss = cov_func(X_new).eval()
    Kxx_inv = inv(Kxx + sigma_true * np.eye(Kxx.shape[0]))

    mu = np.matmul(Ksx, np.dot(Kxx_inv, f))
    cov = Kss - np.matmul(Ksx, np.matmul(Kxx_inv, Ksx.T))

    return mu, cov

def sample_mv(mu_s, cov_s):
    """ Sample from a normal multivariate, add some small noise to covariance for stability """
    return np.random.multivariate_normal(mu_s, cov_s + 1e-8*np.eye(cov_s.shape[0]), 1).flatten()

# def get_Matern52(theta):
#     l, eta = theta[0], theta[1]
#     return eta**2 * pm.gp.cov.Matern52(1, l)



#    