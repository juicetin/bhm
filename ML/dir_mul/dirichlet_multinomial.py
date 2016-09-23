import numpy as np
from scipy.misc import factorial
from ML.utils import gamma
from ML.utils import digamma
from ML.utils import gammaln
from scipy.stats import norm
from scipy.optimize import minimize
import pdb

class DirichletMultinomialRegression:
    def __init__(self, reg=1):
        self.phi = reg # Variance of the weights - regulariser

    def fit(self, X, C):

        self.X = X
        self.C = C.astype(np.float64)
        self.D = X.shape[1]
        self.K = C.shape[1]

        uniq_labels = C[0].shape[0]
        x0 = np.ones(uniq_labels * X.shape[1])              # weights - KxD
        # bounds = [[0, None]] * (uniq_labels * X.shape[1]) # do we want positive params?
        res = minimize(self.dirmult_ll, x0, method='l-bfgs-b', jac=True)
        
        self.res = res
        self.W = res['x'].reshape(self.K, self.D)
        return self

    def predict(self, x, counts=1):
        alpha = self.dm_alpha(x, self.W)
        return counts * (alpha/alpha.sum(axis=1)[:,np.newaxis])

    # x = (N, D)
    # w = (K, D)
    def dm_alpha(self, X, W):
        # return np.array([np.exp(x * w) for w in weights])
        return np.exp(X.dot(W.T))

    def dirmult_ll(self, args):
        W = args.reshape(self.K, self.D)    # w has dimensions KxD!
        alpha = self.dm_alpha(self.X, W) # exp {xT.w} ; alpha_k per k (class) per data point

        C_alpha = self.C + alpha
        alpha_sum = alpha.sum(axis=1)
        C_alpha_sum = C_alpha.sum(axis=1)
        C_sum = self.C.sum(axis=1); C_sum[C_sum == 0] = 1e-10

        # MAP
        joint_ll = (np.sum(                             # sum_n(
            np.log(C_sum) -                             #   log(M_k)                    # infinites here! - some points aren't labelled anything...?
            (np.log(factorial(self.C))).sum(axis=1) +   #   - sum (log(C_k!))
            gammaln(alpha_sum) -                        #   + log(gamma(sum_k( alpha_k(x_n))))
            gammaln(C_alpha_sum)                        #   - log(gamma(sum_k(C_nk + alpha_k(x_n))))
        ) +                                             # ) +
        np.sum(                                         # sum_n sum_k (
            gammaln(C_alpha) -                          #   log(gamma(C_k + alpha))
            gammaln(alpha)                              #   - log(gamma(alpha_k(x_n)))
        ) +                                             # ) +
        np.sum(                                         # sum_k (
            -self.phi/2 * np.log(2*np.pi*self.phi) -    #   -phi/2 * log(2*pi*phi)
            0.5 * (W**2) / self.phi                     #   - 1/2 * w_kT * (phi*I) * w)k
        ))                                              # )

        # Grad
        joint_ll_grad = (alpha * (
            (digamma(alpha_sum) -
                digamma(C_alpha_sum))[: , np.newaxis] +
            digamma(C_alpha) -
            digamma(alpha)
        )).T.dot(self.X) - 1/self.phi * W

        return [-joint_ll, -joint_ll_grad.flatten()]
