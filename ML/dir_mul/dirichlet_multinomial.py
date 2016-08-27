import numpy as np
from scipy.misc import factorial
from ML.utils import gamma
from ML.utils import digamma
from scipy.stats import norm
from scipy.optimize import minimize
import pdb

class DirichletMultinomialRegression:

    def fit(self, X, Y):
        self.X = X
        self.Y = Y

        uniq_labels = Y[0].shape[0]
        x0 = [1] * uniq_labels
        bounds = [[0, None]] * uniq_labels
        res = minimize(self.dirmult_ll, x0, method='l-bfgs-b', jac=self.dirmult_ll_grad, bounds=bounds)
        # res = minimize(self.dirmult_ll, x0, method='l-bfgs-b', bounds=bounds)

        pdb.set_trace()

    def predict(self, X, y):
        pass

    def dirichlet_variance(self, alphas):
        a_0 = np.sum(alphas)
        d_var = (alphas * (a_0 - alphas)) / (a_0**2 * (a_0 + 1))
        return d_var

    def dm_alpha(self, x, weights):
        # return np.array([np.exp(x * w) for w in weights])
        return np.exp(weights)

    def dirmult_ll(self, args):
        w = args
        alpha = self.dm_alpha(self.X, w) # exp {xT.w} ; data * weights
        phi = self.dirichlet_variance(alpha)

        # a = np.sum(                                     # sum_n(
        #     np.log(np.sum(self.Y, axis=1)) -            # log(M_k)-
        #     np.sum(np.log(factorial(self.Y)), axis=1) + # sum (log(C_k!))+
        #     np.log(gamma(np.sum(alpha, axis=0))) -      # log(gamma(sum_k( alpha_k(x_n))))-
        #     np.log(gamma(np.sum(self.Y + alpha)))       # log(gamma(sum_k(C_nk + alpha_k(x_n))))
        # )
        # b = np.sum(
        #     np.log(gamma(self.Y + alpha)) - 
        #     np.log(gamma(alpha))
        # )
        # c = np.sum(
        #     -phi/2 * np.log(2*np.pi*phi) -
        #     0.5 * w.T.dot(phi).dot(w)
        # )

        joint_ll = np.sum(                                # sum_n(
            np.log(np.sum(self.Y, axis=1)) -            # log(M_k)-
            np.sum(np.log(factorial(self.Y)), axis=1) + # sum (log(C_k!))+
            np.log(gamma(np.sum(alpha, axis=0))) -      # log(gamma(sum_k( alpha_k(x_n))))-
            np.log(gamma(np.sum(self.Y + alpha)))       # log(gamma(sum_k(C_nk + alpha_k(x_n))))
        ) + \
        np.sum(
            np.log(gamma(self.Y + alpha)) - 
            np.log(gamma(alpha))
        ) + \
        np.sum(
            -phi/2 * np.log(2*np.pi*phi) -
            0.5 * w.T.dot(phi) * w
        )

        return -joint_ll

    def dirmult_ll_grad(self, args):
        w = args
        alpha = self.dm_alpha(self.X, w)     # (K, ) exp {xT.w} ; data * weights
        phi = self.dirichlet_variance(alpha) # (K, )

        joint_ll_grad = np.sum(
            self.X * alpha * (
                digamma(np.sum(alpha)) -      # Wrong dimensions here?
                digamma(np.sum(self.Y + alpha)) + 
                digamma(np.sum(self.Y + alpha)) -   # Shouldn't be identical to above?
                digamma(alpha)
            )
        ) - 1/alpha*w

        return -joint_ll_grad
