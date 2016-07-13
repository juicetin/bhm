import numpy as np
from numpy import linalg
from sympy import KroneckerDelta
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from scipy.stats.norm import cdf
from scipy.stats.norm import pdf
import math

# TODO when doing K(X, X), simply use (n_err^2)I instead of KroneckerDelta

# using http://www.robots.ox.ac.uk/~mebden/reports/GPtutorial.pdf

# Example
# a = np.array([1,2,3,4,5])
# a = a.reshape(5,1) # 5 rows, 1 column
# b = np.array([6,7,8])
# b = np.reshape(1,3) # 1 row, 3 columns
# print(a.dot(b))

class GaussianProcess:
    def __init__(self):
        pass

    def fit(self, X, y):
        self.X = X  # Inputs
        self.y = y

        # x0 = [1.27, 1, 0.3] # initial guess
        x0 = [1.0] * 3 # initial guess

        # NOTE - works without jac, fails using jac
        res = minimize(self.SE_NLL, x0, method='bfgs')
        # res = minimize(self.SE_NLL, x0, method='bfgs', jac=self.SE_der)

        # Error     Length scale  Noise error
        [self.f_err, self.l_scale, self.n_err] = res['x']
        print("ferr: {}, lscale: {}, nerr: {}".format(self.f_err, self.l_scale, self.n_err))

        self.L = linalg.cholesky(self.K_se(X, X))
        self.alpha = linalg.solve(self.L.T, (linalg.solve(self.L, self.y)))

        # Inverse of K using Cholesky decomp/numerically stable
        self.K_inv = self.alpha/self.y

    def predict(self, x):
        ks = self.K_se(self.X, x)
        fs_mean = ks.T.dot(self.alpha)
        v = linalg.solve(self.L, ks)
        var = np.diag(self.K_se(x, x) - v.T.dot(v))
        return fs_mean, var

    def K_se(self, x1, x2, *args):
        if len(args) == 3:
            f_err, l_scale, n_err = args
        else:
            f_err = self.f_err
            l_scale = self.l_scale
            n_err = self.n_err

        K = (f_err**2) * (np.exp(-self.dist(x1, x2) / (2*l_scale**2)))
        kron_matrix = np.zeros((K.shape[0], K.shape[1]), dtype=float)
        np.fill_diagonal(kron_matrix, 1)
        return K + (n_err**2) * kron_matrix

    def SE_der(self, args):
        # TODO fix - get around apparent bug
        if len(args.shape) != 1:
            print(args)
            args = args[0]

        [f_err, l_scale, n_err] = args
        #TODO use alpha calculated from SE_NLL
        L = linalg.cholesky(self.K_se(self.X, self.X, f_err, l_scale, n_err))
        alpha = linalg.solve(L.T, (linalg.solve(L, self.y))) # save for use with derivative func
        aaT = alpha.dot(alpha.T)
        K_inv = alpha/self.y
        dK_dtheta = np.gradient(self.K_se(self.X, self.X, f_err, l_scale, n_err))[0]
        der = 0.5 * np.matrix.trace((aaT - K_inv).dot(dK_dtheta))
        # print('got here der {}'.format(der))
        return der

    # Args is an array to allow for scipy.optimize
    def SE_NLL(self, args):
        # TODO fix - get around apparent bug
        if len(args.shape) != 1:
            print(args)
            args = args[0]

        [f_err, l_scale, n_err] = args
        # print("sqe nll n_err {}".format(n_err))
        # print(args)
        L = linalg.cholesky(self.K_se(self.X, self.X, f_err, l_scale, n_err))
        alpha = linalg.solve(L.T, (linalg.solve(L, self.y))) # save for use with derivative func
        nll = (
            0.5 * self.y.T.dot(alpha) + 
            np.matrix.trace(L) + # sum of diagonal
            L.shape[0]/2 * math.log(2*math.pi)
                )
        # print('got here NLL {}'.format(nll))
        return nll

    def dist(self, x, xs):
        return cdist(x, xs, 'sqeuclidean')

    ####################### PLSC #######################
    def sigmoid(self, x):
        return 1/(1+np.exp**(-x))

    def LLOO(self, y, a_param, b_param):
        mean = y - (self.alpha/np.diag(self.K_inv))
        var = 1/(np.diag(self.K_inv))
        inner = y*(a_param * mean + b_param) / (np.sqrt(1 + a_param**2 * var))
        return cdf(inner)

    def LLOO_hyperparams_grad(self, y, a_param, b_param):
        mean, var, r, pdf_on_cdf = self.LLOO_grad_params(y, a_param, b_param)
        denom = 1 + a_param**2 * var
        mid = y*a_param/np.sqrt(denom)

        dK_dtheta = np.gradient(self.K_se(self.X, self.X, f_err, l_scale, n_err))[0] # TODO already above, don't recalculate here
        Z = self.K_inv * dK_dtheta
        dmean_dtheta = Z.dot(self.alpha)/np.diag(self.K_inv) - np.diag(self.alpha.dot(Z.dot(self.K_inv)))/np.diag(np.linalg.matrix_power(self.K_inv, 2))
        dvar_dtheta = np.diag(Z.dot(self.K_se))/np.diag(np.linalg.matrix_power(self.K_inv, 2))
        right = dmean_dtheta - (a_param*(a_param*mean + b_param))/(2*(denom)) * dvar_dtheta

        return pdf_on_cdf * mid * right

    def LLOO_a_grad(self, y, a_param, b_param):
        # TODO
        mean, var, r, pdf_on_cdf = self.LLOO_grad_params(y, a_param, b_param)
        denom = 1 + a_param**2 * var
        mid = y/np.sqrt(denom)
        right = (mean - b_param * a_param * var) / denom
        return sum(pdf_on_cdf * mid * right)

    def LLOO_b_grad(self, y, a_param, b_param):
        # TODO
        mean, var, r, pdf_on_cdf = self.LLOO_grad_params(y, a_param, b_param)
        denom = 1 + a_param**2 * var
        right = y/np.sqrt(denom)
        return sum(pdf_on_cdf * right)

    # TODO pre-calculate this to avoid doubling up
    def LLOO_grad_params(self, y, a_param, b_param):
        mean = y - (self.alpha/np.diag(self.K_inv))
        var = 1/(np.diag(self.K_inv))
        r = (a_param * mean + b_param) / np.sqrt(1 + a_param**2 * var)
        pdf_on_cdf = pdf(r)/cdf(y*r)
        return mean, var, r, pdf_on_cdf
