import numpy as np
from numpy import linalg
from sympy import KroneckerDelta
from scipy.optimize import minimize
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
        self.y = y  # Outputs

        # x0 = [1.27, 1, 0.3] # initial guess
        x0 = [1.0, 1.0, 1.0] # initial guess
        res = minimize(self.SQE_NLL, x0, method='bfgs')
        # res = minimize(self.SQE_NLL, x0, method='bfgs', jac=self.SQE_der)

        # Error     Length scale  Noise error
        [self.f_err, self.l_scale, self.n_err] = res['x']
        print("ferr: {}, lscale: {}, nerr: {}".format(self.f_err, self.l_scale, self.n_err))

        self.L = linalg.cholesky(self.K(X, X))
        self.alpha = linalg.solve(self.L.T, (linalg.solve(self.L, self.y)))

    def predict(self, x):
        ks = self.K(x, self.X)
        fs_mean = ks.T.dot(self.alpha)
        v = linalg.solve(self.L, ks)
        var = np.diag(self.K(x, x) - v.T.dot(v))
        return fs_mean, var

    def kernel(self, x, xs, *args):
        f_err, l_scale, n_err = args
        return (
            (f_err**2) 
            * math.exp(
            -sum((x-xs)**2)
            / (2*(l_scale**2))
            + (n_err**2)*KroneckerDelta(x, xs) 
            )
        )
    
    # K(X, X), K(X, X*), K(X*, X), k*, etc.
    def K(self, x1, x2, *args):
        if len(args) == 3:
            f_err, l_scale, n_err = args
        else:
            f_err = self.f_err
            l_scale = self.l_scale
            n_err = self.n_err

        # TODO fix/slow
        # vectorise
        return np.array([self.kernel(x1_point, x2_point, f_err, l_scale, n_err)
                for x2_point in x2 for x1_point in x1]).reshape(len(x2), len(x1))

    # currently taken from here - https://math.stackexchange.com/questions/1030534/gradients-of-marginal-likelihood-of-gaussian-process-with-squared-exponential-co/1072701#1072701
    def SQE_der(self, args):
        # TODO fix - get around buggy scipy
        if len(args.shape) != 1:
            args = args[2]

        [f_err, l_scale, n_err] = args
        #TODO use alpha calculated from SQE_NLL
        L = linalg.cholesky(self.K(self.X, self.X, f_err, l_scale, n_err))
        alpha = linalg.solve(L.T, (linalg.solve(L, self.y))) # save for use with derivative func
        aaT = alpha.dot(alpha.T)
        K_inv = alpha/self.y
        dK_dtheta = np.gradient(self.K(self.X, self.X, f_err, l_scale, n_err))[0]
        der = 0.5 * np.matrix.trace((aaT - K_inv).dot(dK_dtheta))
        # print('got here der {}'.format(der))
        return der

        # return 0.5*np.matrix.trace((aaT - K_inv).dot((dK_dtheta)))

    # Args is an array to allow for scipy.optimize
    def SQE_NLL(self, args):
        # TODO fix - get around buggy scipy
        if len(args.shape) != 1:
            args = args[2]

        [f_err, l_scale, n_err] = args
        # print("sqe nll n_err {}".format(n_err))
        # print(args)
        L = linalg.cholesky(self.K(self.X, self.X, f_err, l_scale, n_err))
        alpha = linalg.solve(L.T, (linalg.solve(L, self.y))) # save for use with derivative func
        nll = (
            0.5 * self.y.T.dot(alpha) + 
            np.matrix.trace(L) + # sum of diagonal
            L.shape[0]/2 * math.log(2*math.pi)
                )
        # print('got here NLL {}'.format(nll))
        return nll
