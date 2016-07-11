import numpy as np
from numpy import linalg
from sympy import KroneckerDelta
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
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
        x0 = [1.0] * 3 # initial guess

        # NOTE - works without jac, fails using jac
        res = minimize(self.SE_NLL, x0, method='bfgs')
        # res = minimize(self.SE_NLL, x0, method='bfgs', jac=self.SE_der)

        # Error     Length scale  Noise error
        [self.f_err, self.l_scale, self.n_err] = res['x']
        print("ferr: {}, lscale: {}, nerr: {}".format(self.f_err, self.l_scale, self.n_err))

        self.L = linalg.cholesky(self.K_se(X, X))
        self.alpha = linalg.solve(self.L.T, (linalg.solve(self.L, self.y)))

    def predict(self, x):
        ks = self.K_se(x, self.X)
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
        return K + (n_err**2) * np.identity(len(self.y))

    # currently taken from here - https://math.stackexchange.com/questions/1030534/gradients-of-marginal-likelihood-of-gaussian-process-with-squared-exponential-co/1072701#1072701
    def SE_der(self, args):
        # TODO fix - get around apparent bug
        if len(args.shape) != 1:
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

        # return 0.5*np.matrix.trace((aaT - K_inv).dot((dK_dtheta)))

    # Args is an array to allow for scipy.optimize
    def SE_NLL(self, args):
        # TODO fix - get around apparent bug
        if len(args.shape) != 1:
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

if __name__ == "__main__":
    gp = GaussianProcess()
    X = np.array([-1.50,-1.00,-0.75,-0.40,-0.25,0.00])
    X = X.reshape(len(X), 1)
    y = np.array([-1.70,-1.20,-0.25,0.30,0.5,0.7])
    y = y.reshape(len(y), 1)
    x = np.array([0.2])
    x = x.reshape(len(x), 1)
    gp.fit(X, y)

