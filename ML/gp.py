import numpy as np
from numpy import linalg
from sympy import KroneckerDelta
from scipy.optimize import minimize
import math

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

        # slow optimiser method
        res = minimize(self.SQE_NLL, [1.27,1,0.3], method='nelder-mead')
        # Error     Length scale  Noise error
        self.f_err, self.l_scale, self.n_err = res['x']
        self.K_inv = self.orig_cov_inv(self.X)
    
    # Squared exponential
    # e^(-1/2 * |x - y|^2)
    def sum_vals(self, vals):
        try:
            return sum(vals)
        except:
            return vals

    def kernel(self, x, xs, *args):
        # Hard-coded fail, need to do MLE here
        # l = 1 # l = 1.25 # length-scale
        # n_err = 0.3
        # f_err = 1.27
        if len(args) == 3:
            f_err, l_scale, n_err = args
        else:
            f_err = self.f_err
            l_scale = self.l_scale
            n_err = self.n_err

        return (f_err**2) * math.exp( \
        -self.sum_vals((x-xs)**2) \
        / (2*(l_scale**2)) \
        + (n_err**2)*KroneckerDelta(x, xs) # Need Gaussian noise variance here...
        )
    
    # K(X, X), K(X, X*), K(X*, X), k*, etc.
    def K(self, x1, x2, *args):
        if len(args) == 3:
            f_err, l_scale, n_err = args
        else:
            f_err = self.f_err
            l_scale = self.l_scale
            n_err = self.n_err

        matrix = []
        # TODO fix/slow
        # vectorise
        for x1_point in x1:
            for x2_point in x2:
                matrix.append(self.kernel(x1_point, x2_point, f_err, l_scale, n_err))
        return np.array(matrix).reshape(len(x1), len(x2))
    
    def orig_cov_inv(self, x):
        #TODO K(x, x) needs to add error sigma^2*I
        return linalg.inv(self.K(x, x))
    
    # E(f*|X,y,X*] = K(X*,X)[K(X,X) + v^2I]^(-1)y
    def mean_func(self, x, xs, y):
        return self.K(xs, x) \
        .dot(self.K_inv)            \
        .dot(y)             
    
    def variance(self, xs):
        return self.K(xs, xs) 
        - self.K(xs, self.X).T     \
        .dot(self.K_inv)             \
        .dot(self.K(xs, self.X).T) \

    def predict(self, x, eval_MSE=False):
        avg_pred = self.mean_func(self.X, x, self.y)
        if eval_MSE is True:
            mse = self.variance(x)
            return avg_pred, np.diag(mse)
        return avg_pred

    def SQE_NLL(self, args):
        [f_err, l_scale, n_err] = args
        # Minimise NLL
        return 0.5 * self.y.T.dot(linalg.inv(self.K(self.X, self.X, f_err, l_scale, n_err))).dot(self.y) \
        + 0.5*math.log(np.linalg.det(self.K(self.X, self.X, f_err, l_scale, n_err))) \
        + len(self.y)/2* math.log(2*math.pi)
    
    # x   # training inputs
    # y   # training outputs
    # xs  # test inputs
    
