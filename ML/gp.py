import numpy as np
from numpy import linalg
from sympy import KroneckerDelta
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from scipy.stats import norm
import math
import sympy

# TODO when doing K(X, X), simply use (n_err^2)I instead of KroneckerDelta

# using http://www.robots.ox.ac.uk/~mebden/reports/GPtutorial.pdf

# Example
# a = np.array([1,2,3,4,5])
# a = a.reshape(5,1) # 5 rows, 1 column
# b = np.array([6,7,8])
# b = np.reshape(1,3) # 1 row, 3 columns

class GaussianProcess:
    def __init__(self):
        pass

    def fit(self, X, y):

        # Set basic data
        self.X = X  # Inputs
        self.y = y
        self.size = len(X)

        # Pre-calculate derivatives of inverted matrix to substitute values in the Squared Exponential NLL gradient
        self.f_err_sym, self.l_scale_sym, self.n_err_sym = sympy.symbols("f_err, l_scale, n_err")
        m = sympy.Matrix(self.f_err_sym**2 * math.e**(-self.dist(self.X, self.X)/(2*self.l_scale_sym**2)) + self.n_err_sym**2 * np.identity(self.size))
        self.dK_dthetas = [
                     m.diff(self.f_err_sym),
                     m.diff(self.l_scale_sym),
                     m.diff(self.n_err_sym)
                     ]

        # Determine optimal GP hyperparameters
        gp_hp_guess = [1.27, 1, 0.3] # initial guess
        # gp_hp_guess = [1.0] * 3 # initial guess
        res = minimize(self.SE_NLL, gp_hp_guess, method='bfgs')
        # res = minimize(self.SE_NLL, gp_hp_guess, method='bfgs', jac=self.SE_der, tol=1e-4)
        [self.f_err, self.l_scale, self.n_err] = res['x']
        print(res)
        print("ferr: {}, lscale: {}, nerr: {}".format(self.f_err, self.l_scale, self.n_err))

        # Set a few 'fixed' variables once GP HPs are determined for later use (with classifier)
        self.L = self.L_create(self.f_err, self.l_scale, self.n_err)
        self.K_inv = np.linalg.inv(self.L.T).dot(np.linalg.inv(self.L))
        self.alpha = linalg.solve(self.L.T, (linalg.solve(self.L, self.y)))

    def L_create(self, X, f_err, l_scale, n_err):
        return linalg.cholesky(self.K_se(X, X, f_err, l_scale) 
                + (n_err**2) * np.identity(self.X.shape[0]))

    def predict(self, x):
        ks = self.K_se(self.X, x, self.f_err, self.l_scale)
        fs_mean = ks.T.dot(self.alpha)
        v = linalg.solve(self.L, ks)
        var = np.diag(self.K_se(x, x, self.f_err, self.l_scale) - v.T.dot(v))
        return fs_mean, var

    def K_se(self, x1, x2, f_err, l_scale):
        K = (f_err**2) * (np.exp(-self.dist(x1, x2) / (2*l_scale**2)))
        return K 

    def SE_der(self, args):
        # TODO fix - get around apparent bug
        # if len(args.shape) != 1:
        #     # print(args)
        #     args = args[0]

        [f_err, l_scale, n_err] = args
        # TODO use alpha calculated from SE_NLL
        L = self.L_create(self.X, f_err, l_scale, n_err)
        alpha = linalg.solve(L.T, (linalg.solve(L, self.y))) # save for use with derivative func
        aaT = alpha.dot(alpha.T)
        K_inv = np.linalg.inv(L.T).dot(np.linalg.inv(L))

        # dK_dtheta = np.gradient(self.K_se(self.X, self.X, f_err, l_scale))[0]
        # dK/dtheta elementwise derivaties
        # m = sympy.Matrix(f_err_sym**2 * math.e**(-self.dist(self.X, self.X)/(2*l_scale_sym**2)) + n_err_sym**2 * np.identity(self.size))
        # dK_dthetas = [
        #              np.matrix(m.diff(f_err_sym).subs({ f_err_sym:f_err , l_scale_sym:l_scale , n_err_sym:n_err })),
        #              np.matrix(m.diff(l_scale_sym).subs({ f_err_sym:f_err , l_scale_sym:l_scale , n_err_sym:n_err })),
        #              np.matrix(m.diff(n_err_sym).subs({ f_err_sym:f_err , l_scale_sym:l_scale , n_err_sym:n_err }))
        #              ]

        eval_dK_dthetas = [np.array(m.subs( { self.f_err_sym:f_err , self.l_scale_sym:l_scale , self.n_err_sym:n_err } )) 
                for m in self.dK_dthetas]
        derivatives = [float(-0.5 * np.matrix.trace((aaT - K_inv).dot(dK_dtheta))) for dK_dtheta in eval_dK_dthetas]
        print("param vals: {}".format(args))
        print("derivatives: {}".format(derivatives))
        # a = -0.5 * np.matrix.trace((aaT - K_inv).dot(dK_dthetas[0]))
        return np.array(derivatives)

    # Args is an array to allow for scipy.optimize
    def SE_NLL(self, args):
        # TODO fix - get around apparent bug
        if len(args.shape) != 1:
            # print(args)
            args = args[0]

        [f_err, l_scale, n_err] = args
        L = self.L_create(self.X, f_err, l_scale, n_err)
        
        alpha = linalg.solve(L.T, (linalg.solve(L, self.y))) # save for use with derivative func
        nll = (
            0.5 * self.y.T.dot(alpha) + 
            0.5 * np.matrix.trace(L) + # sum of diagonal
            0.5 * self.size * math.log(2*math.pi)
                )

        return nll

    def dist(self, x, xs):
        return cdist(x, xs, 'sqeuclidean')
        # return (x - xs.T)**2

    ####################################################
    ####################### PLSC #######################
    ####################################################

    def sigmoid(self, x):
        return 1/(1+np.exp**(-x))

    #################### Negative LOO log predictive probability ####################
    def LLOO(self, args):

        return -sum([self.LOOP(i, args) for i in arange(self.size)])

    # Leave one out probability
    def LOOP(self, i, args):
        [f_err, l_scale, n_err, a_param, b_param] = args

        # Leave one out datasets (omit i-th point)
        X = np.array(self.X[:i] + self.X[i+1:])
        y = np.array(self.y[:i] + self.y[i+1:])

        # Calculate alpha and inverse matrix based on LOO dataset
        L = self.L_create(X, f_err, l_scale, n_err)
        alpha = linalg.solve(L.T, (linalg.solve(L, y))) # save for use with derivative func
        K_inv = np.linalg.inv(L.T).dot(np.linalg.inv(L))

        # Return cumulative Gaussian for single training point
        return norm.cdf(
            self.y[i] * ( a_param * self.mean_one(i, alpha, K_inv) + b_param) /
            np.sqrt( 1 + a_param**2 * self.var_one(i, K_inv) )
        )

    def mean_one(self, i, alpha, K_inv):
        return self.y[i] - alpha[i]/K_inv[i][i]

    def var_one(self, i, K_inv):
        return 1/K_inv[i][i]

    #################### Derivative ####################
    def LLOO_der(self, args):
        [f_err, l_scale, n_err, a_param, b_param] = args

    #################### Prediction ####################
    # Classification
    def predict_class(self, x):
        for label in set(self.y):

            # One vs. all. Set all other labels to 'false'
            y_binary = np.array([1 if i == label else -1 for i in self.y])

            # Do classification prediction
            y_pred, MSE = self.predict_for_binary(y_binary)
            sigma = np.root(MSE)

    def predict_for_binary(self, y):
        gp_ab_guess = [1.0] * 2
        res = minimize(self.LLOO, gp_ab_guess, method='bfgs')
        # # res = minimize(self.LLOO, gp_ab_guess, method='bfgs', jac=self.)
        [self.a_param, self.b_param] = res['x']
        # print("alpha: {}, beta: {}".format(self.a_param, self.b_param))
