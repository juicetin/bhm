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

        # Set a few 'fixed' variables once GP HPs are determined for later use (with classifier)
        self.L = self.L_create(self.X, self.f_err, self.l_scale, self.n_err)
        self.K_inv = np.linalg.inv(self.L.T).dot(np.linalg.inv(self.L))
        self.alpha = linalg.solve(self.L.T, (linalg.solve(self.L, self.y)))

    def L_create(self, X, f_err, l_scales, n_err):
        return linalg.cholesky(self.K_se(X, X, f_err, l_scales) 
                + (n_err**2) * np.identity(X.shape[0]))

    def predict(self, x):
        ks = self.K_se(self.X, x, self.f_err, self.l_scale)
        fs_mean = ks.T.dot(self.alpha)
        v = linalg.solve(self.L, ks)
        var = np.diag(self.K_se(x, x, self.f_err, self.l_scale) - v.T.dot(v))
        return fs_mean, var

    def se_term_one_length_scale(self, x1, x2, l_scales):
        return (1/l_scales**2) * self.dist(x1, x2)

    def se_term_length_scale_per_d(self, x1, x2, l_scales):
        return (
            np.array([sum((i-j)/l_scales) for i in x1 for j in x2 ])
                .reshape(len(x1), len(x2)) ** 2
        )

    def K_se(self, x1, x2, f_err, l_scales):
        return (
            (f_err**2) * 
            np.exp(-0.5 * # (1/l_scales**2) * self.dist(x1, x2)
                self.se_term(x1, x2, l_scales
            ))
        )

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

        # Calculate dK/dtheta over each hyperparameter
        eval_dK_dthetas = [np.array(m.subs( { self.f_err_sym:f_err , self.l_scale_sym:l_scale , self.n_err_sym:n_err } )) 
                for m in self.dK_dthetas]

        # Incorporate each dK/dt into gradient
        derivatives = [float(-0.5 * np.matrix.trace((aaT - K_inv).dot(dK_dtheta))) for dK_dtheta in eval_dK_dthetas]
        return np.array(derivatives)

    # Args is an array to allow for scipy.optimize
    def SE_NLL(self, args):
        # TODO fix - get around apparent bug
        if len(args.shape) != 1:
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
        return 1/(1+np.exp(-x))

    #################### Negative LOO log predictive probability ####################

    # Unpacks arguments to deal with list of length scales in list of arguments
    def unpack_LLOO_args(self, args):
        f_err = args[0]
        l_scales = args[1:self.X.shape[1]+1]
        n_err = args[self.X.shape[1]+1]
        a_param = args[self.X.shape[1]+2]
        b_param = args[self.X.shape[1]+3]
        return f_err, l_scales, n_err, a_param, b_param

    def LLOO(self, args):
        return -sum([self.LOOP(i, args) for i in range(self.size)])

    # Leave one out probability
    def LOOP(self, i, args):

        if len(args) == 5:
            f_err, l_scales, n_err, a_param, b_param = args
            self.se_term = self.se_term_one_length_scale
        else:
            self.se_term = self.se_term_length_scale_per_d
            f_err, l_scales, n_err, a_param, b_param = self.unpack_LLOO_args(args)

        X = np.delete(self.X, (i), axis=0)
        y = np.delete(self.y, i)

        # NOTE Hack here. How is this actually meant to work...?
        if i == self.size-1:
            i = self.size-2

        # Calculate alpha and inverse matrix based on LOO dataset
        L = self.L_create(X, f_err, l_scales, n_err)
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
    def fit_class(self, X, y):

        self.size = len(y)
        self.X = X
        self.classifier_params = {}
        params = ['f_err', 'l_scales', 'n_err', 'a_param', 'b_param']

        # Build OvA classifier for each unique class in y
        for c in set(y):
            self.classifier_params[c] = {}

            # f_err, l_scales (for each dimension), n_err, alpha, beta
            # x0 = [1] + [1] * X.shape[1] + [1, 1, 1]
            x0 = [1, 1, 1, 1, 1] # original

            # Set binary labels for OvA classifier
            self.y = np.array([1 if label == c else 0 for label in y])

            # Optimise and save hyper/parameters for current binary class pair
            res = minimize(self.LLOO, x0, method='bfgs')
            # res = minimize(self.LLOO, x0, method='bfgs', jac=self.SE_der, tol=1e-4)

            # Set params for current binary regressor (classifier)
            for param, val in zip(params, res['x']):
                self.classifier_params[c][param] = val

        # Build all OvA binary classifiers
        # for label in set(self.y):

        #     # One vs. all. Set all other labels to 'false'
        #     y_binary = np.array([1 if i == label else -1 for i in self.y])

        #     # Do classification prediction
        #     y_pred, MSE = self.predict_for_binary(y_binary)
        #     sigma = np.root(MSE)

    def predict_class(self, x):
        y = np.copy(self.y)

        y_preds = [
            self.predict_class_single(x, y, label, params)
            for label, params in self.classifier_params.items()
        ]

        # print(y_preds)

        # Reset y to its original values
        self.y = y

        # Return max squashed value for each data point
        return np.argmax(y_preds, axis=0)

    def predict_class_single(self, x, y, label, params):
        # Set parameters
        self.f_err = params['f_err']
        self.l_scale = params['l_scales']
        self.n_err = params['n_err']

        # Set y to binary one vs. all labels
        self.y = np.array([1 if y_i == label else -1 for y_i in y])

        # Set L and alpha matrices
        self.L = self.L_create(self.X, self.f_err, self.l_scale, self.n_err)
        self.alpha = linalg.solve(self.L.T, (linalg.solve(self.L, self.y)))

        # Get predictions of resulting mean and variances
        y_pred, var = self.predict(x)
        # sigma = np.sqrt(var)
        y_squashed = self.sigmoid(y_pred)

        return y_squashed

    def score(self, y_, y):
        return sum(y_ == y)/len(y_)
