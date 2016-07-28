import numpy as np
import sys
from numpy import linalg
from sympy import KroneckerDelta
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from sympy.utilities.lambdify import lambdify, implemented_function
from scipy.stats import norm
import math
import sympy
from sympy.tensor.array import Array
from sympy.utilities.autowrap import ufuncify
import theano
from sympy.printing.theanocode import theano_function

from datetime import datetime

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

    # def fit(self, X, y):

    #     # Set basic data
    #     self.X = X  # Inputs
    #     self.y = y
    #     self.size = len(X)

    #     # Pre-calculate derivatives of inverted matrix to substitute values in the Squared Exponential NLL gradient
    #     # self.f_err_sym, self.l_scale_sym, self.n_err_sym = sympy.symbols("f_err, l_scale, n_err")
    #     self.f_err_sym, self.n_err_sym = sympy.symbols("f_err, n_err")
    #     self.l_scale_sym = sympy.MatrixSymbol('l', 1, self.size)
    #     m = sympy.Matrix(self.f_err_sym**2 * math.e**(-0.5 * self.dist(self.X/self.l_scale_sym, self.X/self.l_scale_sym)) 
    #                      + self.n_err_sym**2 * np.identity(self.size))
    #     self.dK_dthetas = [
    #                  m.diff(self.f_err_sym),
    #                  m.diff(self.l_scale_sym),
    #                  m.diff(self.n_err_sym)
    #                  ]

    #     # Determine optimal GP hyperparameters
    #     gp_hp_guess = [1.27, 1, 0.3] # initial guess
    #     # gp_hp_guess = [1.0] * 3 # initial guess
    #     res = minimize(self.SE_NLL, gp_hp_guess, method='bfgs')
    #     # res = minimize(self.SE_NLL, gp_hp_guess, method='bfgs', jac=self.SE_der, tol=1e-4)
    #     [self.f_err, self.l_scale, self.n_err] = res['x']

    #     # Set a few 'fixed' variables once GP HPs are determined for later use (with classifier)
    #     self.L = self.L_create(self.X, self.f_err, self.l_scale, self.n_err)
    #     self.K_inv = np.linalg.inv(self.L.T).dot(np.linalg.inv(self.L))
    #     self.alpha = linalg.solve(self.L.T, (linalg.solve(self.L, self.y)))

    def L_create(self, X, f_err, l_scales, n_err):
        self.f_err = f_err
        self.l_scales = l_scales
        self.n_err = n_err
        self.X_1 = X

        m = self.K_se(X, X, f_err, l_scales) + n_err**2 * np.identity(X.shape[0])
        m = np.array(m, dtype=np.float64)
        return linalg.cholesky(m)
        # return linalg.cholesky(self.K_se(X, X, f_err, l_scales) 
        #         + float(n_err**2) * np.identity(X.shape[0]))

    def predict(self, x):
        ks = self.K_se(self.X, x, self.f_err, self.l_scales)
        fs_mean = ks.T.dot(self.alpha)
        v = linalg.solve(self.L, ks)
        var = np.diag(self.K_se(x, x, self.f_err, self.l_scales) - v.T.dot(v))
        return fs_mean, var

    def se_term(self, x1, x2, l_scales):
        # Dividing by length scale first before passing into cdist to
        #   accounts for different length scale for each dimension
        # return self.sqeucl_dist(x1/l_scales, x2/l_scales)
        return cdist(x1/l_scales, x2/l_scales)

    # NOTE cdist can't deal with sympy symbols :(
    def sqeucl_dist(self, x, xs, x_len=0, xs_len=0):
        # BAD. Need to get cdist to work with symbols, or make this way more efficient
        # m = np.sum( np.power([i-j for i in x for j in xs], 2), axis=1).reshape(x.shape[0], xs.shape[0])

        # if self.lib == 'sympy':
        m = np.sum(np.power(
            np.repeat(x[:,None,:], len(self.X), axis=1) - 
            np.resize(xs, (len(self.X), xs.shape[0], xs.shape[1])), 
            2), axis=2)
        # elif self.lib == 'theano':
        #     m = np.power([i-j for i in x for j in xs], 2).reshape(x.shape[0], xs.shape[0])

        return m

        # return cdist(x, xs, 'sqeuclidean')
        # return (x - xs.T)**2

    def K_se(self, x1, x2, f_err, l_scales):

        # m = np.array(self.se_term(x1, x2, l_scales), dtype=np.float32)
        m = self.se_term(x1, x2, l_scales).astype(float)
        # m = cdist(x1/l_scales, x2/l_scales)
        # return np.array((f_err**2) * np.exp(-0.5 * m), dtype=np.float64)
        return f_err**2 * np.exp(-0.5 * m)

    # Evaluates each dK_dtheta pre-calculated symbolic lambda with current iteration's hyperparameters
    def eval_dK_dthetas(self, f_err, l_scales, n_err):
        # Sympy
        if self.lib == 'sympy':
            l_scales = sympy.Matrix(l_scales.reshape(1, len(l_scales)))
            return np.array(self.dK_dthetas(f_err, l_scales, n_err), dtype=np.float64)

        elif self.lib == 'theano':
        # Theano
        # Break down dK_dls into partial derivative over each length scale
            dK_df_raw, dK_dls_merged_raw, dK_dn_raw = self.dK_dthetas
            dK_df, dK_dls_merged, dK_dn = [dK_df_raw(f_err, l_scales, n_err), dK_dls_merged_raw(f_err, l_scales, n_err), dK_dn_raw(f_err, l_scales, n_err)]

            dimensions = self.X.shape[1]
            idxs = [np.arange(i, self.X.shape[0] * dimensions, dimensions) for i in range(dimensions)]
            dK_dls = [dK_dls_merged[idx] for idx in idxs]

            return np.array([dK_df] + dK_dls + [dK_dn], dtype=np.float64)

    # def SE_der(self, args):
    #     # TODO fix - get around apparent bug
    #     # if len(args.shape) != 1:
    #     #     # print(args)
    #     #     args = args[0]

    #     [f_err, l_scale, n_err] = args
    #     # TODO use alpha calculated from SE_NLL

    #     L = self.L_create(self.X, f_err, l_scale, n_err)
    #     alpha = linalg.solve(L.T, (linalg.solve(L, self.y))) # save for use with derivative func
    #     aaT = alpha.dot(alpha.T)
    #     K_inv = np.linalg.inv(L.T).dot(np.linalg.inv(L))

    #     # Calculate dK/dtheta over each hyperparameter
    #     eval_dK_dthetas = self.eval_dK_dthetas(f_err, l_scale, n_err)

    #     # Incorporate each dK/dt into gradient
    #     derivatives = [float(-0.5 * np.matrix.trace((aaT - K_inv).dot(dK_dtheta))) for dK_dtheta in eval_dK_dthetas]
    #     return np.array(derivatives)

    # # Args is an array to allow for scipy.optimize
    # def SE_NLL(self, args):
    #     # TODO fix - get around apparent bug
    #     if len(args.shape) != 1:
    #         args = args[0]

    #     [f_err, l_scale, n_err] = args
    #     L = self.L_create(self.X, f_err, l_scale, n_err)
    #     
    #     alpha = linalg.solve(L.T, (linalg.solve(L, self.y))) # save for use with derivative func
    #     nll = (
    #         0.5 * self.y.T.dot(alpha) + 
    #         0.5 * np.matrix.trace(L) + # sum of diagonal
    #         0.5 * self.size * math.log(2*math.pi)
    #             )

    #     return nll

    ####################################################
    ####################### PLSC #######################
    ####################################################

    def sigmoid(self, x):
        return 1/(1+np.exp(-x))

    #################### Negative LOO log predictive probability ####################

    # Unpacks arguments to deal with list of length scales in list of arguments
    def unpack_LLOO_args(self, args):
        f_err = float(args[0])
        l_scales = args[1:self.X.shape[1]+1]
        n_err = args[self.X.shape[1]+1]
        a = float(args[self.X.shape[1]+2])
        b = float(args[self.X.shape[1]+3])
        return f_err, l_scales, n_err, a, b

    def LLOO(self, args):
        f_err, l_scales, n_err, a, b = self.unpack_LLOO_args(args)

        L = self.L_create(self.X, f_err, l_scales, n_err)
        alpha = linalg.solve(L.T, (linalg.solve(L, self.y))) # save for use with derivative func
        K_inv = np.linalg.inv(L.T).dot(np.linalg.inv(L))
        mu = self.y - alpha/np.diag(K_inv)
        sigma_sq = 1/np.diag(K_inv)

        LLOO = -sum(norm.cdf(
            self.y * (a * mu + b) /
            np.sqrt(1 + a**2 * sigma_sq)
        ))

        return LLOO

    #################### Derivative ####################

    # NOTE likely incorrect - currently doesn't recalculate K per datapoint
    def LLOO_der(self, args):
        d1 = datetime.now()
        
        self.count += 1
        # print("Iterated [{}] times for current fitting...".format(self.count))
        print(self.count, end=" ", flush=True)

        f_err, l_scales, n_err, a, b = self.unpack_LLOO_args(args)

        # This block is common to both LLOO and LLOO_der
        L = self.L_create(self.X, f_err, l_scales, n_err)
        alpha = linalg.solve(L.T, (linalg.solve(L, self.y))) # save for use with derivative func
        K_inv = np.linalg.inv(L.T).dot(np.linalg.inv(L))
        mu = self.y - alpha/np.diag(K_inv)
        sigma_sq = 1/np.diag(K_inv)

        r = a * mu + b

        ####################### This block takes 99.75% of the entire derivative function's timeshare #######################
        K_i_diag = np.diag(K_inv)

        # d3 = datetime.now()
        dK_dthetas = self.eval_dK_dthetas(f_err, l_scales, n_err) # ~99.7%
        # param_time = datetime.now() - d3
        # print("Time for eval dK_dts {}".format(param_time))

        Zs = np.array([K_inv.dot(dK_dtheta) for dK_dtheta in dK_dthetas])
        dvar_dthetas = [np.diag(Z.dot(K_inv))/K_i_diag**2 for Z in Zs] 
        dmu_dthetas = [Z.dot(alpha) / K_i_diag - alpha * dvar_dtheta for Z, dvar_dtheta in zip(Zs, dvar_dthetas)]
        #####################################################################################################################

        pdf_on_cdf = norm.pdf(r) / norm.cdf(self.y * r)

        # Dervative over LLOO for each of the hyperparameters
        dLLOO_dthetas = [
                -sum(pdf_on_cdf * 
                    (self.y * a / np.sqrt(1 + a**2 * sigma_sq)) * 
                    (dmu_dtheta - 0.5 * a * (a * mu + b) / (1 + a**2 * sigma_sq) * dvar_dtheta))
                    for dmu_dtheta, dvar_dtheta in zip(dmu_dthetas, dvar_dthetas)
        ]

        # Derivative of LLOO w.r.t b
        dLLOO_db_arr = (
            pdf_on_cdf *
            self.y / np.sqrt(1 + a**2 * sigma_sq)
        )
        dLLOO_db = -sum(dLLOO_db_arr)

        # Derivative of LLOO w.r.t a, utilising dLLOO/db
        dLLOO_da = -sum(dLLOO_db_arr *
                        (mu - b * a * sigma_sq) /
                        (1 + a**2 * sigma_sq)
                       )
        
        gradients = dLLOO_dthetas + [dLLOO_da] + [dLLOO_db]

        # tottime = datetime.now() - d1
        # print("Total time in LLOO_der: {}, % spent in params: {}".format(tottime, param_time/tottime*100))
        return np.array(gradients, dtype=np.float64)

    #################### Prediction ####################

    def build_symbolic_derivatives(self):
        # Pre-calculate derivatives of inverted matrix to substitute values in the Squared Exponential NLL gradient
        self.f_err_sym, self.n_err_sym = sympy.symbols("f_err, n_err")

        # (1,n) shape 'matrix' (vector) of length scales for each dimension
        self.l_scale_sym= sympy.MatrixSymbol('l', 1, self.X.shape[1])

        # K matrix
        print("Building sympy matrix...")
        m = sympy.Matrix(self.f_err_sym**2 * math.e**(-0.5 * self.sqeucl_dist(self.X/self.l_scale_sym, self.X/self.l_scale_sym)) 
                         + self.n_err_sym**2 * np.identity(self.size))


        # Element-wise derivative of K matrix over each of the hyperparameters
        print("Getting partial derivatives over all hyperparameters...")
        dK_df   = m.diff(self.f_err_sym)
        dK_dls  = [m.diff(l_scale_sym) for l_scale_sym in self.l_scale_sym]
        dK_dn   = m.diff(self.n_err_sym)

        # Lambdify each of the dK/dts to speed up substitutions per optimization iteration
        print("Lambdifying ")
        self.dK_dthetas = [dK_df] + dK_dls + [dK_dn]
        self.dK_dthetas = sympy.lambdify((self.f_err_sym, self.l_scale_sym, self.n_err_sym), self.dK_dthetas, 'numpy')
        # self.dK_dthetas = sympy.lambdify((self.f_err_sym, self.l_scale_sym, self.n_err_sym), self.dK_dthetas, [{'ImmutableMatrix':np.array, "numpy"])

    def build_symbolic_derivatives_theano(self):
        self.f_err_sym = theano.tensor.dscalar('f_err')
        self.n_err_sym = theano.tensor.dscalar('n_err')
        self.l_scale_sym = theano.tensor.dvector('l_scales')

        m = (self.f_err_sym**2 * math.e**(-0.5 * self.sqeucl_dist(self.X/self.l_scale_sym, self.X/self.l_scale_sym))
                + self.n_err_sym**2 * np.identity(self.size))

        # Partial derivatives over all hyperparameters
        dK_df, dK_dls_merged, dK_dn = theano.gradient.jacobian(m.flatten(), [self.f_err_sym, self.l_scale_sym, self.n_err_sym])
        args = [self.f_err_sym, self,l_scale_sym, self.n_err_sym]

        dK_df_eval = theano.function(args, dK_df)
        dK_dls_merged_eval = theano.function(args, dK_dls_merged)
        dK_dn_eval = theano.function(args, dK_dn)

        self.dK_dthetas = [dK_df_eval, dK_dls_merged_eval, dK_dn_eval]

    # Classification
    def fit_class(self, X, y):
        self.lib = 'sympy'
        # self.lib = 'theano'

        self.count = 0

        self.size = len(y)
        self.X = X
        self.classifier_params = {}
        params = ['f_err', 'l_scales', 'n_err', 'a', 'b']

        # Build lambdified partial derivatives
        if self.lib == 'sympy':
            self.build_symbolic_derivatives()
        elif self.lib == 'theano':
            self.build_symbolic_derivatives_theano()

        # Build OvA classifier for each unique class in y
        print("Starting to build OvA classifier per class...")
        print("Current iterations... ", end=" ", flush=True)
        for c in set(y):
            self.classifier_params[c] = {}

            # f_err, l_scales (for each dimension), n_err, alpha, beta
            x0 = [1] + [1] * X.shape[1] + [1, 1, 1]

            # Set binary labels for OvA classifier
            self.y = np.array([1 if label == c else 0 for label in y])

            # Optimise and save hyper/parameters for current binary class pair
            # res = minimize(self.LLOO, x0, method='bfgs')
            res = minimize(self.LLOO, x0, method='bfgs', jac=self.LLOO_der)

            # Set params for current binary regressor (classifier)
            for param, val in zip(params, res['x']):
                self.classifier_params[c][param] = val

            # Reset ys
            self.y = y
        print()

    def predict_class(self, x):
        print("Original: {}".format(self.y))

        # Copy y for modification and resetting/restoring
        y = np.copy(self.y)

        # Generate squashed y precidtions
        y_preds = [
            self.predict_class_single(x, y, label, params)
            for label, params in self.classifier_params.items()
        ]

        print("Regression values for binary cases:")
        for y_pred in y_preds:
            print(y_pred)

        # Return max squashed value for each data point representing class prediction
        return np.argmax(y_preds, axis=0)

    def predict_class_single(self, x, y, label, params):
        # Set parameters
        self.f_err = params['f_err']
        self.l_scales = params['l_scales']
        self.n_err = params['n_err']

        # Set y to binary one vs. all labels
        self.y = np.array([1 if y_i == label else -1 for y_i in y])

        # Set L and alpha matrices
        self.L = self.L_create(self.X, self.f_err, self.l_scales, self.n_err)
        self.alpha = linalg.solve(self.L.T, (linalg.solve(self.L, self.y)))

        # Get predictions of resulting mean and variances
        y_pred, var = self.predict(x)
        # sigma = np.sqrt(var)
        y_squashed = self.sigmoid(y_pred)

        # Restore y
        self.y = y

        return y_squashed

    def score(self, y_, y):
        return sum(y_ == y)/len(y_)
