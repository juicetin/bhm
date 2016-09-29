""" Implementation of MAP Dirichlet-Multinomial Regression for regression over
    discrete distributions.
"""

import logging
import numpy as np
from scipy.special import psi, gammaln
# from yavanna.linalg import logistic, softplus, softmax    # no yavanna! (?)
from revrand.mathfun.special import softplus, softmax
from scipy.stats import logistic as sci_logistic
# from revrand.optimize import minimize
from scipy.optimize import minimize

def logistic(M):
    return sci_logistic.cdf(M)

# Set up logging
log = logging.getLogger(__name__)

def dirmultreg_learn(X, C, activation='soft', reg=1, verbose=False, ftol=1e-6,
                     maxit=3000):
    """ Train a Dirichlet-Multinomial Regressor using MAP to learn the weights.

        Arguments:
            X: (N, D) array of input features.
            C: (N, K) array of multinomial data.
            activation: the type of activation function to use. Valid
                inputs are:
                    - 'exp' for exponential (less numerically stable)
                    - 'soft' for soft-plus (leads to non-uniform priors)
            reg: weight prior variance (regulariser).
            verbose: print out learning status.

        Returns:
            array: (K, D) array of regression weights to be used for prediction
    """

    if activation != 'exp' and activation != 'soft':
        raise Exception("Invalid activation function name!")

    N, K = C.shape
    if N != X.shape[0]:
        raise ValueError("C and X have to have the same number of rows!")

    _, D = X.shape
    it = [0]

    def MAPobj(W):

        W = W.reshape(K, D)  # This comes out flattened

        if activation == 'exp':
            alpha = np.exp(X.dot(W.T))  # NxK matrix of alpha values
        else:
            alpha = softplus(X.dot(W.T))
            alpha[alpha < 1e-300] = 1e-300  # hack to avoid instability

        # Re-usable computations
        asum = alpha.sum(axis=1)
        acsum = (C + alpha).sum(axis=1)

        # Digamma and Log-Gamma terms
        psi_terms = (psi(asum) - psi(acsum))[:, np.newaxis] \
            + psi(C + alpha) - psi(alpha)  # NxK
        lgam_terms = (gammaln(asum) - gammaln(acsum)).sum() \
            + (gammaln(C + alpha) - gammaln(alpha)).sum()

        if activation == 'exp':
            grad = W / reg - (alpha * psi_terms).T.dot(X)  # KxD
        else:
            grad = W / reg - (logistic(X.dot(W.T)) * psi_terms).T.dot(X)

        MAP = - lgam_terms + (W**2).sum() / (2 * reg)

        if verbose:
            log.info("Iter. {}, Objective = {}".format(it[0], MAP))
            it[0] += 1

        return MAP, grad.flatten()

    # start_W = np.ones(K*D)
    start_W = np.random.rand(K*D)
    optres = minimize(MAPobj, start_W, jac=True, method='L-BFGS-B',
                      tol=ftol, options={'maxiter':maxit})

    if verbose:
        log.info("Success: {}, final objective = {}."
                 .format(optres.success, optres.fun))
    print("Success: {}, final objective = {}."
             .format(optres.success, optres.fun))

    return np.reshape(optres.x, (K, D))


def dirmultreg_predict(X, W, activation='soft', counts=1):
    """ Predict Multinomial counts from a Dirichlet Multinomial regressor.

        Arguments:
            X: (N, D) array of input features.
            W: (K, D) array of regression weights to be used for prediction.
            activation: the type of activation function to use. Valid
                inputs are:
                    - 'exp' for exponential (less numerically stable)
                    - 'soft' for soft-plus (leads to non-uniform priors)
            counts: Integer or N vector of the number of observations in
                each multinomial. This is optional, and if counts=1 the
                expected proportions are returned.

        Returns:
            array: (N, K) array of (unnormalised) expected multinomial counts.
                   I.e. multiply this by the number of multinomial counts.
            array: (K,) array of expected Dirichlet parameters, i.e. alphas.
    """

    if activation != 'exp' and activation != 'soft':
        raise Exception("Invalid activation function name!")

    XW = X.dot(W.T)

    if activation == 'exp':
        alpha = np.exp(XW)
        EC = softmax(XW, axis=1)

    else:
        alpha = softplus(XW)
        EC = alpha / alpha.sum(axis=1)[:, np.newaxis]

    if not np.isscalar(counts):
        return np.atleast_2d(counts).T * EC, alpha
    else:
        return counts * EC, alpha
