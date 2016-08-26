#! /usr/bin/env python3
""" Script to test out Dirichlet-Multinomial Regression. """

import logging
import numpy as np
import matplotlib.pyplot as pl
from revrand import basis_functions as bases
from yavanna.supervised.dirmultreg import dirmultreg_learn, dirmultreg_predict
from yavanna.linalg.linalg import logsumexp
from yavanna.distrib import dirichlet


# Set up logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def main():

    # Settings
    sampperclass1 = 20
    sampperclass2 = 20
    multisamp1 = 500
    multisamp2 = 10
    kfolds = 5
    activation = 'soft'

    # Make data
    X1 = np.random.multivariate_normal([-5, -5], [[1, 0], [0, 1]], sampperclass1)
    X2 = np.random.multivariate_normal([5, 5], [[1, 0], [0, 1]], sampperclass2)
    C1 = np.random.multinomial(multisamp1, [0.7, 0.2, 0.1], sampperclass1)
    C2 = np.random.multinomial(multisamp2, [0.2, 0.7, 0.1], sampperclass2)

    # Concatenate data
    X = np.vstack((X1, X2))
    XFeature = bases.RadialBasis(X)(X, lenscale=1)
    C = np.vstack((C1, C2))

    # Cross-validation
    permind = np.random.permutation(np.arange(sampperclass1+sampperclass2))
    cvind = np.array_split(permind, kfolds)

    rmse = 0
    for k in range(kfolds):
        sind = cvind[k]
        rind = np.concatenate(cvind[0:k] + cvind[k + 1:])
        XFeatureR = XFeature[rind, :]
        XFeatureS = XFeature[sind, :]
        Cr = C[rind, :]
        Cs = C[sind, :]
        # Learn regression weights
        W = dirmultreg_learn(XFeatureR, Cr, verbose=True, reg=0.1)
        # re-predict training just to see if this works!
        EC = dirmultreg_predict(XFeatureS, W)
        rmse += np.sqrt(((Cs - EC)**2).sum() / len(sind))

    log.info("Prediction RMSE = {} for {} folds.".format(rmse/kfolds, kfolds))

    # Generate a bunch of query points
    res = 30
    xeva,yeva = np.meshgrid(np.linspace(-10,10,res),np.linspace(-10,10,res))
    XQuery = np.concatenate((xeva.ravel()[:,np.newaxis],
                             yeva.ravel()[:,np.newaxis]),axis=1)
    # XQueryFeature = radialFeatureGen(XQuery,X, s=5)
    XQueryFeature = bases.RadialBasis(X)(XQuery, lenscale=5)

    #Evaluate most common draw
    EC, _ = dirmultreg_predict(XQueryFeature, W, counts=100)
    maxCat = np.argmax(EC,axis=1)
    pl.figure()
    pl.scatter(XQuery[:,0],XQuery[:,1],c=maxCat,s=50)
    pl.plot(X[:,0],X[:,1],'bo')
    pl.title('Most common draw')
    pl.colorbar()

    #Calculate entropy
    if activation == 'soft':
        NQuery = XQuery.shape[0]
        N = X.shape[0]
        K = W.shape[0]
        alpha = np.zeros([N,K])
        alphaQuery = np.zeros([NQuery,K])
        for k in range(K):
            XQueryW = XQueryFeature.dot(W[k,:].T)
            alphaQuery[:,k] = logsumexp(np.concatenate((np.zeros([NQuery,1]), XQueryW[:,np.newaxis]),axis=1),axis=1)
            alphaQuery[alphaQuery==0]=1e-300

            XW = XFeature.dot(W[k,:].T)
            alpha[:,k] = logsumexp(np.concatenate((np.zeros([N,1]), XW[:,np.newaxis]),axis=1),axis=1)
            alpha[alpha==0]=1e-300

    elif activation == 'exp':
        alpha = np.exp(XFeature.dot(W.T))
        alphaQuery = np.exp(XQueryFeature.dot(W.T))
    else:
        raise Exception('Choose a valid activation function')

    H = np.zeros(alphaQuery.shape[0])
    for i in range(alphaQuery.shape[0]):
        H[i] = dirichlet.entropy(alphaQuery[i,:])

    # H = np.log(H-np.min(H)+1e-300)

    #Plot entropy
    pl.figure()
    pl.scatter(XQuery[:,0],XQuery[:,1],c=H,s=50)
    pl.plot(X[:,0],X[:,1],'bo')
    pl.colorbar()
    pl.title('Entropy')


    for i in range(alphaQuery.shape[0]):
        H[i] = dirichlet.entropy(alphaQuery[i,:])

    stdDev = np.zeros(alphaQuery.shape)
    for i in range(alphaQuery.shape[0]):
        stdDev[i,:] = dirichlet.var(alphaQuery[i,:])

    stdMaxCat = np.zeros(alphaQuery.shape[0])
    for i in range(alphaQuery.shape[0]):
        stdMaxCat[i] = stdDev[i,maxCat[i]]

    pl.figure()
    pl.scatter(XQuery[:,0],XQuery[:,1],c=stdMaxCat,s=50)
    pl.plot(X[:,0],X[:,1],'bo')
    pl.title('Standard deviation on most commonly drawn category')
    pl.colorbar()
    pl.show()

if __name__ == "__main__":
    main()
