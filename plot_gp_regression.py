print(__doc__)

import numpy as np
from sklearn.gaussian_process import GaussianProcess
# from ML.gp import GaussianProcess
from matplotlib import pyplot as pl

np.random.seed(1)
# confidence = 1.9600
confidence = 1.9600

def f(x):
    return x * np.sin(x)

datasize = 1000

# Noiseless case
X = np.atleast_2d([1., 3., 5., 6., 7., 8.]).T

# Observations
y = f(X).ravel()

# Mesh input space for evaluations of real function, prediction, variance
x = np.atleast_2d(np.linspace(0, 10, datasize)).T

# Instantiate GP model
try:
    # gp = GaussianProcess(corr='cubic', theta0=1e-2, thetaL=1e-4, thetaU=1e-1, random_start=100)
    gp = GaussianProcess(corr='squared_exponential', theta0=1e-1,
                         thetaL=1e-3, thetaU=1,
                         nugget=(dy / y) ** 2,
                         random_start=100)
except:
    gp = GaussianProcess()

# Fit data using MLE of params
gp.fit(X, y)

# Make prediction on meshed x-axis (Asking for variance too)
try:
    y_pred, variance = gp.predict(x, eval_MSE=True)
except:
    y_pred, variance = gp.predict(x)
sigma = np.sqrt(variance)

# Plot function, prediction, and 95% confidence interval based on variance
fig = pl.figure()
pl.plot(x, f(x), 'r:', label=u'$f(x) = x\, \sin(x)$')
pl.plot(X, y, 'r.', markersize=10, label=u'Observations')
pl.plot(x, y_pred, 'b-', label=u'Prediction')
pl.fill(np.concatenate([x, x[::-1]]),
        np.concatenate([y_pred - confidence * sigma,
                      (y_pred + confidence * sigma)[::-1]]),
        alpha=.5, fc='b', ec='None', label='95% confidence interval')
pl.xlabel('$x$')
pl.ylabel('$y$')
pl.ylim(-10, 20)
pl.legend(loc='upper left')

#---------------------------------------------------------------------------------------------------------
# noisy case
X = np.linspace(0.1, 9.9, 20)
X = np.atleast_2d(X).T

# Observations and noise
y = f(X).ravel()
dy = 0.5 + 1.0 * np.random.random(y.shape)
noise = np.random.normal(0, dy)
y += noise

# Mesh input space for evaluations of real function, prediction, and its variance
x = np.atleast_2d(np.linspace(0, 10, datasize)).T

# Instantiate GP model
try:
    gp = GaussianProcess(corr='squared_exponential', theta0=1e-1,
                         thetaL=1e-3, thetaU=1,
                         nugget=(dy / y) ** 2,
                         random_start=100)
except:
    gp = GaussianProcess()

# Fit to data using MLE of params
gp.fit(X, y)

# Make prediction on meshed x-axis (ask for variance as well)
try:
    y_pred, variance = gp.predict(x, eval_MSE=True)
except:
    y_pred, variance = gp.predict(x)
sigma = np.sqrt(variance)

# Plot function, prediction, 95% confidence interval based on variance
fig = pl.figure()
pl.plot(x, f(x), 'r:', label=u'$f(x) = x\,\sin(x)$')
pl.errorbar(X.ravel(), y, dy, fmt='r.', markersize=10, label=u'Observations')
pl.plot(x, y_pred, 'b-', label=u'Prediction')
pl.fill(np.concatenate([x, x[::-1]]),
        np.concatenate([y_pred - confidence * sigma,
                       (y_pred + confidence * sigma)[::-1]]),
        alpha=.5, fc='b', ec='None', label='95% confidence interval')
pl.xlabel('$x$')
pl.ylabel('$f(x)$')
pl.ylim(-10, 20)
pl.legend(loc='upper left')

pl.show()
