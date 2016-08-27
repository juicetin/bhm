""" Various linear algebra utilities and nonlinear functions. """

import numpy as np
import scipy.linalg as la


def jitchol(A, lower=False):
    """ Do cholesky decomposition with a bit of diagonal jitter if needs be.

        Aarguments:
            A: a [NxN] positive definite symmetric matrix to be decomposed as
                A = L.dot(L.T).
            lower: Return lower triangular factor, default False (upper).

        Returns:
            An upper or lower triangular matrix factor, L, also [NxN].
            Also wheter or not a the matrix is lower triangular form,
            (L, lower).
    """

    # Try the cholesky first
    try:
        cholA = la.cholesky(A, lower=lower)
        return cholA, lower
    except la.LinAlgError:
        pass

    # Now add jitter
    D = A.shape[0]
    jit = 1e-16
    cholA = None
    di = np.diag_indices(D)
    Amean = A.diagonal().mean()

    while jit < 1e-3:

        try:
            Ajit = A.copy()
            Ajit[di] += Amean * jit
            cholA = la.cholesky(Ajit, lower=lower)
            break
        except la.LinAlgError:
            jit *= 10

    if cholA is None:
        raise la.LinAlgError("Added maximum jitter and A still not PSD!")

    return cholA, lower


def logdet(L):
    """ Compute the log determinant of a matrix.

        Arguments:
            L: The [NxN] cholesky factor of the matrix.

        Returns:
            The log determinant (scalar)
    """

    return 2 * np.log(L.diagonal()).sum()


def sdelete_rows(A, r):
    """ Delete rows from a sparse matrix. """

    N, D = A.shape
    keeprows = [i for i in range(N) if i not in r]
    form = A.format

    return (A.tocsc()[keeprows, :]).asformat(form)


def sdelete_cols(A, c):
    """ Delete columns from a sparse matrix. """

    N, D = A.shape
    keepcols = [i for i in range(D) if i not in c]
    form = A.format

    return (A.tocsr()[:, keepcols]).asformat(form)


def logsumexp(X, axis=0):
    """ Log-sum-exp trick for matrix X for summation along a specified axis """

    mx = X.max(axis=axis)
    return np.log(np.exp(X - mx[:, np.newaxis]).sum(axis=axis)) + mx


def softplus(X):
    """ Pass X through a soft-plus function, log(1 + exp(X)), in a numerically
        stable way (using the log-sum-exp trick).

        Arguments:
            X: shape (N,) array or shape (N, D) array of data.

        Returns:
            array of same shape of X with the result of softmax(X).
    """

    if np.isscalar(X):
        return logsumexp(np.vstack((np.zeros(1), [X])).T, axis=1)[0]

    N = X.shape[0]

    if X.ndim == 1:
        return logsumexp(np.vstack((np.zeros(N), X)).T, axis=1)
    elif X.ndim == 2:
        sftX = np.empty(X.shape, dtype=float)
        for d in range(X.shape[1]):
            sftX[:, d] = logsumexp(np.vstack((np.zeros(N), X[:, d])).T, axis=1)
        return sftX
    else:
        raise ValueError("This only works on up to 2D arrays.")


def softmax(X, axis=0):
    """ Pass X through a softmax function, exp(X) / sum(exp(X), axis=axis), in
        a numerically stable way using the log-sum-exp trick.
    """

    if axis == 1:
        return np.exp(X - logsumexp(X, axis=1)[:, np.newaxis])
    elif axis == 0:
        return np.exp(X - logsumexp(X, axis=0))
    else:
        raise ValueError("This only works on 2D arrays for now.")


def logistic(X):
    """ Pass X through a logistic sigmoid, 1 / (1 + exp(-X)), in a numerically
        stable way (using the log-sum-exp trick).

        Arguments:
            X: shape (N,) array or shape (N, D) array of data.

        Returns:
            array of same shape of X with the result of logistic(X).
    """

    N = X.shape[0]

    if X.ndim == 1:
        return np.exp(-logsumexp(np.vstack((np.zeros(N), -X)).T, axis=1))
    elif X.ndim == 2:
        lgX = np.empty(X.shape, dtype=float)
        for d in range(X.shape[1]):
            lgX[:, d] = np.exp(-logsumexp(np.vstack((np.zeros(N), -X[:, d])).T,
                               axis=1))
        return lgX
    else:
        raise ValueError("This only works on up to 2D arrays.")
