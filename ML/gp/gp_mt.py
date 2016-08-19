from ML.gp.gp import GaussianProcess
from scipy.spatial.distance import cdist
import numpy as np
from scipy.optimize import minimize

class GPMT(GaussianProcess):
    
    def fit_multi_task(self, X, y):
        self.N = self.X.shape[0]
        self.M = np.unique(y).shape[0]

        bounds = [[0,None]] * (X.shape[1] + 2)
        x0 = np.random.rand(2 + X.shape[1])
        res = minimize(self.L_comp_NLL, x0, method='l-bfgs-b', bounds=bounds)

        args = self.unpack_GP_args(res['x'])

    def predict_multi_task(self, x):
        Kf
        Kx
        task_variances
        D = np.diag(task_variances)
        I = np.identity(Kf.shape[0]) # Not sure about shape of identity matrix here
        sigma = np.kron(Kf, Kx) + np.kron(D, I)

        means = np.kron(Kf, Kx).T.dot(np.inv(sigma)).dot(y)
        return means

    # Returns lower triangular Cholesky decomposition for complete-data log-likelihood
    def L_comp_NLL(self, args):
        f_err, l_scales, n_err = self.unpack_GP_args(args)

        N = self.N
        M = self.M
        Kf = self.Kf()
        Kx = self.Kx()
        F = self.F()
        Y = self.Y()
        task_variances
        D = np.diag(task_variances)
        task_variances
        L_comp_ll = -N/2 * np.log(Kf) - M/2 * log(Kx) - 0.5 * np.trace(np.inv(Kf).dot(F.T).dot(np.inv(Kx).dot(F))) \
                            - N/2 np.sum(np.log(task_variances) - 0.5 * np.trace((Y-F).dot(np.inv(D)).dot((Y-F).T)) - M*N/2 * np.log(2*np.pi)) 
        return -L_comp_ll

    # Returns updated hyperparameters by finding argmin of the log likelihood
    def multi_task_update_theta(self, thetas):
        res = minimize(self.multi_task_theta_ll, thetas, method='bfgs')
        return res['x']

    # Returns updated hyperparams
    def thetas(self):
        hyperparams
        Kx
        N = self.N
        F
        M = self.M
        theta_xs = N * np.abs(np.log(F.T.dot(np.inv(Kx)).dot(F))) + M * np.log(Kx)
        return theta_xs

    # Returns updated K^f matrix
    def Kf(self, f_err, l_scales, n_err):
        Kx = self.Kx(f_err, l_scales, n_err)
        N = self.N
        F
        Kf = 1/N * F.T.dot(np.inv(Kx)).dot(F)
        return Kf

    # Returns updated noise over tasks
    def task_noise(self):
        N = self.N
        Y
        F
        task_variances = 1/N * (Y-F).T.dot(Y-F)
        return task_variances

    # Covariance functions over inputs
    # NOTE stationary covariance functions as K^f explains variance
    #   unit variance, (zero mean?)
    def Kx(self, f_err, l_scales, n_err):
        # TODO follow NOTE from comment above function
        return self.K_se(f_err, l_scales, n_err)
    
    # NxM matrix Y such that y = vecY
    # y_il is response for l-th task on i-th input x_i
    def Y(self):
        pass

    # Vector of function values corresponding to Y
    def F(self):
        pass

