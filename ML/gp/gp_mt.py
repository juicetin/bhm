from ML.gp.gp import GaussianProcess
from scipy.spatial.distance import cdist
import numpy as np
from scipy.optimize import minimize
import pdb

class GPMT(GaussianProcess):
    
    def fit(self, X, y):
        self.N = X.shape[0]
        self.M = X.shape[1]

        self.X = X
        self.y = y
        # NxM matrix Y such that y = vecY
        # y_il is response for l-th task on i-th input x_i
        self.Y = y.reshape(X.shape)


        ########### Initialisation of parameters/variables #############
        x0 = np.random.rand(2 + X.shape[1]) # Initial guess
        # x0 = np.array([1] + [1] * X.shape[1] + [0.1])
        self.f_err, self.l_scales, self.n_err = self.unpack_GP_args(x0)
        self.Kx = self.Kx_update(self.X, self.X, self.f_err, self.l_scales, self.n_err)
        self.Kf = 1/self.N * self.Y.T.dot(np.linalg.inv(self.Kx)).dot(self.Y)
        self.sigma_ls = np.full(self.Kf.shape, 0.01)
        self.F = self.F_update(X, self.Kf, self.Kx, np.diag(1/self.sigma_ls))
        ################################################################

        # Optimise over parameters for multi-task GP
        bounds = [[0,None]] * (X.shape[1] + 2)
        res = minimize(self.L_comp_NLL, x0, method='l-bfgs-b', bounds=bounds)

        # TODO don't use minimize, use EM

        self.f_err, self.l_scales, self.n_err = self.unpack_GP_args(res['x'])
        print(res['x'])

    def predict(self, x):
        Kf = self.prev_Kf()
        Kx = self.Kx_update(self.X, x, self.f_err, self.l_scales, self.n_err)
        D = np.diag(self.prev_sigma_ls())
        F = self.F_update(x, Kf, Kx, D)
        sigma_ls = self.sigma_ls_update(x, F)
        pdb.set_trace()
        return F, sigma_ls

    # Returns lower triangular Cholesky decomposition for complete-data log-likelihood
    def L_comp_NLL(self, args):

        f_err, l_scales, n_err = self.unpack_GP_args(args)

        N = self.N
        M = self.M

        # Build using old vars/params
        Kx = self.Kx_update(self.X, self.X, f_err, l_scales, n_err) # Kx using previous params
        F = self.prev_F()
        Kf = self.Kf_update(self.X, f_err, l_scales, n_err, F) # TODO Kf using previous F, Kx

        # Built using new params/vars
        D = np.diag(1/self.prev_sigma_ls())
        F = self.F_update(self.X, Kf, Kx, D); 
        YmF = self.Y-F
        sigma_ls = 1/N * YmF.T.dot(YmF) # TODO how to initialise these to 0.01 on first pass?
        D_inv = 1/sigma_ls
        # D_inv = np.diag(1/sigma_ls) # We know D is a diag matrix. Unneeded - D = np.diag(sigma_ls); 

        # pdb.set_trace()

        # TODO remove this try-catch in future
        # Usual case
        if Kf.shape != (1,1):
            L_comp_ll = - N/2 * np.log(np.linalg.det(Kf)) \
                        - M/2 * np.log(np.linalg.det(self.Kx)) \
                        - 0.5 * np.trace(np.linalg.inv(Kf).dot(F.T).dot(np.linalg.inv(self.Kx).dot(F))) \
                        - N/2 * np.sum(np.log(sigma_ls)) \
                        - 0.5 * np.trace((YmF).dot(D_inv).dot((YmF).T)) \
                        - M*N/2 * np.log(2*np.pi)
        # 1d data - only 1 task
        else:
            L_comp_ll = - N/2 * np.log(Kf) \
                        - M/2 * np.log(np.linalg.det(self.Kx)) \
                        - 0.5 * np.trace((1/Kf).dot(F.T).dot(np.linalg.inv(self.Kx).dot(F))) \
                        - N/2 * np.sum(np.log(sigma_ls)) \
                        - 0.5 * np.trace((YmF).dot(D_inv).dot(YmF.T)) \
                        - M*N/2 * np.log(2*np.pi)

        self.update_cur_to_prev(F, Kf, sigma_ls)

        return -L_comp_ll

    def update_cur_to_prev(self, F, Kf, sigma_ls):
        self.F = F
        self.Kf = Kf
        self.sigma_ls = sigma_ls

    # Returns updated hyperparameters by finding argmin of the log likelihood
    def multi_task_update_theta(self, thetas):
        res = minimize(self.multi_task_theta_ll, thetas, method='bfgs')
        return res['x']

    def thetas_NLL(self, args):
        f_err, l_scales, n_err = self.unpack_GP_args(args)

        N  = self.N
        M  = self.M
        Kf = self.Kf_update(self.X, f_err, l_scales, n_err, self.prev_Kf())
        F  = self.F_update(self.X, Kf, self.prev_Kx())
        Kx = self.Kx_update(self.X, self.X, self.f_err, self.l_scales, self.n_err)
        Kx_inv = np.linalg.inv(Kx)

        theta_xs = N * np.log(np.linalg.det(F.T.dot(Kx_inv).dot(F))) + M * np.log(np.linalg.det(Kx))
        return theta_xs

    # Returns updated hyperparams
    def thetas_argmin(self, x, f_err, l_scales, n_err, F):
        x0 = [f_err] + l_scales + [n_err]
        bounds = [[0,None]] * (X.shape[1] + 2)
        res = minimize(self.thetas_NLL, x0, method='l-bfgs-b', bounds=bounds)

        f_err, l_scales, n_err = self.unpack_GP_args(res['x'])

        return f_err, l_scales, n_err

    # Returns updated K^f matrix, given x, current hyperparameters, and Function values
    def Kf_update(self, x, f_err, l_scales, n_err, F):
        Kx = self.Kx_update(self.X, x, f_err, l_scales, n_err)
        N = self.N
        Kf = 1/N * F.T.dot(np.linalg.inv(Kx)).dot(F)
        return Kf

    # Returns updated noise over tasks
    def sigma_ls_update(self, x, F):
        N = self.N
        diff = self.Y-F
        sigma_ls = 1/N * diff.T.dot(diff)
        return sigma_ls
    
    # Vector of function values corresponding to Y
    #   Inference is needed when learning hyperparameters for noisy observations
    def F_update(self, x, Kf, Kx, D):
        I = np.identity(x.shape[0])
        sigma = np.kron(Kf, Kx) + np.kron(D, I)

        means = np.kron(Kf, Kx).T.dot(np.linalg.inv(sigma)).dot(self.y)
        return means

    # Covariance functions over inputs
    # NOTE stationary covariance functions as K^f explains variance
    #   unit variance, (zero mean?)
    def Kx_update(self, X, x, f_err, l_scales, n_err):
        # TODO follow NOTE from comment above function
        return self.K_se(X, x, f_err, l_scales)

    # Getters
    def prev_thetas(self):
        return self.f_err, self.l_scales, self.n_err

    def prev_F(self):
        return self.F

    def prev_Kf(self):
        return self.Kf

    def prev_Kx(self):
        return self.Kx

    def prev_sigma_ls(self):
        return self.sigma_ls

