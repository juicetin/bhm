from ML.gp.gp import GaussianProcess
from scipy.spatial.distance import cdist
import numpy as np
from scipy.optimize import minimize
import pdb
import math
import sys

from sklearn.preprocessing import normalize
from sklearn.preprocessing import scale

class GPMT(GaussianProcess):

    # Fit the GP
    def fit(self, X, y):

        self.N = X.shape[0]
        self.M = X.shape[1]

        self.X = X
        self.y = y
        # NxM matrix Y such that y = vecY
        # y_il is response for l-th task on i-th input x_i
        self.Y = y.reshape(X.shape)

        ########### Initialisation of parameters/variables #############
        # x0 = np.random.rand(2 + X.shape[1]) # Initial guess
        x0 = np.array([1] + [1] * X.shape[1] + [0.1])
        f_err, l_scales, n_err = self.unpack_GP_args(x0)
        self.Kx = self.Kx_update(self.X, self.X, f_err, l_scales, n_err)
        self.Kf = 1/self.N * self.Y.T.dot(self.inverse(self.Kx)).dot(self.Y) # NOTE ??? too big!!
        # self.Kf = np.ones_like(self.Kf)
        self.sigma_ls = np.full(self.Kf.shape, 0.01)
        self.F = self.F_update(self.Kf, self.Kx, self.Kx, np.diag(1/self.sigma_ls))
        self.singular_count = 0

        ################################################################

        self.f_err, self.l_scales, self.n_err = self.MT_EM(f_err, l_scales, n_err)
        print(self.singular_count)

    # Predict a set of x points
    def predict(self, x):

        f_err, l_scales, n_err = self.prev_thetas()

        Kf = self.prev_Kf()
        Kx = self.Kx_update(self.X, self.X, f_err, l_scales, n_err)
        D = np.diag(self.prev_sigma_ls())
        K_star = self.Kx_update(self.X, x, f_err, l_scales, n_err, training=False) # TODO

        # Predictive means
        F = self.F_update(Kf, K_star, Kx, D)

        # Predictive variances
        L = self.L_create(self.X, f_err, l_scales, n_err)
        var = self.predictive_variances(x, L, K_star, f_err, l_scales, n_err)

        # pdb.set_trace()

        # TODO variances!
        return F[:,0], var

    # Store prev values of hyperparams to keep track of convergence
    def cache_prev_hyperparams(self, f_err, l_scales, n_err):
        self.prev_f_err = f_err
        self.prev_l_scales = l_scales
        self.prev_n_err = n_err

    # Check whether hyperparams have moved less than some tolerance value
    def hyperparams_stabilised(self, f_err, l_scales, n_err, tol):
        try:
            cur = np.array([f_err] + [l_scales] + [n_err])
            prev = np.array([self.prev_f_err] + [self.prev_l_scales] + [self.prev_n_err])
            diff = np.abs(cur-prev < tol)
            return np.sum(diff) == cur.shape[0]
        except:
            False

    # Expectectation Maximisation over thetas, Kf, task_noises
    def MT_EM(self, f_err, l_scales, n_err, tol=1e-5, n_iter=None):
        diff = 1
        prev_L_LL = sys.maxsize # Could calculate first iteration - but extra code overhead
        if n_iter != None:
            cur_iters = 1
        cur_iters = 1

        # EM until allowed difference is below tolerance threshold
        while not self.hyperparams_stabilised(f_err, l_scales, n_err, tol):
            print(cur_iters, end=" ", flush=True)
            cur_iters+=1

            # get argmin on thetas
            self.cache_prev_hyperparams(f_err, l_scales, n_err)
            f_err, l_scales, n_err = self.thetas_argmin(f_err, l_scales, n_err, self.prev_F())
            print("EM iteration hyperparams: ", end=" ", flush=True)
            print(f_err, l_scales, n_err)

            # # TODO (no?) calculate new Kx - done within the new Kf
            Kx = self.Kx_update(self.X, self.X, f_err, l_scales, n_err)

            # # TODO calculate new F, as new Kx has changed (no ?)
            # F = self.F_update(self.X, self.prev_Kf(), Kx, D)

            # TODO calculate new Kf
            Kf = self.Kf_update(self.X, f_err, l_scales, n_err, self.prev_F())

            # TODO calculate new sigma_ls
            D = np.diag(self.prev_sigma_ls())
            F = self.F_update(Kf, Kx, Kx, D)

            sigma_ls = self.sigma_ls_update(self.X, F)

            cur_L_LL = self.L_LL(Kf, Kx, F, sigma_ls)
            diff = np.abs(cur_L_LL - prev_L_LL)
            prev_L_LL = cur_L_LL

            self.update_cur_to_prev(F, Kf, sigma_ls)

            print(prev_L_LL)

        return f_err, l_scales, n_err
    
    # L_comp log likelihood
    def L_LL(self, Kf, Kx, F, sigma_ls):
        YmF = self.Y - F
        D_inv = np.diag(1/sigma_ls)
        M = self.M
        N = self.N

        Kx_inv = self.inverse(Kx)

        if Kf.shape != (1,1):
            L_ll = - N/2 * np.log(np.linalg.det(Kf)) \
                   - M/2 * np.log(np.linalg.det(self.Kx)) \
                   - 0.5 * np.trace(np.linalg.inv(Kf).dot(F.T).dot(Kx_inv.dot(F))) \
                   - N/2 * np.sum(np.log(sigma_ls)) \
                   - 0.5 * np.trace((YmF).dot([D_inv]).dot((YmF).T)) \
                   - M*N/2 * np.log(2*np.pi)
        # 1d data - only 1 task
        else:
            # pdb.set_trace()
            L_ll = - N/2 * np.log(Kf) \
                   - M/2 * np.log(np.linalg.det(self.Kx)) \
                   - 0.5 * np.trace((np.linalg.inv(Kf)).dot(F.T).dot(Kx_inv.dot(F))) \
                   - N/2 * np.sum(np.log(sigma_ls)) \
                   - 0.5 * np.trace((YmF).dot([D_inv]).dot(YmF.T)) \
                   - M*N/2 * np.log(2*np.pi)

        return L_ll

    def update_cur_to_prev(self, F, Kf, sigma_ls):
        self.F = F
        self.Kf = Kf
        self.sigma_ls = sigma_ls

    # > 75% time spent in this function
    def thetas_NLL(self, args):
        f_err, l_scales, n_err = self.unpack_GP_args(args)
        # print("cur iter: ", end=" ", flush=True)
        # print(f_err, l_scales, n_err)

        N  = self.N
        M  = self.M
        Kx = self.Kx_update(self.X, self.X, f_err, l_scales, n_err)
        Kf = self.Kf_update(self.X, f_err, l_scales, n_err, self.prev_F())
        D = np.diag(self.prev_sigma_ls())
        F  = self.F_update(Kf, Kx, Kx, D)

        Kx_inv = self.inverse(Kx)
        Kf_det = np.linalg.det(F.T.dot(Kx_inv).dot(F))
        Kx_det = np.linalg.det(Kx)

        theta_xs = N * np.log(Kf_det) + M * np.log(Kx_det)
        return theta_xs

    # Returns updated hyperparams
    def thetas_argmin(self, f_err, l_scales, n_err, F):
        # NOTE l_scales only accounts for 1-D data here
        # print("Warning - [l_scales] only accounts for 1-D data here")
        x0 = [f_err] + [l_scales] + [n_err]

        bounds = [[0,None]] * (self.X.shape[1] + 2)
        res = minimize(self.thetas_NLL, x0, method='l-bfgs-b', bounds=bounds)

        f_err, l_scales, n_err = self.unpack_GP_args(res['x'])

        return f_err, l_scales, n_err

    # Adjusts for singular matrices ._.
    def inverse(self, m):
        # cond_num = np.linalg.cond(m)
        # if np.isfinite(cond_num): # and cond_num < 1/sys.float_info.epsilon:
        #     inv = np.linalg.inv(m)
        # else:
        #     print('SINGULAR MATRIX')
        #     # pdb.set_trace()
        #     # While loop here
        #     inv = np.linalg.inv(m + np.diag([0.1] * m.shape[0]))
        try:
            return np.linalg.inv(m)
        except:
            return np.linalg.inv(m+ np.diag([0.01] * m.shape[0]))

    # Returns updated K^f matrix, given x, current hyperparameters, and Function values
    def Kf_update(self, x, f_err, l_scales, n_err, F):
        Kx = self.Kx_update(self.X, x, f_err, l_scales, n_err)
        N = self.N

        # Adjust matrix if singular - NOTE results break if singular matrices DO occur
        Kx_inv = self.inverse(Kx)
        Kf = 1/N * F.T.dot(np.linalg.inv(Kx_inv)).dot(F) # NOTE singular matrix Kx here sometimes!
        return Kf

    # Returns updated noise over tasks
    def sigma_ls_update(self, x, F):
        N = self.N
        diff = self.Y-F
        sigma_ls = 1/N * diff.T.dot(diff)
        return sigma_ls
    
    # Vector of function values corresponding to Y
    #   Inference is needed when learning hyperparameters for noisy observations

    def F_update(self, Kf, Kx_star, Kx, D):
        I = np.identity(Kx.shape[0])
        sigma = np.kron(Kf, Kx) + np.kron(D, I)
        sigma_inv = self.inverse(sigma)
        means = np.kron(Kf, Kx_star).T.dot(sigma_inv).dot(self.y)
        return means

    # Covariance functions over inputs
    # NOTE stationary covariance functions as K^f explains variance
    #   unit variance, (zero mean?)
    def Kx_update(self, X, x, f_err, l_scales, n_err, training=True):
        # TODO follow NOTE from comment above function
        K = self.K_se(X, x, f_err, l_scales) 
        if training == True:
            K = K + n_err**2 * np.identity(X.shape[0])
        return K

    # Getters
    def prev_thetas(self):
        return self.f_err, self.l_scales, self.n_err

    def prev_F(self):
        return self.F

    def prev_Kf(self):
        print("Kf: {}".format(self.Kf))
        return self.Kf

    def prev_Kx(self):
        return self.Kx

    def prev_sigma_ls(self):
        return self.sigma_ls
