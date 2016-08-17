from ML.gp.gp import GaussianProcess

class GPRegressor(GaussianProcess):
    def __init__(self):
        pass

    # Regression fit
    def fit_regression(self, X, y):

        # Set basic data
        self.X = X  # Inputs
        self.y = y
        self.size = len(X)

        # Determine optimal GP hyperparameters
        # f_err, l_scales, n_err
        bounds = [[0,None]] * (X.shape[1] + 2)
        # x0 = [1000] + [50] * X.shape[1] + [0.1]
        x0 = np.random.rand(2 + X.shape[1])

        # res = minimize(self.SE_NLL, x0, method='bfgs')
        # res = minimize(self.SE_NLL, x0, method='bfgs', jac=self.SE_der)
        res = minimize(self.SE_NLL, x0, method='l-bfgs-b', jac=self.SE_der, bounds=bounds)

        # res['x'] = np.array([3076.7471, 100.58150154, 0.304933902061]) # NOTE hardcoded sample values 
        # res['x'] = np.array([3000, 100, 0.3])
        self.f_err, self.l_scales, self.n_err = self.unpack_GP_args(res['x'])

        # S et a few 'fixed' variables once GP HPs are determined for later use (with classifier)
        self.L = self.L_create(self.X, self.f_err, self.l_scales, self.n_err)
        self.K_inv = np.linalg.inv(self.L.T).dot(np.linalg.inv(self.L))
        self.alpha = linalg.solve(self.L.T, (linalg.solve(self.L, self.y)))

    def predict_regression(self, x, L=None, alpha=None, f_err=None, l_scales=None, n_err=None):

        # Assign hyperparameters and other calculated variables
        if L==None and alpha==None and f_err==None and l_scales==None:
            L = self.L
            alpha = self.alpha
            f_err = self.f_err
            l_scales = self.l_scales
            n_err = self.n_err

        # TODO fix - mean and var need to be calculated per point
        k_star = self.K_se(self.X, x, f_err, l_scales)
        f_star = k_star.T.dot(alpha)
        v = np.linalg.solve(L, k_star)
        var = self.K_se(x, x, f_err, l_scales) - v.T.dot(v)

        # Corner case with only one dimension 
        if len(f_star.shape) == 2 and f_star.shape[1] == 1:
            f_star = f_star.reshape(f_star.shape[0])

        return f_star, var


    def SE_der(self, args):
        # TODO fix - get around apparent bug
        # if len(args.shape) != 1:
        #     # print(args)
        #     args = args[0]

        f_err, l_scales, n_err = self.unpack_GP_args(args)
        # TODO use alpha calculated from SE_NLL

        L = self.L_create(self.X, f_err, l_scales, n_err)
        alpha = linalg.solve(L.T, (linalg.solve(L, self.y))) # save for use with derivative func
        aaT = alpha.dot(alpha.T)
        K_inv = np.linalg.inv(L.T).dot(np.linalg.inv(L))

        # Calculate dK/dtheta over each hyperparameter
        eval_dK_dthetas = self.eval_dK_dthetas(f_err, l_scales, n_err)

        # Incorporate each dK/dt into gradient
        derivatives = [float(-0.5 * np.matrix.trace((aaT - K_inv).dot(dK_dtheta))) for dK_dtheta in eval_dK_dthetas]
        return np.array(derivatives)

    # Args is an array to allow for scipy.optimize
    def SE_NLL(self, args):
        # TODO fix - get around apparent bug
        if len(args.shape) != 1:
            args = args[0]

        f_err, l_scales, n_err = self.unpack_GP_args(args)
        L = self.L_create(self.X, f_err, l_scales, n_err)
        
        alpha = linalg.solve(L.T, (linalg.solve(L, self.y))) # save for use with derivative func
        nll = (
            0.5 * self.y.T.dot(alpha) + 
            np.matrix.trace(np.log(L)) + # sum of diagonal
            0.5 * self.size * math.log(2*math.pi)
        )

        return nll
