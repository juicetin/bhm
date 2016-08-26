import numpy as np
from scipy.misc import factorial

class DirichletMultinomialRegression:

    def fit(self, X, Y):
        self.X = X
        self.Y = Y
        pass

    def predict(self, X, y):
        pass

    def dirmult_log_likelihood(self, args):
        w
        alpha # exp {xT.w} ; data * weights
        phi # ??? variance of normally distributed weights?

        joint_ll = np.sum(                                # sum_n(
            np.log(np.sum(self.Y, axis=1)) -              # log(M_k)
            np.sum(np.log(factorial(self.Y)), axis=1) -   # sum (log(C_k!))
            np.log(np.math.gamma(
                np.sum(alpha * self.X, axis=1))
            ) -                                           # log(gamma(sum_k( alpha_k(x_n))))
            np.log(np.math.gamma(np.sum(self.Y + alpha))) # log(gamma(sum_k(C_nk + alpha_k(x_n))))
        ) + \
        np.sum(
            np.log(np.math.gamma(self.Y + alpha)) - 
            np.log(np.math.gamma(alpha))
        ) + \
        np.sum(
            -phi/2 * log(2*np.pi*phi) -
            0.5 * w.T.dot(phi).dot(w)
        )

        return -joint_ll

    def dirmult_log_likelihood_grad(self):
        joint_ll_grad = 
