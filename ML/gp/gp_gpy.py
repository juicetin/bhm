import GPy
import numpy as np

class GPyC:
    def __init__(self):
        pass

    def fit(self, X, C):
        """
        Fit the GPy-wrapper classifier 
        """
        K = GPy.kern.Matern32(X.shape[1])
        uniq_C = np.unique(C)
        self.models = []
        for c in uniq_C:
            labels = np.array([1 if c == label else 0 for label in C])[:,np.newaxis]
            m = GPy.models.GPRegression(X, labels, kernel=K.copy())
            self.models.append(m)
        self.models = np.array(self.models)

    def predict(self, x):
        """
        Make predictions using the GPy-wrapper classifier
        """
        all_preds = np.empty((self.models.shape[0], x.shape[0]))
        all_vars = np.empty(all_preds.shape)
        for i, m in enumerate(self.models):
            print("On label {}".format(i))
            gp_preds, gp_vars = m.predict(x)
            all_preds[i] = gp_preds.flatten()
            all_vars[i] = gp_vars.flatten()

        return all_preds, all_vars
