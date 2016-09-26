import GPy
import numpy as np

class GPyC:
    def __init__(self):
        pass

    def fit(X, C):
        """
        Fit the GPy-wrapper classifier 
        """
        K = GPy.kern.Matern32(1)
        uniq_C = np.unique(C)
        self.models = []
        for c in uniq_C:
            labels = np.array([1 if c == label else 0 for label in C_train.argmax(axis=1)])[:,np.newaxis]
            m = GPy.models.GPRegression(X_train, labels, kernel=K.copy())
            self.models.append(m)

    def predict(x):
        """
        Make predictions using the GPy-wrapper classifier
        """
        all_preds = np.empty((self.models.shape[0], x.shape[0]))
        all_vars = np.empty(all_preds.shape)
        for i, m in enumerate(self.models):
            gp_preds, gp_vars = m.predict(x)
            all_preds[i] = gp_preds
            all_vars[i] = all_vars
