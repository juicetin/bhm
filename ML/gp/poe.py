from ML.gp.gp import GaussianProcess
from ML.helpers import partition_indexes
from ML.gp.gp_ensemble_estimators import GP_ensembles

import numpy as np
import math

class PoGPE(GP_ensembles):
    def __init__(self, args):
        super().__init__(args)

    # def fit(self, X, y):
    #     self.num_classes = np.unique(y).shape[0]

    #     # Shuffle the data in a separate copy
    #     # TODO need to 'organise' these into 
    #     shuf_idxs = np.arange(y.shape[0])
    #     np.random.shuffle(shuf_idxs)
    #     X_s = X[shuf_idxs]
    #     y_s = y[shuf_idxs]
    #     # 200 data points per expert classifier
    #     expert_count = math.ceil(X_s.shape[0]/200)

    #     # Create and train all local GP experts
    #     gp_experts = np.full(expert_count, GaussianProcess(), dtype='object')
    #     idxs = partition_indexes(X_s.shape[0], expert_count)
    #     for gp_expert, (start, end) in zip(gp_experts, idxs):
    #         gp_expert.fit(X_s[start:end], y_s[start:end])
    #     self.gp_experts = gp_experts

    # def gp_means_vars(self, x):
    #     for gp_expert in self.gp_experts:
    #         gp_expert.predict(x)

    #     # Means, variances for each binary class case for each GP regressor (classifier)
    #     y_preds = np.array([gp_expert.predict(x, keep_probs=True, parallel=True, PoE=True) for gp_expert in self.gp_experts])
    #     self.y_preds = y_preds

    #     vars_gp = y_preds[:,:,1]
    #     means_gp = y_preds[:,:,0]

    #     return means_gp, vars_gp

    def predict(self, x, keep_probs=False):
        means_gp, vars_gp = self.gp_means_vars(x)

        # These contain a row for each binary class case (OvR)
        vars_poe = np.sum(vars_gp, axis=0)  # vars
        means_poe = vars_poe**2 * np.sum(vars_gp**(-2) * means_gp, axis=0)  # means

        print(vars_poe.shape)
        print(means_poe.shape)

        if keep_probs == True:
            return means_poe, vars_poe

        return np.argmax(means_poe, axis=0)
