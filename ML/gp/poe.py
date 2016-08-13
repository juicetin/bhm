from ML.gp.gp import GaussianProcess
from ML.helpers import partition_indexes
from ML.gp.gp_ensemble_estimators import GP_ensembles

import numpy as np
import math

class PoGPE(GP_ensembles):
    def __init__(self, args):
        super().__init__(args)

    def predict(self, x, keep_probs=False):
        means_gp, vars_gp = self.gp_means_vars(x)

        # These contain a row for each binary class case (OvR)
        #   AFTER summing along axis 0 (each of the local experts)
        vars_poe = np.sum(vars_gp, axis=0)  # vars
        means_poe = vars_poe * np.sum(vars_gp**(-2) * means_gp, axis=0)  # means

        if keep_probs == True:
            return means_poe, vars_poe

        if self.gp_type == 'classification':
            return np.argmax(means_poe, axis=0)
        elif self.gp_type == 'regression':
            return means_poe, vars_poe
