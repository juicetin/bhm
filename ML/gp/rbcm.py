from ML.gp.gp import GaussianProcess
from ML.helpers import partition_indexes
from ML.gp.gp_ensemble_estimators import GP_ensembles

import numpy as np
import math

class rBCM(GP_ensembles):
    def __init__(self, args):
        super().__init__(args)

    def predict(self, x, keep_probs=False):
        means_gp, vars_gp = self.gp_means_vars(x)

        # These contain a row for each binary class case (OvR)
        #   AFTER summing along axis 0 (each of the local experts)
        betas = []
        M = vars_gp.shape[0]
        vars_bcm_sm2 = np.sum(vars_gp**(-2) + (1-M) * vars_gp**(-2), axis=0) # vars
        means_bcm = vars_bcm**2 * np.sum(vars_gp**(-2) * means_gp, axis=0)  # means

        if keep_probs == True:
            return means_bcm, vars_bcm

        return np.argmax(means_bcm, axis=0)

