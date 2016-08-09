from ML.gp.gp import GaussianProcess
from ML.helpers import partition_indexes
from ML.gp.gp_ensemble_estimators import GP_ensembles

import numpy as np
import math

class BCM(GP_ensembles):
    def __init__(self, args):
        super().__init__(args)

    def predict(self, x, keep_probs=False):
        means_gp, vars_gp = self.gp_means_vars(x)

        # NOTE actually correct but incredibly poor results below?
        # M = vars_gp.shape[0]
        # vars_bcm_sq_m2 = np.sum(vars_gp**(-2) + (1-M) * vars_gp**(-2), axis=0) # vars
        # vars_bcm_sq = vars_bcm_sq_m2**(-1)
        # means_bcm = vars_bcm_sq * np.sum(vars_gp**(-1) * means_gp, axis=0)  # means

        # These contain a row for each binary class case (OvR)
        #   AFTER summing along axis 0 (each of the local experts)
        M = vars_gp.shape[0]
        vars_bcm_inv = np.sum(vars_gp**(-2) + (1-M) * vars_gp**(-2), axis=0)  # vars
        means_bcm = vars_bcm*(-1) * np.sum(vars_gp**(-2) * means_gp, axis=0)  # means

        if keep_probs == True:
            return means_bcm, vars_bcm

        return np.argmax(means_bcm, axis=0)
