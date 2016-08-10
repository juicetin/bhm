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
        gaus_prec = vars_gp**(-2)
        # prior_precision = vars_gp**(-2) # NOTE WRONG - prior precision is diag of elementwise inverse of cov matrix
        prior_precision = np.ones_like(vars_gp)
        vars_bcm_inv = np.sum(gaus_prec + (1-M) * prior_precision, axis=0) # variances
        vars_bcm = vars_bcm_inv**(-1)
        means_bcm = vars_bcm * np.sum(gaus_prec * means_gp, axis=0)  # means

        if keep_probs == True:
            return means_bcm, vars_bcm

        return np.argmax(means_bcm, axis=0)
