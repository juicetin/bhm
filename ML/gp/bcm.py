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

        # These contain a row for each binary class case (OvR)
        #   AFTER summing along axis 0 (each of the local experts)
        M = vars_gp.shape[0]
        gaussian_precision = vars_gp**(-1)

        # TODO prior precision - squared element-wise inverse of diagonal along covariance matrix
        # prior_precision = vars_gp**(-2) # NOTE WRONG - prior precision is diag of elementwise inverse of cov matrix
        prior_variances = np.empty_like(gaussian_precision)
        for idx, gp_expert in enumerate(self.gp_experts):
            prior_variance = gp_expert.prior_variance(x)
            prior_variances[idx] = prior_variance
        prior_precisions = prior_variances**(-1)

        vars_bcm_inv = np.sum(gaussian_precision + (1-M) * prior_precisions, axis=0) # variances
        vars_bcm = vars_bcm_inv**(-1)
        means_bcm = vars_bcm * np.sum(gaussian_precision * means_gp, axis=0)  # means

        if keep_probs == True:
            return means_bcm, vars_bcm

        return np.argmax(means_bcm, axis=0)
