from ML.gp.gp import GaussianProcess
from ML.helpers import partition_indexes
from ML.gp.gp_ensemble_estimators import GP_ensembles

import numpy as np
import math

class GPoGPE(GP_ensembles):
    def __init__(self, args):
        super().__init__(args)

    def predict(self, x, keep_probs=False):
        means_gp, vars_gp = self.gp_means_vars(x)

        # Expert contributions 
        expert_count = self.gp_experts.shape[0]
        betas = np.full(x.shape[0], 1/expert_count)

        # These contain a row for each binary class case (OvR)
        # NOTE betas can only be factored out as it is consistent throughout
        vars_poe = betas * np.sum(vars_gp, axis=0)  # vars
        means_poe = vars_poe * betas * np.sum(vars_gp**(-2) * means_gp, axis=0)  # means

        if keep_probs == True:
            return means_poe, vars_poe

        return np.argmax(means_poe, axis=0)
