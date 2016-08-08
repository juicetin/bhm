from ML.gp.gp import GaussianProcess
from ML.helpers import partition_indexes

import numpy as np
import math

class GPoGPE(PoGPE):
    def __init__(self):
        pass

    def predict(self, x, keep_probs=False):
        means_gp, vars_gp = self.gp_means_vars(x)

        # Expert contributions 
        expert_count = gp.gp_experts.shape[0]
        betas = np.full(expert_count, 1/expert_count)

        # These contain a row for each binary class case (OvR)
        vars_poe = np.sum(betas * vars_gp, axis=0)  # vars
        means_poe = vars_poe**2 * np.sum(betas * vars_gp**(-2) * means_gp, axis=0)  # means

        if keep_probs == True:
            return means_poe, vars_poe

        return np.argmax(means_poe, axis=0)
