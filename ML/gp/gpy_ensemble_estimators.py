from ML.gp import gp_gpy
from ML.helpers import partition_indexes

import numpy as np
import math
import pdb

class GP_ensembles():
    def __init__(self, expert_size=200):
        self.expert_size = expert_size
    
    def fit(self, X, y, parallel=False):
        # if type(y[0]) != np.int64:
        #     self.gp_type = 'regression'
        # else:
        #     self.gp_type = 'classification'

        # self.X = X

        # self.num_classes = np.unique(y).shape[0]
    
        # Shuffle the data in a separate copy
        # TODO need to 'organise' these into 
        shuf_idxs = np.arange(y.shape[0])
        np.random.shuffle(shuf_idxs)
        X_s = X[shuf_idxs]
        y_s = y[shuf_idxs]
        # 200 data points per expert classifier
        expert_count = math.ceil(X_s.shape[0]/self.expert_size)
    
        # Create and train all local GP experts
        gp_experts = np.full(expert_count, gp_gpy.GPyC(), dtype='object')
        idxs = partition_indexes(X_s.shape[0], expert_count)
        for gp_expert, (start, end) in zip(gp_experts, idxs):
            gp_expert.fit(X_s[start:end], y_s[start:end], parallel=parallel)
        self.gp_experts = gp_experts
    
    # Returns the means and variances for each GP expert
    def gp_means_vars(self, x, parallel=False):
    
        # Means, variances for each binary class case for each GP regressor (classifier)
        # Shape - (experts, 2, classes, data points)
        #   2 - 0-axis for means, 1-axis for variances
        y_preds = np.array([gp_expert.predict(x, parallel=parallel) for gp_expert in self.gp_experts])
    
        # Extract means and variances
        means_gp = y_preds[:,0]
        vars_gp = y_preds[:,1]
    
        return means_gp, vars_gp

