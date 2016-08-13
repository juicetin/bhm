from ML.gp.gp import GaussianProcess
from ML.helpers import partition_indexes

import numpy as np
import math

class GP_ensembles():
    def __init__(self, expert_size=200):
        self.expert_size = expert_size
        pass
    
    def fit(self, X, y):
        if type(y[0]) != np.int64:
            self.gp_type = 'regression'
        else:
            self.gp_type = 'classification'

        self.X = X

        self.num_classes = np.unique(y).shape[0]
    
        # Shuffle the data in a separate copy
        # TODO need to 'organise' these into 
        shuf_idxs = np.arange(y.shape[0])
        np.random.shuffle(shuf_idxs)
        X_s = X[shuf_idxs]
        y_s = y[shuf_idxs]
        # 200 data points per expert classifier
        expert_count = math.ceil(X_s.shape[0]/self.expert_size)
    
        # Create and train all local GP experts
        gp_experts = np.full(expert_count, GaussianProcess(), dtype='object')
        idxs = partition_indexes(X_s.shape[0], expert_count)
        for gp_expert, (start, end) in zip(gp_experts, idxs):
            gp_expert.fit(X_s[start:end], y_s[start:end])
        self.gp_experts = gp_experts
    
    def gp_means_vars(self, x, parallel=False):
        # for gp_expert in self.gp_experts:
        #     gp_expert.predict(x)
    
        # Means, variances for each binary class case for each GP regressor (classifier)
        # Shape - (experts, 2, classes, data points)
        #   2 - 0-axis for means, 1-axis for variances
        y_preds = np.array([gp_expert.predict(x, keep_probs=True, parallel=parallel) for gp_expert in self.gp_experts])
    
        vars_gp = y_preds[:,1]
        means_gp = y_preds[:,0]
    
        return means_gp, vars_gp
