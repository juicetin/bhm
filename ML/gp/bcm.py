from ML.gp.gp import GaussianProcess
from ML.helpers import partition_indexes
from ML.gp.gp_ensemble_estimators import GP_ensembles

class BCM(GP_ensembles):
    def __init__(self, args):
        super().__init__(args)
