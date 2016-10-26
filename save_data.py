import sys
import numpy as np
from utils import load_data
from ML.gp.poe import PoGPE
from ML.gp.gpoe import GPoGPE
from ML.gp.bcm import BCM
from ML.gp.rbcm import rBCM

def info(type, value, tb):

    """
    This function is for replacing Python's default exception hook
    so that we drop into the debugger on uncaught exceptions to be
    able to immediately identify the issue/potentially recover
    operations
    """
    import traceback, pdb
    traceback.print_exception(type, value, tb)
    print
    pdb.pm()
sys.excepthook = info

red_features, red_mlabels4, red_mlabels24, red_coords, red_scoords, \
    red_sfeatures, red_slabels4, red_slabels24 = load_data.load_reduced_data()
qp_red_features = np.load('data/red_qp_features.npy')

test = False
if test == True:
    subset = 400
    red_sfeatures = red_sfeatures[:subset]
    red_slabels4 = red_slabels4[:subset]
    red_slabels24 = red_slabels24[:subset]
    qp_red_features = qp_red_features[:subset]

###### PoGPE ######
poe = PoGPE()
poe.fit(red_sfeatures, red_slabels4)
poe_4p = poe.predict(qp_red_features, True)
np.save('preds/poe_4p', poe_4p)
del(poe_4p)

poe = PoGPE()
poe.fit(red_sfeatures, red_slabels24)
poe_24p = poe.predict(qp_red_features, True)
np.save('preds/poe_24p', poe_24p)
del(poe_24p)

###### GPoGPE ######
gpoe = GPoGPE()
gpoe.fit(red_sfeatures, red_slabels4, False)
gpoe_4p = gpoe.predict(qp_red_features, False)
np.save('preds/gpoe_4p', gpoe_4p)
del(gpoe_4p)

gpoe = GPoGPE()
gpoe.fit(red_sfeatures, red_slabels24)
gpoe_24p = gpoe.predict(qp_red_features, True)
np.save('preds/gpoe_24p', gpoe_24p)
del(gpoe_24p)

###### BCM ######
bcm = BCM()
bcm.fit(red_sfeatures, red_slabels4)
bcm_4p = bcm.predict(qp_red_features, True)
np.save('preds/bcm_4p', bcm_4p)
del(bcm_4p)

bcm = BCM()
bcm.fit(red_sfeatures, red_slabels24)
bcm_24p = bcm.predict(qp_red_features, True)
np.save('preds/bcm_24p', bcm_24p)
del(bcm_24p)

###### rBCM ######
rbcm = rBCM()
rbcm.fit(red_sfeatures, red_slabels4)
rbcm_4p = rbcm.predict(qp_red_features, True)
np.save('preds/rbcm_4p', rbcm_4p)
del(rbcm_4p)

rbcm = rBCM()
rbcm.fit(red_sfeatures, red_slabels24)
rbcm_24p = rbcm.predict(qp_red_features, True)
np.save('preds/rbcm_24p', rbcm_24p)
del(rbcm_24p)
