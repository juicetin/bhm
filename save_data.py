import sys
import numpy as np
from utils import load_data
from ML.gp.poe import PoGPE
from ML.gp.gpoe import GPoGPE
from ML.gp.bcm import BCM
from ML.gp.rbcm import rBCM
from datetime import datetime

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

def class_to_str(cls):
    return str(cls).split('.')[-1][:-2]

def save_for_ensemble(e_GP, train_features, train_labels, pred_features):
    label_count = np.unique(train_labels).shape[0]
    print('Now doing {} with {} labels'.format(class_to_str(e_GP), label_count))
    t1 = datetime.now()
    model = e_GP()
    model.fit(train_features, train_labels, True)
    preds = model.predict(pred_features, True)
    np.save('preds/{}{}_p'.format(class_to_str(e_GP), label_count), preds)
    t2 = datetime.now()
    print('{} took {} to train and predict all query points with {} labels'.format(class_to_str(e_GP), t2-t1, label_count))
    print()

red_features, red_mlabels4, red_mlabels24, red_coords, red_scoords, \
    red_sfeatures, red_slabels4, red_slabels24 = load_data.load_reduced_data()
qp_red_features = np.load('data/red_qp_features.npy')

test = True
if test == True:
    subset = 400
    red_sfeatures = red_sfeatures[:subset]
    red_features = red_features[:subset]
    red_slabels4 = red_slabels4[:subset]
    red_slabels24 = red_slabels24[:subset]
    red_mlabels4 = red_mlabels4[:subset]
    red_mlabels24 = red_mlabels24[:subset]
    qp_red_features = qp_red_features[:subset]

red_mlabels4 = red_mlabels4.argmax(axis=1)
red_mlabels24 = red_mlabels24.argmax(axis=1)

###### PoGPE ######
save_for_ensemble(PoGPE, red_features, red_mlabels4, qp_red_features)
save_for_ensemble(PoGPE, red_features, red_mlabels24, qp_red_features)

###### GPoGPE ######
save_for_ensemble(GPoGPE, red_features, red_mlabels4, qp_red_features)
save_for_ensemble(GPoGPE, red_features, red_mlabels24, qp_red_features)

# ###### BCM ######
save_for_ensemble(BCM, red_features, red_mlabels4, qp_red_features)
save_for_ensemble(BCM, red_features, red_mlabels24, qp_red_features)

# ###### rBCM ######
save_for_ensemble(rBCM, red_features, red_mlabels4, qp_red_features)
save_for_ensemble(rBCM, red_features, red_mlabels24, qp_red_features)
