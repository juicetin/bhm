SETUP="
import numpy as np
from ML.dir_mul.nicta.dirmultreg import dirmultreg_learn, dirmultreg_predict, predict_parallel
from utils.data_transform import features_squared_only
red_features = np.load('data/red_features.npy')
f_sq2r = features_squared_only(red_features)
query_sn = np.load('data/query_sn.npy')
q_sq2r = features_squared_only(query_sn)
l4 = np.load('data/red_mlabels4.npy')
"

DM4_PREDS="
W = dirmultreg_learn(f_sq2r, l4)"

python -m timeit -v -n 10 -s "$SETUP" "$DM4_PREDS"


