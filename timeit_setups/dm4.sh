SETUP="
import numpy as np
from ML.dir_mul.nicta.dirmultreg import dirmultreg_learn, dirmultreg_predict, predict_parallel
from utils.data_transform import features_squared_only
red_features = np.load('data/red_features.npy')
f_sq2r = features_squared_only(red_features)
query_sn = np.load('data/query_sn.npy')
q_sq2r = features_squared_only(query_sn)
l4 = np.load('data/red_mlabels4.npy')
W = dirmultreg_learn(f_sq2r, l4)
"

DM4_PREDS="
dm_preds = dirmultreg_predict(q_sq2r, W)"

echo "dirichlet multinomial regression fit and predict full query set for 4-label case"
python -m timeit -v -n 10 -s "$SETUP" "$DM4_PREDS"


