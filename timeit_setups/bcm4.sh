SETUP="
import numpy as np
from ML.gp.bcm import BCM
red_sfeatures = np.load('data/red_features_singlelabel.npy')
red_slabels4 = np.load('data/red_labels4_single.npy')
qp_red_features = np.load('data/red_qp_features.npy')
"

BCM4="
bcm = BCM()
bcm.fit(red_sfeatures[:3000], red_slabels4[:3000], True)
bcm_p = bcm.predict(qp_red_features[:3000], True)
"

python -m timeit -v -n 1 -s "$SETUP" "$BCM4"
