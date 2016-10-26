SETUP="
from ML.gp.bcm import BCM
red_sfeatures = np.load('data/red_sfeatures.npy')
red_slabels24 = np.load('data/red_slabels24.npy')
"

BCM24="
bcm = BCM()
bcm.fit(red_sfeatures, red_slabels24)
bcm_p = bcm.predict(qp_red_features, True)
"

python -m timeit -v -n 3 -s "$SETUP" "$BCM24"
