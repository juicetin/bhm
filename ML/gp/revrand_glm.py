import numpy as np
from revrand import GeneralisedLinearModel
from revrand.btypes import Parameter, Positive
from revrand.basis_functions import RandomMatern52, BiasBasis, RandomRBF
from revrand.likelihoods import Bernoulli
from revrand.optimize import AdaDelta, Adam

# TODO
# bump up nbases
# check what nbases
# lengthscale 10 is quite high -lower?
# try diff polynomial fns
def revrand_glm(nbases=50, lenscale=10., regulariser=1, maxiter=3000, batch_size=10, num_features=11):
    # Algorith settings
    updater = Adam()

    # Bounded variables
    regulariser_init = Parameter(regulariser, Positive())
    lenscale_init = Parameter(lenscale, Positive())

    # Feature Transform
    # basis = RandomMatern52(nbases, num_features, lenscale_init=lenscale_init) + BiasBasis()
    basis = RandomRBF(nbases, num_features, lenscale_init=lenscale_init) + BiasBasis()

    llhood = Bernoulli()
    glm = GeneralisedLinearModel(llhood,
            basis,
            regulariser=regulariser_init,
            maxiter=maxiter,
            batch_size=batch_size,
            updater=updater
            )
    # glm.fit(features, labels)
    return glm

class RGLM:

    def fit(self, X, y):
        uniq_y = np.unique(y)
        glms = np.full(uniq_y.shape, revrand_glm(num_features=X.shape[1]), dtype=np.object)
        for i in range(uniq_y.shape[0]):
            glms[i].fit(X, np.array([1 if label == uniq_y[i] else 0 for label in y]))

        self.glms = glms

    def predict(self, x):
        preds = np.array([glm.predict(x) for glm in self.glms])
        return preds
