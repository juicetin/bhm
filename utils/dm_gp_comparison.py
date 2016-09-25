import numpy as np

from ML.gp.gp_gpy import GPyC

from utils.load_data import generate_dm_toy_ex
from ML.dir_mul.dirichlet_multinomial import DirichletMultinomialRegression
from ML.dir_mul.nicta.dirmultreg import dirmultreg_learn, dirmultreg_predict
from ML.gp.gp import GaussianProcess
from ML.dir_mul.dirichlet_multinomial import DirichletMultinomialRegression
import pdb

def normalise(data):
    return data/data.sum(axis=1)[:,np.newaxis]

def dm_vs_gp():
    X_train_c, X_test_c, X_train, X_test, C_train, C_test = generate_dm_toy_ex()   

    # Dirichlet multinomial stuff
    W = dirmultreg_learn(X_train, C_train, verbose=True, reg=0.1)
    EC, _ = dirmultreg_predict(X_test, W)

    C_test_norm = normalise(C_test)
    mean_err_dm1 = np.average(np.abs(EC-C_test_norm))
    print(mean_err_dm1)

    dm = DirichletMultinomialRegression()
    dm.fit(X_train, C_train)
    preds = dm.predict(X_test)
    mean_err_dm2 = np.average(np.abs(preds-C_test_norm))
    print(mean_err_dm2)

    # # GP Stuff
    # gp = GaussianProcess()
    # gp.fit(X_train, C_train.argmax(axis=1))
    # gp_probs, gp_vars1 = gp.predict(C_train.argmax(axis=1), keep_probs=True)
    # gp_preds1 = gp_probs.argmax(axis=0)
    # mean_err_gp1 = np.average(np.abs(gp_preds1 - C_test.argmax(axis=1)))
    # print(mean_err_gp1)

    # GPy
    # K = GPy.kern.Matern32(1)
    # m = GPy.models.GPRegression(X_train, C_train.argmax(axis=1)[:,np.newaxis], kernel=K.copy())
    gpy = GPyC()
    gpy.fit(X_train, C_train.argmax(axis=1)[:,np.newaxis])
    gp_preds2, gp_vars2 = m.predict(X_test)
    mean_err_gp2 = np.average(np.abs(gp_preds2 - C_test.argmax(axis=1)))
    print(mean_err_gp2)

    pdb.set_trace()