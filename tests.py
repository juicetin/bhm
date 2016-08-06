from ML.gp import GaussianProcess
from datetime import datetime
import numpy as np
import unittest
from numpy import testing as np_test
from scipy.spatial.distance import cdist
from sympy.utilities.autowrap import autowrap
import sympy as sp
import math

class TestGPMethods(unittest.TestCase):

    def test_multi_length_scale(self):
        print("\nTesting if pre-processing of arrays to account for multiple length scales before using cdist is equivalent to the 'dumb' method\n")
        gp = GaussianProcess()
        vec = np.random.rand(7,7)
        l_scales = np.random.rand(7)

        # Using library
        SE_term_lib = cdist(vec/l_scales, vec/l_scales, 'sqeuclidean')

        # Using custom function (to account for arbitrary objects)
        SE_term = gp.sqeucl_dist(vec/l_scales, vec/l_scales)
        # SE_term = gp.se_term(vec, vec, l_scales)

        np_test.assert_almost_equal(SE_term_lib, SE_term)

    def test_dK_dl_equivalence(self):
        print("\nTesting if 'fast' creation of dK_dls is equivalent to using sympy matrix differentiation\n")

        # Setup
        gp = GaussianProcess()
        dims = 5
        size = 10
        vec = np.array(10*np.random.rand(size, dims), dtype='int64')
        gp.X = vec

        # Symbols
        f_err_sym = sp.Symbol('f_err')
        l_scales_sym = sp.MatrixSymbol('l', 1, dims)
        n_err_sym = sp.Symbol('n_err')

        # Actuals
        f_err, n_err = np.array([1, 1])
        l_scales = np.array([1] * dims)

        # Build full K matrix
        scaled_data = vec/l_scales_sym
        m = (f_err_sym**2 * math.e ** (-0.5 * gp.sqeucl_dist(vec/l_scales_sym, vec/l_scales_sym)) + 
            n_err_sym**2 * np.identity(vec.shape[0]))
        m = sp.Matrix(m)

        # Using sympy differentiations
        print("Sympy differentiations and evaluation...")
        # dK_dls_using_lib = [m.diff(l_scale_sym) for l_scale_sym in l_scales_sym]
        dK_dl_0_sp = m.diff(l_scales_sym[0])
        eval_lib = np.array(sp.lambdify([f_err_sym, l_scales_sym], dK_dl_0_sp)(f_err, sp.Matrix([l_scales])), dtype='float64')

        print("Closed-form solution differential/evaluation...")
        m_ = np.exp(-0.5 * gp.dist(vec, vec, l_scales))
        eval_cf = gp.dK_dls_eval(m_, f_err, l_scales)[0]

        # Check float equalities
        self.assertEqual(True, np.allclose(eval_lib, eval_cf))

    def test_index_splitting(self):
        print("\nTesting that indexes are split up properly based on cores/length")

        gp = GaussianProcess()
        idxs = gp.parallelism_indexes(100, 3)
        self.assertEqual([(0,33), (33,66), (66,100)], idxs)

        idxs = gp.parallelism_indexes(5, 3)
        self.assertEqual([(0,1), (1,2), (2,5)], idxs)

if __name__ == '__main__':
    unittest.main()
