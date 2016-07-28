from ML.gp import GaussianProcess
import numpy as np
import unittest
from numpy import testing as np_test
from scipy.spatial.distance import cdist

class TestGPMethods(unittest.TestCase):

    def test_multi_length_scale(self):
        print("Testing if pre-processing of arrays to account for multiple length scales before using cdist is equivalent to the 'dumb' method")
        gp = GaussianProcess()
        vec = np.random.rand(7,7)
        l_scales = np.random.rand(7)
        SE_term_manual = (np.sum(np.array([((i-j)/l_scales)**2 for i in vec for j in vec ]), axis=1)
            .reshape(len(vec), len(vec)))

        SE_term = gp.se_term(vec, vec, l_scales)

        np_test.assert_almost_equal(SE_term, SE_term_manual)

if __name__ == '__main__':
    unittest.main()
