from ML.gp import GaussianProcess
import numpy as np
import unittest
from numpy import testing as np_test
from scipy.spatial.distance import cdist

class TestGPMethods(unittest.TestCase):

    def test_multi_length_scale(self):
        print("Testing if pre-processing of arrays to account for multiple length scales before using cdist is equivalent to the 'dumb' method")
        gp = GaussianProcess()
        vec = np.array([[0, 15.5913078497458, 18.0544980252505, 16.8986782917844, 21.6712810146222, 33.3617753898887, 13.1073433181501],
                [15.5913078497458, 0, 20.6720247383852, 6.06247432740044, 14.0649698036318, 14.8501523494137, 12.6308468842494],
                [18.0544980252505, 20.6720247383852, 0, 20.5218066743874, 17.8117131464945, 19.9032794483337, 4.47698629113567],
                [16.8986782917844, 6.06247432740044, 20.5218066743874, 0, 25.5523150658152, 27.9094542449488, 14.6987658251752],
                [21.6712810146222, 14.0649698036318, 17.8117131464945, 25.5523150658152, 0, 5.58486180779463, 6.77865103315562],
                [33.3617753898887, 14.8501523494137, 19.9032794483337, 27.9094542449488, 5.58486180779463, 0, 9.40546301932786],
                [13.1073433181501, 12.6308468842494, 4.47698629113567, 14.6987658251752, 6.77865103315562, 9.40546301932786, 0]])
        l_scales = [2,5,3,8,9,1,4]
        SE_term_manual = (np.sum(np.array([((i-j)/l_scales)**2 for i in vec for j in vec ]), axis=1)
            .reshape(len(vec), len(vec)))

        SE_term = gp.se_term_length_scale_per_d(vec, vec, l_scales)

        np_test.assert_almost_equal(SE_term, SE_term_manual)

if __name__ == '__main__':
    unittest.main()
