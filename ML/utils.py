import numpy as np
from scipy.special import gamma as sci_gamma
from scipy.special import psi as sci_psi

def gamma(m):
    return sci_gamma(m)

# Psi/digamma
def digamma(m):
    return sci_psi(m)

