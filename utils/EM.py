import numpy as np
import math

def EM(func=None, x0=None, tol=1e-5):
    prev = 100
    diff = 1
    while diff > tol:
        result = func(x0)
        diff = math.abs(result-prev)
