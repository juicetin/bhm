import numpy as np
from scipy.spatial.distance import cdist
from datetime import datetime
from ML.gp import GaussianProcess

a = np.array(10 * np.random.rand(5000, 5), dtype='int64')

d1 = datetime.now()
m_1 = cdist(a, a, 'sqeuclidean')
d2 = datetime.now()
t1 = d2-d1
print(t1)

m_2 = np.sum(np.power([(i-j) for i in a for j in a], 2), axis=1).reshape(a.shape[0], a.shape[0])
d3 = datetime.now()
t2 = d3 - d2
print(t2)

gp = GaussianProcess()
d4 = datetime.now()
m_3 = gp.sqeucl_dist(a, a)
d5 = datetime.now()
t3 = (d5-d4)
print(t3)

print("cdist was: {} times faster than python list comp".format(t2/t1))
print("cdist was: {} times faster than my sqeucl_dist".format(t3/t1))
print("my sqeucl_dist was: {} times faster than python list comp".format(t2/t3))

eq = np.sum(m_1 == m_2)
print(eq)
