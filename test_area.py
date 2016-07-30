import theano
import sympy
from sympy.utilities.autowrap import autowrap
from sympy.utilities.autowrap import binary_function
import numpy as np
import math
from datetime import datetime

# 'Vectorized' cdist that can handle symbols/arbitrary types
def sqeucl_dist(x, xs):
    m = np.sum(np.power(
        np.repeat(x[:,None,:], len(xs), axis=1) -
        np.resize(xs, (len(x), xs.shape[0], xs.shape[1])),
        2), axis=2)
    return m


def build_symbolic_derivatives(X):
    # Pre-calculate derivatives of inverted matrix to substitute values in the Squared Exponential NLL gradient
    f_err_sym, n_err_sym = sympy.symbols("f_err, n_err")

    # (1,n) shape 'matrix' (vector) of length scales for each dimension
    l_scale_sym = sympy.MatrixSymbol('l', 1, X.shape[1])

    # K matrix
    print("Building sympy matrix...")
    eucl_dist_m = sqeucl_dist(X/l_scale_sym, X/l_scale_sym)
    m = sympy.Matrix(f_err_sym**2 * math.e**(-0.5 * eucl_dist_m) 
                     + n_err_sym**2 * np.identity(len(X)))


    # Element-wise derivative of K matrix over each of the hyperparameters
    print("Getting partial derivatives over all hyperparameters...")
    pd_t1 = datetime.now()
    dK_df   = m.diff(f_err_sym)
    dK_dls  = [m.diff(l_scale_sym) for l_scale_sym in l_scale_sym]
    dK_dn   = m.diff(n_err_sym)
    print("Took: {}".format(datetime.now() - pd_t1))

    # Lambdify each of the dK/dts to speed up substitutions per optimization iteration
    print("Lambdifying ")
    l_t1 = datetime.now()
    dK_dthetas = [dK_df] + dK_dls + [dK_dn]
    dK_dthetas = sympy.lambdify((f_err_sym, l_scale_sym, n_err_sym), dK_dthetas, 'numpy')
    print("Took: {}".format(datetime.now() - l_t1))
    return dK_dthetas


# Evaluates each dK_dtheta pre-calculated symbolic lambda with current iteration's hyperparameters
# def eval_dK_dthetas(dK_dthetas_raw, f_err, l_scales, n_err):
#     l_scales = sympy.Matrix(l_scales.reshape(1, len(l_scales)))
#     return np.array(dK_dthetas_raw(f_err, l_scales, n_err), dtype=np.float64)
# 
# 
# dimensions = 3
# X = np.random.rand(50, dimensions)
# dK_dthetas_raw = build_symbolic_derivatives(X)
# 
# f_err = np.random.rand()
# l_scales = np.random.rand(3)
# n_err = np.random.rand()
# 
# t1 = datetime.now()
# dK_dthetas = eval_dK_dthetas(dK_dthetas_raw, f_err, l_scales, n_err) # ~99.7%
# print(datetime.now() - t1)

# Sample data
size = 10
print("Data size: {}".format(size))
data = np.array(np.array(10*np.random.rand(size, size), dtype='int64'), dtype='float64')
l_scales = np.random.rand(size)
f_err = np.random.rand()
n_err = np.random.rand()

######################### Sympy #########################
# Symbols
f_err_sym = sympy.Symbol('f_err')
data_sym = sympy.MatrixSymbol('m', data.shape[0], data.shape[1])
l_scales_sym = sympy.MatrixSymbol('l', 1, data.shape[1])
n_err_sym = sympy.Symbol("n_err")
l_scales = l_scales.reshape(1, len(l_scales))

m = sympy.Matrix(f_err_sym**2 * data/l_scales_sym) # + n_err_sym**2 * np.identity(data.shape[0]))

# t_t1 = datetime.now()
# dm_df = m.diff(f_err_sym)
# dm_df_eval = sympy.lambdify([f_err_sym, l_scales_sym], dm_df)
# dm_df_eval = autowrap(dm_df, backend='cython')
# dm_df_eval = binary_function('dm_df', dm_df)
# t_t2 = datetime.now()
# print("Sympy creating deriv function: {}".format(t_t2 - t_t1))

# t_t3 = datetime.now()
# dm_df_eval(f_err, l_scales)
# t_t4 = datetime.now()
# print("Sympy evaluating function: {}".format(t_t4 - t_t3))

m_1 = f_err_sym**2 * np.array(data_sym) / np.array(l_scales_sym) + n_err_sym**2 * np.identity(data.shape[0])
m_1 = sympy.Matrix(m_1)

#### Using diff for each respective dimension 
d_t1 = datetime.now()
dK_df = m_1.diff(f_err_sym)
dK_dls = [m_1.diff(l_scale_sym) for l_scale_sym in l_scales_sym]
dK_dn = m_1.diff(n_err_sym)
d_t2 = datetime.now()
print("Using diff for each dimension separately took: {}".format(d_t2 - d_t1))

#### Using jacobian all at once instead of individual diffs
j_t1 = datetime.now()
d_wrt = sympy.Matrix([f_err_sym] + list(l_scales_sym) + [n_err_sym])
m_1_flat = sympy.Matrix(sympy.flatten(m_1))
m_1_ds_flat = m_1_flat.jacobian(d_wrt)
j_t2 = datetime.now()
print("Using jacobian over all dimensions took: {}".format(j_t2 - j_t1))

# t_t1 = datetime.now()
# dm1_df = m_1.diff(f_err_sym)
# dm1_df_eval = autowrap(dm1_df, backend='cython')
# t_t2 = datetime.now()
# print("Sympy creating deriv function + autowrap: {}".format(t_t2 - t_t1))

# t_t3 = datetime.now()
# dm1_df_eval(f_err, l_scales, data)
# t_t4 = datetime.now()
# print("Cython Autowrap evaluating function: {}".format(t_t4 - t_t3))

######################### Theano #########################
# f_err_sym = theano.tensor.dscalar('f_err')
# l_scales_sym = theano.tensor.dvector('l_scales')
# 
# m = f_err_sym**2 * data/l_scales_sym
# m_eval = theano.function([f_err_sym, l_scales_sym], m)
# 
# s_t1 = datetime.now()
# dm_df = theano.gradient.jacobian(m.flatten(), f_err_sym)
# dm_df_eval = theano.function([f_err_sym, l_scales_sym], dm_df)
# s_t2 = datetime.now()
# print("Theano creating deriv function: {}".format(s_t2 - s_t1))
# 
# dm_df_eval(f_err, l_scales)
# s_t3 = datetime.now()
# print("Theano evaluating function: {}".format(s_t3 - s_t2))

# dm_dl = theano.gradient.jacobian(m.flatten(), l_scales_sym)
# dm_dl_eval = theano.function([f_err_sym, l_scales_sym], df_dm)


