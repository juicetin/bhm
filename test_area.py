import theano
import numpy as np

# Sample data
data = np.array(10*np.random.rand(5, 3), dtype='int64')

# Not including data as tensor, incorrect/invalid indexing of symbolic vector
f_err_sym = theano.tensor.dscalar('f_err')
l_scales_sym = theano.tensor.dvector('l_scales')
# x = theano.tensor.dmatrix('x')
# f = f_err_sym**2 * x/l_scales_sym
#f_eval = theano.function([f_err_sym, x, l_scales_sym], f)

f = f_err_sym**2 * data/l_scales_sym
f_eval = theano.function([f_err_sym, l_scales_sym], f)

df_dl = theano.gradient.jacobian(f.flatten(), l_scales_sym)
df_dl_eval = theano.function([f_err_sym, l_scales_sym], df_dl)

state = theano.shared(np.ones(data.shape[1]))

# Attempt to use list of vectors instead to allow indexing them for partial derivatives
# x = theano.tensor.dmatrix('x')
# l_symbols = [theano.tensor.dvector('l' + str(i)) for i in range(data.shape[1])]
# f = x/2
# f_eval = theano.function([x], f)
# f = x/l_symbols
# f_eval = theano.function([x] + l_symbols, f)
# df_dl0 = theano.gradient.jacobian(f.flatten(), l_symbols[0])
# df_dl0_eval = theano.function([x, l_symbols[0], l_symbols[1], l_symbols[2]], df_dl0)

