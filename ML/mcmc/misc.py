import numpy as np

def pow2(x):
        return x*x

def imputation_r_square_hat(posteriori):
    x_bar_bar = (1/n) * np.average(x, axis=1).sum()
    P = x_bar_bar/2

    r_square_hat = ((1/n) * (x_bar - x_bar_bar).sum()**2)/(2*P*(1-P))
    return r_square_hat

	# """
	# >>> print r_square_hat([(0.3333333, 0.3333333, 0.3333333), (1.0, 0.0, 0.0)])
	# 0.428571367347
	# """

	# n=float( len(posteriori))
	# meanAB = sum([x[1] for x in posteriori])
	# meanBB = sum([x[2] for x in posteriori])

	# sumX2 = sum([x[1] + (4.0 *x[2]) for x in posteriori])
	# sumXbar2 = sum([(x[1] + (2.0 * x[2]))**2.0 for x in posteriori])
	# meanX = (meanAB+2.*meanBB)/n
	# rSqHat = (sumXbar2/n-pow2(meanX))/(sumX2/n-pow2(meanX))

	# return rSqHat



#Method name =imputation_r_square_hat()
if __name__ == '__main__':
    posteriori = [(0.3333333, 0.3333333, 0.3333333), (1.0, 0.0, 0.0)]

    returned = imputation_r_square_hat(posteriori=posteriori)
    if returned:
        print ('Method returned:')
        print (returned)

