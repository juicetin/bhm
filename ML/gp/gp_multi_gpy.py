import GPy
import numpy as np
import multiprocessing as mp
from ML.helpers import partition_indexes
import multiprocessing as mp
from multiprocessing import Pool
from progressbar import ProgressBar
import pdb

class GPyMultiOutput:
    """
    Performs GP regression separately over each axis of the input data to
    act as a fake 'multi-output' Gaussian Process regressor
    """
    def __init__(self):
        pass

    def fit_axis(self, i, X, C, K): 
        m = GPy.models.GPRegression(X, C[:,i][:,np.newaxis], kernel=K.copy())
        m.optimize()
        print('Finished optimising label {}'.format(i))
        return m

    def fit(self, X, C, parallel=False):
        """ 
        Fit the GPy-wrapper classifier 
        """
        var = np.random.rand()
        l_scales = np.random.rand(X.shape[1])
        print(var, l_scales)
        K = GPy.kern.RBF(input_dim=X.shape[1], variance=var, lengthscale=l_scales, ARD=True)
        self.output_count = C.shape[1]
        self.models = []
        if parallel == True:
            args = [ (i, X, C, K) for i in range(C.shape[1])]
            pool = Pool(processes=C.shape[1])
            print("Distributing GP multi-output model fitting across {} processes...".format(C.shape[1]))
            self.models = pool.starmap(self.fit_axis, args)
        else:
            bar = ProgressBar(maxval=C.shape[1])
            bar.start()
            for i in range(C.shape[1]):
                bar.update(i)
                m = GPy.models.GPRegression(X, C[:,i][:,np.newaxis], kernel=K.copy())
                m.optimize()
                self.models.append(m)
            bar.finish()
        self.models = np.array(self.models)
        return self

    def predict(self, x, parallel=False):
        """
        Make predictions using the GPy-wrapper classifier
        """

        if parallel == True:
            return self.predict_parallel(x)

        # all_preds = np.empty((self.models.shape[0], x.shape[0]))
        all_preds = np.empty((x.shape[0], self.models.shape[0]))
        all_vars = np.empty(all_preds.shape)
        bar = ProgressBar(maxval=x.shape[0])
        bar.start()
        for i, m in enumerate(self.models):
            if x.shape[0] > 5000:
                step = 5000
                # Break into blocks of 5000
                for start in range(0, x.shape[0], step):
                    bar.update(start)
                    next_idx = start + 5000
                    end = next_idx if next_idx <= x.shape[0] else x.shape[0]
                    cur_preds = self.predict(x[start:end])
                    all_preds[start:end] = cur_preds[0]
                    all_vars[start:end] = cur_preds[1]
                bar.finish()
            else:
                gp_preds, gp_vars = m.predict(x)
                all_preds[:,i] = gp_preds.flatten().astype(np.float64)
                all_vars[:,i]  = gp_vars.flatten().astype(np.float64)

        # The transpose here is to match the output of the Dirichlet Multinomial stuff
        return np.array((all_preds, all_vars))

    def predict_parallel(self, x):
        """
        Does predictions in parallel
        """
        # Set up the parallel jobs on separate processes, to overcome 
        # Python's GIL for proper parallelisation
        nprocs = mp.cpu_count() - 1
        jobs = partition_indexes(x.shape[0], nprocs)
        args = [ (x[start:end], False) for start, end in jobs]
        pool = Pool(processes=nprocs)
        print("Distributing predictions across {} processes...".format(nprocs))
        predict_results = pool.starmap(self.predict, args)
        return np.hstack(predict_results)

# HACKY - for use when models are saved to remove need for retraining
def predict(x, models_shape=None, parallel=False, models=None, index_range=None, npy_name=None):
    """
    Make predictions using the GPy-wrapper classifier
    """
    if models != None:
        models_shape = len(models)
    elif models_shape == None:
        raise NameError('models_shape needs to be provided if models aren\'t!')

    if parallel == True:
        return predict_parallel(x, models_shape)

    # if index_range == None or npy_name == None:
    #     raise NameError('index_range and npy_name must be given for actual predictions!')

    # Load memory-mapped data into this process in READ-ONLY mode within the given indices
    # x = np.load(npy_name, mmap_mode='r')[index_range[0]:index_range[1]]

    all_preds = np.empty((x.shape[0], models_shape))
    all_vars = np.empty(all_preds.shape)
    for i, m in enumerate(models):
        if x.shape[0] > 5000:
            step = 5000
            # Break into blocks of 5000
            bar = ProgressBar(maxval=x.shape[0])
            bar.start()
            for start in range(0, x.shape[0], step):
                bar.update(start)
                next_idx = start + 5000
                end = next_idx if next_idx <= x.shape[0] else x.shape[0]
                cur_preds = predict(x[start:end], None, False, models)
                all_preds[start:end] = cur_preds[0]
                all_vars[start:end] = cur_preds[1]
            bar.finish()
        else:
            gp_preds, gp_vars = m.predict(x)
            all_preds[:,i] = gp_preds.flatten().astype(np.float64)
            all_vars[:,i]  = gp_vars.flatten().astype(np.float64)

    # The transpose here is to match the output of the Dirichlet Multinomial stuff
    return np.array((all_preds, all_vars))

def predict_parallel(x, models_shape):
    """
    Does predictions in parallel
    """
    if models_shape == None:
        raise NameError('models_shape needs to be provided')

    # Set up the parallel jobs on separate processes, to overcome
    # Python's GIL for proper parallelisation
    nprocs = mp.cpu_count() - 1
    jobs = partition_indexes(x.shape[0], nprocs)
    args = [(None, models_shape, False, None, [start, end], 'data/qp_red_features.npy') for start, end in jobs]
    pool = Pool(processes=nprocs)
    print("Distributing predictions across {} processes...".format(nprocs))
    predict_results = pool.starmap(predict, args)

    # Concat along class list axis
    # return np.concatenate(predict_results, axis=0)

    return np.hstack(predict_results)
