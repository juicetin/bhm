import GPy
import numpy as np
import multiprocessing as mp
from multiprocessing import Pool
from ML.helpers import partition_indexes
from ML.helpers import sigmoid
from progressbar import ProgressBar
import pdb

class GPyC:
    def __init__(self):
        pass

    def fit_label_model(self, c, X, C, K, verbose=False): 
        labels = np.array([1 if c == label else 0 for label in C])[:,np.newaxis]
        m = GPy.models.GPRegression(X, labels, kernel=K.copy())
        m.optimize()
        if verbose == True:
            print('Finished optimising label {}'.format(c))
        return m

    def fit(self, X, C, parallel=True, optimize=True):
        """
        Fit the GPy-wrapper classifier 
        """
        var = np.random.rand()
        l_scales = np.random.rand(X.shape[1])
        K = GPy.kern.RBF(input_dim=X.shape[1], variance=var, lengthscale=l_scales, ARD=True)

        uniq_C = np.unique(C)
        # NOTE hacky fix for small datasets here!
        if uniq_C.shape[0] > 4:
            uniq_C = np.arange(24)

        self.models = []
        if parallel==True:
            args = [ (c, X, C, K) for c in uniq_C]
            pool = Pool(processes=uniq_C.shape[0])
            self.models = pool.starmap(self.fit_label_model, args)
            pool.close()
            pool.join()
        else:
            for c in uniq_C:
                labels = np.array([1 if c == label else 0 for label in C])[:,np.newaxis]
                m = GPy.models.GPRegression(X, labels, kernel=K.copy())
                if optimize==True:
                    m.optimize()
                self.models.append(m)
        self.models = np.array(self.models)
        return self

    def predict(self, x, parallel=True):
        """
        Make predictions using the GPy-wrapper classifier
        """

        if parallel == True:
            return self.predict_parallel(x)

        all_preds = np.empty((x.shape[0], self.models.shape[0]))
        all_vars = np.empty(all_preds.shape)
        for i, m in enumerate(self.models):
            if x.shape[0] > 5000:
                step = 5000
                # Break into blocks of 5000
                for start in range(0, x.shape[0], step):
                    next_idx = start + 5000
                    end = next_idx if next_idx <= x.shape[0] else x.shape[0]
                    cur_preds = self.predict(x[start:end], False)
                    all_preds[start:end] = cur_preds[0]
                    all_vars[start:end] = cur_preds[1]
            else:
                gp_preds, gp_vars = m.predict(x)
                all_preds[:,i] = gp_preds.flatten().astype(np.float64)
                all_vars[:,i]  = gp_vars.flatten().astype(np.float64)

        # The transpose here is to match the output of the Dirichlet Multinomial stuff
        return np.array((sigmoid(all_preds), all_vars))

    def predict_parallel(self, x):
        """
        Does predictions in parallel
        """
        # Set up the parallel jobs on separate processes, to overcome 
        # Python's GIL for proper parallelisation
        nprocs = mp.cpu_count() - 1
        # if nprocs > 4:
        #     nprocs = 4
        jobs = partition_indexes(x.shape[0], nprocs)
        args = [(x[start:end], False) for start, end in jobs]
        pool = Pool(processes=nprocs)
        predict_results = pool.starmap(self.predict, args)
        pool.close()
        pool.join()

        # Concat along class list axis
        # return np.concatenate(predict_results, axis=0)
        return np.hstack(predict_results)

    # k(x_*, x_*) is the prior variance - http://www.tsc.uc3m.es/~fernando/l1.pdf
    def prior_variance(self, x):

        params = np.array([model.param_array for model in self.models])
        # averaged_params = np.average(params, axis=0)
        f_errs = params[:,0][:,np.newaxis]
        n_errs = params[:,-1][:,np.newaxis]

        # [0] because the (x_p-x_q)**2 in the kern for x_p=x_1 = 0**2
        prior_var = f_errs**2 * np.exp([0] * x.shape[0]) + n_errs**2 * np.full(x.shape[0], 1)
        # prior_var = np.ones((f_errs.shape[0], x.shape[0]))
        return prior_var.T

# # HACKY - for use when models are saved to remove need for retraining
# def predict(x, models_shape=None, parallel=False, models=None, index_range=None, npy_name=None):
#     """
#     Make predictions using the GPy-wrapper classifier
#     """
#     if models != None:
#         models_shape = len(models)
#     elif models_shape == None:
#         raise NameError('models_shape needs to be provided if models aren\'t!')
# 
#     if parallel == True:
#         return predict_parallel(x, models_shape)
# 
#     # if index_range == None or npy_name == None:
#     #     raise NameError('index_range and npy_name must be given for actual predictions!')
# 
#     # Load memory-mapped data into this process in READ-ONLY mode within the given indices
#     # x = np.load(npy_name, mmap_mode='r')[index_range[0]:index_range[1]]
# 
#     all_preds = np.empty((x.shape[0], models_shape))
#     all_vars = np.empty(all_preds.shape)
#     for i, m in enumerate(models):
#         if x.shape[0] > 5000:
#             step = 5000
#             # Break into blocks of 5000
#             bar = ProgressBar(maxval=x.shape[0])
#             bar.start()
#             for start in range(0, x.shape[0], step):
#                 bar.update(start)
#                 next_idx = start + 5000
#                 end = next_idx if next_idx <= x.shape[0] else x.shape[0]
#                 cur_preds = predict(x[start:end], None, None, models)
#                 all_preds[start:end] = cur_preds[0]
#                 all_vars[start:end] = cur_preds[1]
#             bar.finish()
#         else:
#             gp_preds, gp_vars = m.predict(x)
#             all_preds[:,i] = gp_preds.flatten().astype(np.float64)
#             all_vars[:,i]  = gp_vars.flatten().astype(np.float64)
# 
#     # The transpose here is to match the output of the Dirichlet Multinomial stuff
#     return np.array((all_preds, all_vars))
# 
# def predict_parallel(x, models_shape):
#     """
#     Does predictions in parallel
#     """
#     if models_shape == None:
#         raise NameError('models_shape needs to be provided')
# 
#     # Set up the parallel jobs on separate processes, to overcome 
#     # Python's GIL for proper parallelisation
#     nprocs = mp.cpu_count() - 1
#     jobs = partition_indexes(x.shape[0], nprocs)
#     args = [(None, models_shape, False, None, [start, end], 'data/qp_red_features.npy') for start, end in jobs]
#     pool = Pool(processes=nprocs)
#     predict_results = pool.starmap(predict, args)
#     pool.close()
#     pool.join()
# 
#     # Concat along class list axis
#     # return np.concatenate(predict_results, axis=0)
# 
#     # return np.hstack(predict_results)
=======
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
                cur_preds = predict(x[start:end], None, None, models)
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
    predict_results = pool.starmap(predict, args)
    pool.close()
    pool.join()

    # Concat along class list axis
    # return np.concatenate(predict_results, axis=0)

    # return np.hstack(predict_results)

class GPR:
    def __init__(self):
        pass

    def fit(self, X, Y, parallel=True, K=None):
        """
        Fit the GPy-wrapper classifier 
        """
        var = np.random.rand()
        l_scales = np.random.rand(X.shape[1])
        if K == None:
            K = GPy.kern.RBF(input_dim=X.shape[1], variance=var, lengthscale=l_scales, ARD=True)
        m = GPy.models.GPRegression(X, Y, kernel=K.copy())
        m.optimize()
        self.model = m
        return self

    def predict(self, x, parallel=False):
        """
        Make predictions using the GPy-wrapper classifier
        """

        # if parallel == True:
        #     return self.predict_parallel(x)

        # all_preds = np.empty(x.shape[0])
        # all_vars = np.empty(all_preds.shape)
        # if x.shape[0] > 5000:
        #     step = 5000
        #     # Break into blocks of 5000
        #     for start in range(0, x.shape[0], step):
        #         next_idx = start + 5000
        #         end = next_idx if next_idx <= x.shape[0] else x.shape[0]
        #         cur_preds = self.predict(x[start:end])
        #         all_preds[start:end] = cur_preds[0]
        #         all_vars[start:end] = cur_preds[1]
        # else:
        #     gp_preds, gp_vars = self.model.predict(x)
        #     all_preds = gp_preds.flatten().astype(np.float64)
        #     all_vars = gp_vars.flatten().astype(np.float64)

        # The transpose here is to match the output of the Dirichlet Multinomial stuff
        gp_preds, gp_vars = self.model.predict(x)
        # all_preds = gp_preds.flatten().astype(np.float64)
        return np.array((gp_preds, gp_vars))

    # def predict_parallel(self, x):
    #     """
    #     Does predictions in parallel
    #     """
    #     # Set up the parallel jobs on separate processes, to overcome 
    #     # Python's GIL for proper parallelisation
    #     nprocs = mp.cpu_count() - 1
    #     # if nprocs > 4:
    #     #     nprocs = 4
    #     jobs = partition_indexes(x.shape[0], nprocs)
    #     args = [(x[start:end], False) for start, end in jobs]
    #     pool = Pool(processes=nprocs)
    #     predict_results = pool.starmap(self.predict, args)
    #     pool.close()
    #     pool.join()

    #     # Concat along class list axis
    #     # return np.concatenate(predict_results, axis=0)
    #     return np.hstack(predict_results)
