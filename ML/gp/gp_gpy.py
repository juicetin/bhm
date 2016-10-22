import GPy
import numpy as np
import multiprocessing as mp
from multiprocessing import Pool
from ML.helpers import partition_indexes

class GPyC:
    def __init__(self):
        pass

    def fit_label_model(self, c, X, C, K): 
        labels = np.array([1 if c == label else 0 for label in C])[:,np.newaxis]
        m = GPy.models.GPRegression(X, labels, kernel=K.copy())
        m.optimize()
        print('Finished optimising label {}'.format(c))
        return m

    def fit(self, X, C, parallel=False):
        """
        Fit the GPy-wrapper classifier 
        """
        K = GPy.kern.Matern32(X.shape[1])
        uniq_C = np.unique(C)
        self.models = []
        if parallel==True:
            args = [ c, X, C, K for c in uniq_C]
            pool = Pool(processes=uniq_C.shape[0])
            print("Distributing GP per-class model fitting across {} processes...".format(uniq_C.shape[0]))
            self.models = pool.starmap(self.fit_label_model, args)
        else:
            for c in uniq_C:
                labels = np.array([1 if c == label else 0 for label in C])[:,np.newaxis]
                print('Optimising for label {}...'.format(c))
                m = GPy.models.GPRegression(X, labels, kernel=K.copy())
                m.optimize()
                self.models.append(m)
        self.models = np.array(self.models)
        return self

    def predict(self, x, parallel=False):
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
                    cur_preds = self.predict(x[start:end])
                    all_preds[start:end] = cur_preds[0]
                    all_vars[start:end] = cur_preds[1]
            else:
                gp_preds, gp_vars = m.predict(x)
                all_preds[:,i] = gp_preds.flatten()
                all_vars[:,i]  = gp_vars.flatten()

        # The transpose here is to match the output of the Dirichlet Multinomial stuff
        return all_preds, all_vars

    def predict_parallel(self, x):
        """
        Does predictions in parallel
        """
        # Set up the parallel jobs on separate processes, to overcome 
        # Python's GIL for proper parallelisation
        nprocs = mp.cpu_count() - 1
        jobs = partition_indexes(x.shape[0], nprocs)
        args = [(x[start:end], False) for start, end in jobs]
        pool = Pool(processes=nprocs)
        print("Distributing predictions across {} processes...".format(nprocs))
        predict_results = pool.starmap(self.predict, args)

        # Concat along class list axis
        return np.concatenate(predict_results, axis=0)
