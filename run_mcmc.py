import numpy as np
from utils.thesis_experiments import multi_dm_mcmc_chains_continue
from utils.thesis_experiments import multi_dm_mcmc_chains
from ML.dir_mul.dm_mcmc import continue_mcmc
from utils.data_transform import features_squared_only
from utils.load_data import load_reduced_data

def start():
    red_features, red_mlabels4, red_mlabels24, _, _, \
        _, _, _ = load_reduced_data()

    f_sq2 = features_squared_only(red_features)
    # l4 = red_mlabels4
    # l4_norm = l4/l4.sum(axis=1)[:,np.newaxis]
    # multi_dm_mcmc_chains_continue(f_sq2, l4_norm, 5000000)

    l24 = red_mlabels24
    l24_norm = l24/l24.sum(axis=1)[:,np.newaxis]
    # multi_dm_mcmc_chains(f_sq2, l24_norm, 100000)
    multi_dm_mcmc_chains_continue(f_sq2, l24_norm, 350000) #950,000 (1.3m after) total atm

if __name__ == "__main__":
    print('Continuing mcmc from previous runs...')
    start()
