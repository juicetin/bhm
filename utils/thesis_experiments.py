import numpy as np
import pdb
from utils import visualisation as vis
from ML.dir_mul.nicta.dirmultreg import dirmultreg_learn, dirmultreg_predict

def find_even_split_areas(q_preds, q_vars, bounds=[0.4, 0.6], split_labels=[1,2]):
    """
    Finds areas which contain certain mixes of labels with a minimum threshold of equality
    """
    even_split_idxs = np.where((q_preds[:,split_labels[0]] > bounds[0]) & (q_preds[:,split_labels[0]] < bounds[1]) & (q_preds[:,split_labels[1]] > bounds[0]) & (q_preds[:,split_labels[1]] < bounds[1]))
    even_splits_preds = q_preds[even_split_idxs]
    even_splits_vars = q_vars[even_split_idxs]
    
    avg_var = np.average(even_splits_vars)
    print('Average variance in condition-satisfied splits was: {}'.format(avg_var))
    print('There were {} matches'.format(even_splits_preds.shape[0]))
    return even_splits_preds, even_splits_vars

def plot_dm_per_label_maps(q_locations, q_preds):
    """
    Plots heatmap for each label in data
    """
    for i in range(q_preds.shape[1]):
        vis.show_map(q_locations, q_preds[:,i], display=False, filename='dm_simplelabel_heatmap_'+str(i))

def check_overlaps(f, l, q, reg=100):
    W = dirmultreg_learn(f, l, verbose=True, reg=reg)
    query_preds = dirmultreg_predict(q, W)
    return find_even_split_areas(query_preds)

def f_max_var_rank(f, v):
    """
    Returns the n-rank of the largest matching labels' variance
    """
    f_max, f_max_idx = f.max(), f.argmax()
    f_max_var = v[f_max_idx]
    rank = np.argwhere(v == f_max_var)[0][0]
    return rank

def plot_training_data_per_label(locations, labels, gen_images=False):
    """
    Plots all points per label on separate maps to visualise where they are/be able to semi-manually find images of a certain label
    """
    uniq_labels = np.unique(labels)
    file_base = 'training_points_'+str(uniq_labels.shape[0])+'labels_'
    label_coord_map = {}
    for c in uniq_labels:
        cur_label_idx = np.where(labels == c)
        cur_label_locations = locations[cur_label_idx]
        cur_labels = labels[cur_label_idx]
        if gen_images == True:
            vis.show_map(cur_label_locations, cur_labels, display=False, filename=file_base+str(c))
        label_coord_map[c] = cur_label_locations

    return label_coord_map

def dm_vs_gp_even_split_vars(dm_results, gp_results):
    """
    Compares the predictions/variance of a DM vs GP in areas where they believe there are even splits
    """
    dm_preds, dm_vars = dm_results
    gp_preds, gp_vars = gp_results

    # Check vars in GP compared to DM for even splits
    dm_evensplit_preds, dm_evensplit_vars = find_even_split_areas(dm_preds, dm_vars)
    gp_evensplit_preds, gp_evensplit_vars = find_even_split_areas(gp_preds, gp_vars)
