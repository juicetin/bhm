import numpy as np
import pdb
from utils import visualisation as vis

def find_even_split_areas(q_preds, bounds=[0.48, 0.52], split_labels=[1,2]):
    """
    Finds areas which contain certain mixes of labels with a minimum threshold of equality
    """
    even_splits = q_preds[(q_preds[:,split_labels[0]] > bounds[0]) & (q_preds[:,split_labels[0]] < bounds[1]) & \
            (q_preds[:,split_labels[1]] > bounds[0]) & (q_preds[:,split_labels[1]] < bounds[1])]
    return even_splits

def plot_dm_per_label_maps(q_locations, q_preds):
    """
    Plots heatmap for each label in data
    """
    for i in range(q_preds.shape[1]):
        vis.show_map(q_locations, q_preds[:,i], display=False, filename='dm_simplelabel_heatmap_'+str(i))
