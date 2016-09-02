import code
import math
import numpy as np
from scipy.cluster import hierarchy
import pdb

def label_stats(multi_labels):
    non_0s = np.apply_along_axis(lambda x: np.sum(x!=0), 1, multi_labels)
    total_counts = np.apply_along_axis(np.sum, 1, multi_labels)

    return np.bincount(non_0s)[1:], total_counts

def round_down(num, divisor, base=0):
    """
    Given a number 'num', 'divisor', and starting number 'base',
    find the largest number less than 'num such that base + some
    multiple of 'divisor' equals the number
    """
    diff = num-base
    return base + (diff - (diff % divisor))

def find_nearest_grid(x_coord, x_step, x_min, y_coord, y_step, y_min):
    """
    Given an x and y coord, finds the nearest decreased resolution gridbox
    that would contain this coordinate (points belong to the grid with 
    relative origin in the upper left coord of the low-res system). This is
    done by simply rounding the x_coord, y_coord to the next multiple of
    x_step, y_step **down**, respectively.
    """
    return round_down(x_coord, x_step, x_min), round_down(y_coord, y_step, y_min)

def grid_dist_metric(grid_key, x_point, y_point):
    """
    Returns the distance between an (x,y) coord and the grid-coordinate (top left corner) 
    of the grid it belongs to
    """
    return np.sqrt((x_point-cur_grid_key[0])**2 + (y_point-cur_grid_key[1])**2)

def downsample_by_fixed_grid(coords, data, label_counts, reduction_factor=2):
    """
    Downsamples by allocating each point into the evenly distributed fixed side grids
    'overlaid' over the original 2-dimensional space
    """

    # Get extremes of the coordinate system
    x_min = coords[:,0].min()
    x_max = coords[:,0].max()
    y_min = coords[:,1].min()
    y_max = coords[:,1].max()

    # Decide on number of points in reduced low-res space
    reduced_point_count = int(coords.shape[0]/reduction_factor) # default red_factor - 4

    # Calculate number of x/y blocks
    # Assumes x is longer than y!
    xy_ratio = (y_max-y_min)/(x_max-x_min)
    x_block_cnt = np.sqrt(reduced_point_count/xy_ratio)
    y_block_cnt = reduced_point_count/x_block_cnt

    # Build coordinates in low-res space
    # TODO x_step and y_step as long floats too troublesome - round up to nearest integer
    x_step = 21 # math.ceil((x_max-x_min)/x_block_cnt)
    reduced_x_coords = np.arange(x_min, x_max+x_step, x_step)
    y_step = 21 # math.ceil((y_max-y_min)/y_block_cnt)
    reduced_y_coords = np.arange(y_min, y_max+y_step, y_step)
    x_mesh, y_mesh = np.meshgrid(reduced_x_coords, reduced_y_coords)

    orig_stats = label_stats(label_counts)

    # Creates all bins with fixed overlaid low-res grid size
    # At the moment - takes the first seen coordinate in a grid to be the definitive one
    coord_bins = {}
    for (x_point, y_point), features, labels in zip(coords, data, label_counts):

        cur_grid_key = find_nearest_grid(x_point, x_step, x_min, y_point, y_step, y_min)
        grid_dist_cmp = np.sqrt((x_point-cur_grid_key[0])**2 + (y_point-cur_grid_key[1])**2)
        
        # Create bin if doesn't exist yet
        if cur_grid_key not in coord_bins:
            coord_bins[cur_grid_key] = [features, labels, grid_dist_cmp, 1]
        # Otherwise aggregate labels/update features if necessary
        else:
            # TODO take closest coordinate to centre of low-res grid instead of first seen
            _, _, prev_smallest_grid_dist_cmp, _ = coord_bins[cur_grid_key]
            coord_bins[cur_grid_key][1] += labels
            coord_bins[cur_grid_key][3] += 1

    # Extract reduced features and multi-labels
    reduced_coords = np.array(list(coord_bins.keys()))
    reduced_features_and_labels= np.array(list(coord_bins.values()))
    grid_bins = reduced_features_and_labels[:,3].astype(int)
    reduced_features = np.concatenate(reduced_features_and_labels[:,0]).reshape(reduced_coords.shape[0], data.shape[1])
    reduced_mlabels = np.concatenate(reduced_features_and_labels[:,1]).reshape(reduced_coords.shape[0], label_counts.shape[1])

    reduced_stats = label_stats(reduced_mlabels)

    print("=============== SUMMARY STATS ===============")
    print("Min points contained in one bin: {}\nMax points contained in one bin: {}\nAverage points contained in grid bins: {}\n"\
            .format(np.min(grid_bins), np.max(grid_bins), np.average(grid_bins)))
    print("=============================================")


    hclusters = hierarchy.linkage(coords, 'ward')
    code.interact(local=locals())

    return reduced_coords, reduced_features, reduced_mlabels

def downsample_limited_nearest_points():
    """
    Uses hierarchical clustering to group points into clusters based on criteria:
        1. Cannot exceed more than a certain total count of label counts
        2. Cannot exceed more than a certain number of points per aggregation cluster
        3. Points grouped together in cluster must not exceed distance X
    """

def downsample_spatial_data(coords, data, label_counts):
    """
    Takes input *data* (any number of dimensions) and *label_counts*, where each row
    corresponds to an array of 'bincounts' that can be added to each other that would
    correspond to aggregation of points
    """
    return downsample_by_fixed_grid(coords, data, label_counts, reduction_factor=1.1)
