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


    # hclusters = hierarchy.linkage(coords, 'ward')
    code.interact(local=locals())

    return reduced_coords, reduced_features, reduced_mlabels

def find_child_nodes_in_dendrogram(dendrogram, dend_idx, orig_size):
    """
    Given a dendorgram object, a dendrogram index, and the size of the original
    input data, this function finds all the child nodes of a particular node.
    Generalised so that the child node of an original node will simply be itself
    (as opposed to what the strict definition would give - null).
    """

    # Once we've reached the original set of nodes, return this in an array
    if dend_idx < orig_size:
        return [dend_idx]

    dend_row = dendrogram[dend_idx-orig_size]
    return find_child_nodes_in_dendrogram(dendrogram, dend_row[0], orig_size) + \
        find_child_nodes_in_dendrogram(dendrogram, dend_row[1], orig_size)

def assign_points_in_cluster(num_points, cluster_idx, dendrogram, dend_entry, cluster_assignments):
    """
    For a given dendrogram entry, find all nodes within cluster and assign them their cluster.
    Exit function if any entry encountered with nodes that are already assigned clusters.
    """
    # Create list to hold indexes in cluster (?)

    # Iteratively iterate into each index until size of cluster is 2 (can't do recursive as we have
    #   to keep track of number of points added - recursive branches won't know how many nodes 
    #   the other branch has)

    # If cluster size is hit, start a new cluster

    # return number of nodes added to clusters this round!

    idx_stack = [dend_entry[1], dend_entry[0]]
    while len(stack) > 0:
        top_idx = idx_stack.pop()

        if top_idx < num_points:
            # Index is that of one of the original points
            cluster_assignments[top_idx] = cluster_idx
        else:
            # Index is one of the dendrogram clusters
            dend_idx = dendrogram[top_idx - num_points]

def downsample_limited_nearest_points(coords, cluster_idx, dendrogram, clust_dist=21, clust_size=5):
    """
    Uses hierarchical clustering to group points into clusters based on criteria:
        1. Cannot exceed more than a certain total count of label counts
        2. Cannot exceed more than a certain number of points per aggregation cluster
        3. Points grouped together in cluster must not exceed distance X
    """
    clusters = {}
    cluster_assignments = np.full(-1, coords.shape[0])
    cluster_idx = 0
    points_assigned_count = 0

    # Iterate over all dendrogram entries to create desired clusters
    for dend_entry in reversed(dendrogram):

        # Check if necessary conditions to form a cluster are met
        if cluster_cond_check(dend_entry, clust_dist, clust_size):
            points_assigned_count += assign_points_in_cluster(coords.shape[0], cluster_idx, dendrogram, dend_entry, cluster_assignments)
            cluster_idx += 1

            # Stop going through dendrogram once all points have been assigned
            if points_assigned_count == coords:
                break

def cluster_cond_check(dend_row, clust_dist, clust_size):
    """
    Function for checking whether conditions for a dendrogram row can be taken as a cluster
    """
    return dend_row[2] <= clust_dist && dend_row[3] <= clust_size

def downsample_spatial_data(coords, data, label_counts):
    """
    Takes input *data* (any number of dimensions) and *label_counts*, where each row
    corresponds to an array of 'bincounts' that can be added to each other that would
    correspond to aggregation of points
    """
    return downsample_by_fixed_grid(coords, data, label_counts, reduction_factor=1.1)
