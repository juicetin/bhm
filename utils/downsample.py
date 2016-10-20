import code
import math
import numpy as np
from scipy.cluster import hierarchy
import pdb
label_map={1:0,2:0,3:1,4:3,5:1,6:3,7:3,8:3,9:3,10:1,11:3,12:3,13:2,14:2,15:2,16:1,17:1,18:0,19:1,20:0,21:0,22:1,23:0,24:0}

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

def fixed_grid_blocksize(coords, reduction_factor):
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

    return x_min, y_min, x_max, y_max, x_step, y_step, reduced_x_coords, reduced_y_coords

def downsample_counts(*, key, coord_bins, features, labels, grid_dist_cmp):
    """
    Helper function for downsample_by_fixed_grid to downsample by summing counts per overlay rid
    """
    if key not in coord_bins:
        # Create bin if doesn't exist yet
        coord_bins[key] = [features, labels, grid_dist_cmp, 1]     # 3rd element - keep track of number of original points in grid
    else:
        # Otherwise aggregate labels/update features if necessary
        # TODO take closest coordinate to centre of low-res grid instead of first seen
        _, _, prev_smallest_grid_dist_cmp, _ = coord_bins[key]
        coord_bins[key][1] += labels
        coord_bins[key][3] += 1            # 3rd element - keep track of number of original points in grid

def downsample_single_labels(*, key, coord_bins, features, labels, grid_dist_cmp):
    """
    Helper function for downsample_by_fixed_grid to downsample by taking the label of the first label seen in overlay grid
    """
    if key not in coord_bins:
        coord_bins[key] = [features, labels, grid_dist_cmp, 1]     # 3rd element - keep track of number of original points in grid
    else:
        _, cur_label, _, _ = coord_bins[key]
        rand = np.random.rand()
        # if cur_label == 3:
        if label_map[cur_label] == 3:
            if rand >= 0.05:
                coord_bins[key] = [features, labels, grid_dist_cmp, 1]
        else:
            if rand < 0.05:
                coord_bins[key] = [features, labels, grid_dist_cmp, 1]
        # Key already contains coordinates

    # Otherwise, do nothing, as we only take first label
    # Possible TODO - keep a list of labels and pick from them randomly at the end

def downsample_by_fixed_grid(coords, data, label_counts, reduction_factor=2):
    """
    Downsamples by allocating each point into the evenly distributed fixed side grids
    'overlaid' over the original 2-dimensional space
    """
    # Build coordinates in low-res space
    x_min, y_min, x_max, y_max, x_step, y_step, reduced_x_coords, reduced_y_coords = fixed_grid_blocksize(coords, reduction_factor)
    x_mesh, y_mesh = np.meshgrid(reduced_x_coords, reduced_y_coords)

    # orig_stats = label_stats(label_counts)

    # Decide which downsample helper to use
    if len(label_counts.shape) == 2:
        downsample_helper = downsample_counts
    elif len(label_counts.shape) == 1:
        downsample_helper = downsample_single_labels

    # Creates all bins with fixed overlaid low-res grid size
    # At the moment - takes the first seen coordinate in a grid to be the definitive one
    coord_bins = {}
    for (x_point, y_point), features, labels in zip(coords, data, label_counts):

        cur_grid_key = find_nearest_grid(x_point, x_step, x_min, y_point, y_step, y_min)
        grid_dist_cmp = np.sqrt((x_point-cur_grid_key[0])**2 + (y_point-cur_grid_key[1])**2) # Stores distance from cur grid's coords

        # Calls either downsample_counts or downsample_single_labels depending on labels passed in
        downsample_helper(key=cur_grid_key, coord_bins=coord_bins, features=features, labels=labels, grid_dist_cmp = grid_dist_cmp)

        # ########## Sums up label counts for each downsampled overlaid grid ##########
        # # Create bin if doesn't exist yet
        # if cur_grid_key not in coord_bins:
        #     coord_bins[cur_grid_key] = [features, labels, grid_dist_cmp, 1]     # 3rd element - keep track of number of original points in grid
        # # Otherwise aggregate labels/update features if necessary
        # else:
        # # TODO take closest coordinate to centre of low-res grid instead of first seen
        #     _, _, prev_smallest_grid_dist_cmp, _ = coord_bins[cur_grid_key]
        #     coord_bins[cur_grid_key][1] += labels
        #     coord_bins[cur_grid_key][3] += 1            # 3rd element - keep track of number of original points in grid

        # ########## TODO Take first label seen in downsampled grid as 'the one' ##########
        # # just create a separate function for it or make it modular?

    # Extract reduced features and multi-labels
    reduced_coords = np.array(list(coord_bins.keys()))                      # Get coord list from coord dict keys
    reduced_features_and_labels= np.array(list(coord_bins.values()))        # Get all values from coord keys
    grid_bins = reduced_features_and_labels[:,3].astype(int)                # 
    reduced_features = np.concatenate(reduced_features_and_labels[:,0]) \
            .reshape(reduced_coords.shape[0], data.shape[1])
    if len(reduced_features_and_labels[:,1].shape) == 2:
        reduced_mlabels = np.concatenate(reduced_features_and_labels[:,1]) \
            .reshape(reduced_coords.shape[0], label_counts.shape[1])
    else:
        reduced_mlabels = np.array(reduced_features_and_labels[:,1], dtype=np.int64)

    # reduced_stats = label_stats(reduced_mlabels)

    print("=============== SUMMARY STATS ===============")
    print("Min points contained in one bin: {}\nMax points contained in one bin: {}\nAverage points contained in grid bins: {}\n"\
            .format(np.min(grid_bins), np.max(grid_bins), np.average(grid_bins)))
    print("=============================================")


    # hclusters = hierarchy.linkage(coords, 'ward')
    # code.interact(local=locals())

    return reduced_coords, reduced_features, reduced_mlabels, red_idxs

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

    # Python list of child nodes at the dendrogram node
    child_idxs = np.array(find_child_nodes_in_dendrogram(dendrogram, dend_entry[0], num_points) +
                 find_child_nodes_in_dendrogram(dendrogram, dend_entry[1], num_points)).astype(int)

    # Check that none of these child nodes have yet been assigned clusters
    unassigned_count = np.sum(cluster_assignments[child_idxs] == 0)

    if unassigned_count == child_idxs.shape[0]:
        cluster_assignments[child_idxs] = cluster_idx
        return child_idxs.shape[0]

    # Some child nodes already have a cluster!
    cluster_assignments[np.where(cluster_assignments[child_idxs] == 0)] = cluster_idx
    return unassigned_count

def downsample_limited_nearest_points(coords, dendrogram, data, label_counts, clust_dist=21, clust_size=5):
    """
    Uses hierarchical clustering to group points into clusters based on criteria:
        1. Cannot exceed more than a certain total count of label counts
        2. Cannot exceed more than a certain number of points per aggregation cluster
        3. Points grouped together in cluster must not exceed distance X
    """
    clusters = {}
    cluster_assignments = np.full(coords.shape[0], 0, dtype=np.int64)
    cluster_idx = 1
    points_assigned_count = 0

    # Iterate over all dendrogram entries to create desired clusters
    for dend_entry in reversed(dendrogram):

        # Check if necessary conditions to form a cluster are met
        if cluster_cond_check(dend_entry, clust_dist, clust_size):
            points_assigned_count += assign_points_in_cluster(coords.shape[0], cluster_idx, dendrogram, dend_entry, cluster_assignments)
            cluster_idx += 1

            # Stop going through dendrogram once all points have been assigned
            if points_assigned_count == coords.shape[0]:
                break

    # return cluster_assignments
    # TODO return all the *ACTUAL* stuff
    reduced_data = {}
    for idx, entry in enumerate(cluster_assignments):

        # For non-0 index (unassigned) cluster-ids
        if entry != 0:
            # TODO do an update on coords and features here too - rules TBC!
            if entry not in reduced_data:
                reduced_data[entry] = [coords[idx], data[idx], label_counts[idx]]
            else:
                reduced_data[entry][-1] += label_counts[idx]

        else:
            reduced_data[entry] = [coords[idx], data[idx], label_counts[idx]]

    reduced_data_dict_vals = np.array(list(reduced_data.values()))
    reduced_coords      = np.concatenate(reduced_data_dict_vals[:,0]).reshape(reduced_data_dict_vals.shape[0], 2)
    reduced_features    = np.concatenate(reduced_data_dict_vals[:,1]).reshape(reduced_data_dict_vals.shape[0], data.shape[1])
    if len(reduced_data_dict_vals[:,2].shape) == 2:
        reduced_mlabels = np.concatenate(reduced_data_dict_vals[:,2]).reshape(reduced_data_dict_vals.shape[0], label_counts.shape[1])
    else:
        reduced_mlabels = reduced_data_dict_vals[:,2]

    return reduced_coords, reduced_features, reduced_mlabels

def cluster_cond_check(dend_row, clust_dist, clust_size):
    """
    Function for checking whether conditions for a dendrogram row can be taken as a cluster
    """
    return dend_row[2] <= clust_dist and dend_row[3] <= clust_size

def downsample_spatial_data(coords, data, label_counts, method='cluster-rules'):
    """
    Takes input *data* (any number of dimensions) and *label_counts*, where each row
    corresponds to an array of 'bincounts' that can be added to each other that would
    correspond to aggregation of points
    """
    if method == 'cluster-rules':
        dendrogram = np.load('data/hcluster_idxs.npy')
        return downsample_limited_nearest_points(coords, dendrogram, data, label_counts)
    elif method == 'fixed-grid':
        return downsample_by_fixed_grid(coords, data, label_counts, reduction_factor=1.1)
