import code
import math
import numpy as np

def label_stats(multi_labels):
    non_0s = np.apply_along_axis(lambda x: np.sum(x!=0), 1, multi_labels)
    total_counts = np.apply_along_axis(np.sum, 1, multi_labels)

    return np.bincount(non_0s)[1:], total_counts

def round_down(num, divisor, base=0):
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

def downsample_by_fixed_grid(coords, data, label_counts, reduction_factor=4):
    # Get extremes of the coordinate system
    x_min = coords[:,0].min()
    x_max = coords[:,0].max()
    y_min = coords[:,1].min()
    y_max = coords[:,1].max()

    # Decide on number of points in reduced low-res space
    reduced_point_count = int(coords.shape[0]/reduction_factor) # default red_factor - 4

    # Calculate number of x/y blocks
    # Assumes x is longer than y!
    x_y_ratio = (y_max-y_min)/(x_max-x_min)
    unit = reduced_point_count/(x_y_ratio+1)
    x_block_cnt = math.ceil(unit)
    y_block_cnt = reduced_point_count-x_block_cnt

    # Build coordinates in low-res space
    # TODO x_step and y_step as long floats too troublesome - round up to nearest integer
    x_step = math.ceil((x_max-x_min)/x_block_cnt)
    reduced_x_coords = np.arange(x_min, x_max+x_step, x_step)
    y_step = math.ceil((y_max-y_min)/y_block_cnt)
    reduced_y_coords = np.arange(y_min, y_max+y_step, y_step)
    x_mesh, y_mesh = np.meshgrid(reduced_x_coords, reduced_y_coords)

    # Creates all bins with fixed overlaid low-res grid size
    # At the moment - takes the first seen coordinate in a grid to be the definitive one
    coord_bins = {}
    for (x_point, y_point), features, labels in zip(coords, data, label_counts):
        cur_grid_key = find_nearest_grid(x_point, x_step, x_min, y_point, y_step, y_min)
        
        # Create bin if doesn't exist yet
        if cur_grid_key not in coord_bins:
            coord_bins[cur_grid_key] = [features, labels]
        # Otherwise aggregate labels/update features if necessary
        else:
            # TODO take closest coordinate to centre of low-res grid instead of first seen
            coord_bins[cur_grid_key][1] += labels

    # Extract reduced features and multi-labels
    reduced_coords = np.array(list(coord_bins.keys()))
    reduced_features_and_labels= np.array(list(coord_bins.values()))
    reduced_features = np.concatenate(reduced_features_and_labels[:,0]).reshape(reduced_coords.shape[0], data.shape[1])
    reduced_mlabels = np.concatenate(reduced_features_and_labels[:,1]).reshape(reduced_coords.shape[0], label_counts.shape[1])

    orig_stats = label_stats(label_counts)
    reduced_stats = label_stats(reduced_mlabels)

    # TODO transform x, y into the origin of each grid
    #   take out last row and last column - remaining should achieve this

    code.interact(local=locals())

    return reduced_features, reduced_mlabels

def downsample_limited_nearest_points():
    pass

def downsample_spatial_data(coords, data, label_counts):
    """
    Takes input *data* (any number of dimensions) and *label_counts*, where each row
    corresponds to an array of 'bincounts' that can be added to each other that would
    correspond to aggregation of points
    """
    return downsample_by_fixed_grid(coords, data, label_counts, reduction_factor=10)
