import code
import math
import numpy as np

def round_down(num, divisor):
    return num - (num % divisor)

def find_nearest_grid(x_coord, y_coord, x_step, y_step):
    """
    Given an x and y coord, finds the nearest decreased resolution gridbox
    that would contain this coordinate (points belong to the grid with 
    relative origin in the upper left coord of the low-res system). This is
    done by simply rounding the x_coord, y_coord to the next multiple of
    x_step, y_step **down**, respectively.
    """
    return round_down(x_coord, x_step), round_down(y_coord, y_step)

def downsample_spatial_data(coords, data, label_counts, reduction_factor=4):
    """
    Takes input *data* (any number of dimensions) and *label_counts*, where each row
    corresponds to an array of 'bincounts' that can be added to each other that would
    correspond to aggregation of points
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
    x_y_ratio = (y_max-y_min)/(x_max-x_min)
    unit = reduced_point_count/(x_y_ratio+1)
    x_block_cnt = math.ceil(unit)
    y_block_cnt = reduced_point_count-x_block_cnt

    # Build coordinates in low-res space
    x_step = (x_max-x_min)/x_block_cnt
    reduced_x_coords = np.arange(x_min, x_max+x_step, x_step)
    y_step = (y_max-y_min)/y_block_cnt
    reduced_y_coords = np.arange(y_min, y_max+y_step, y_step)
    x, y = np.meshgrid(reduced_x_coords, reduced_y_coords)
    
    # TODO transform x, y into the origin of each grid
    #   take out last row and last column - remaining should achieve this

    code.interact(local=locals())
