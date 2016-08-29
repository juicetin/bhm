import numpy as np
from pandas import read_csv
import pdb

# Load all the data
def load_training_data():
    bath_and_dom_lowres = np.load('data/bathAndDomLabel_lowres.npz')
    label = bath_and_dom_lowres['label']
    labelcounts = bath_and_dom_lowres['labelcounts']
    bath_locations = bath_and_dom_lowres['locations']
    features = bath_and_dom_lowres['features']
    
    return (label, labelcounts, bath_locations, features)

def load_multi_label_data():
    multi_label_data = np.load('data/multiple_labels_training_data.npz')
    locations = multi_label_data['locations']
    labels = multi_label_data['labels']
    features = multi_label_data['features']    
    return locations, features, labels

def load_test_data():
    querypoints_lowres = np.load('data/queryPoints_lowres_v2_.npz')
    qp_locations = querypoints_lowres['locations']
    validQueryID = querypoints_lowres['validQueryID']
    x_bins = querypoints_lowres['x_bins']
    query = querypoints_lowres['query']
    y_bins = querypoints_lowres['y_bins']

    return (qp_locations, validQueryID, x_bins, query, y_bins)

def mini_batch_idxs(labels, point_count, split_type):

    # Match ratios of original labels
    if split_type == 'stratified':
        sss = StratifiedShuffleSplit(labels, 1, test_size=point_count/len(labels))
        for train_index, test_index in sss:
            pass
        return test_index

    # Have even number of labels ignoring original ratios
    elif split_type == 'even':

        # Find how many of each class to generate
        uniq_classes = np.unique(labels)
        num_classes = len(uniq_classes)
        class_size = int(point_count/num_classes)
        class_sizes = np.full(num_classes, class_size, dtype='int64')

        # Adjust for non-divisiblity
        rem = point_count % class_size
        if rem != 0:
            class_sizes[-1] += rem

        # Generate even class index list
        class_idxs = np.concatenate(np.array(
            [np.random.choice(np.where(labels==cur_class)[0], cur_class_size, replace=False)
                for cur_class, cur_class_size 
                in zip(uniq_classes, class_sizes)
            ]
        ), axis=0)

        return class_idxs

def summarise_list(old_lst, label_map):
    new_lst = np.copy(old_lst)
    for k, v in label_map.items():
        new_lst[old_lst == k] = v
    return new_lst

def summarised_labels(labels):

    label_map={1:0,2:0,3:1,4:3,5:1,6:3,7:3,8:3,9:3,10:1,11:3,12:3,13:2,14:2,15:2,16:1,17:1,18:0,19:1,20:0,21:0,22:1,23:0,24:0}

    if isinstance(labels[0], list):
        # for i, lst in enumerate(new_labels):
        #     summarise_list(np.array(lst), np.array(labels[i]), label_map)
        return np.array([summarise_list(np.array(labels[i]), label_map) for i in range(labels.shape[0])])
    else:
        return summarise_list(labels, label_map)
        # for k, v in label_map.items(): 
        #     new_labels[labels==k] = v

def csv_to_npz(filename):
    data = read_csv(filename, sep=',')
    return data

def fill(labels, num_uniqs):
    counts = np.bincount(labels)
    missing = num_uniqs - len(counts)
    if missing > 0:
        counts = np.concatenate((counts, [0] * missing), axis=0)
    # if (len(counts) != 25):
    #     pdb.set_trace()
    #     print(len(counts))
    return list(counts)[1:]


def multi_label_counts(labels):
    """
    Converts lists of uneven category counts into a bincount of them in a uniform matrix 
    """

    uniqs = np.unique(np.concatenate(labels, axis=0))
    num_uniqs = uniqs.shape[0]
    multi_labels = np.array([fill(labellist, num_uniqs+1) for labellist in labels])
    multi_labels = np.concatenate(multi_labels, axis=0)
    multi_labels = multi_labels.reshape(labels.shape[0], num_uniqs)

    return multi_labels
