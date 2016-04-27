#!/bin/python3

import numpy as np

bath_and_dom_lowres = np.load('./bathAndDomLabel_lowres.npz')
label = bath_and_dom_lowres['label']
labelcounts = bath_and_dom_lowres['labelcounts']
bath_locations = bath_and_dom_lowres['locations']
features = bath_and_dom_lowres['features']

querypoints_lowres = np.load('./queryPoints_lowres_v2_.npz')
qp_locations = querypoints_lowres['locations']
validQueryID = querypoints_lowres['validQueryID']
x_bins = querypoints_lowres['x_bins']
query = querypoints_lowres['query']
y_bins = querypoints_lowres['y_bins']

print(len(querypoints_lowres['locations']))
