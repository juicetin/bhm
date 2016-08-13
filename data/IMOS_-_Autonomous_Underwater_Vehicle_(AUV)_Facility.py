#!/usr/bin/env python

import csv
import urllib2
import StringIO
import sys

# The URL to the collection (as comma-separated values).
collection_url = "http://geoserver-123.aodn.org.au/geoserver/ows?typeName=auv_trajectory_st_data&SERVICE=WFS&outputFormat=csv&REQUEST=GetFeature&VERSION=1.0.0&CQL_FILTER=INTERSECTS(geom%2CPOLYGON((121.95013046265001%20-14.194797515869%2C121.95013046265001%20-14.048885345459%2C121.75237655639998%20-14.048885345459%2C121.75237655639998%20-14.194797515869%2C121.95013046265001%20-14.194797515869)))"

# Fetch data...
response = urllib2.urlopen(collection_url)

f = open('filename.txt', 'w')

# Iterate on data...
for row in csv.reader(response):
    print >>f, ','.join(row)
