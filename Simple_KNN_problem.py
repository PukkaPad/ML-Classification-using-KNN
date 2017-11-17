# -*- coding: UTF-8 -*-
# K-Nearest Neighbour

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from sklearn.metrics import accuracy_score
from matplotlib.colors import ListedColormap

# Creating 25 family members to train the model
# I am creating a 25 x 4 arrays of float32
# between 0 and 99
trainDataset = np.random.randint(0,100,(25,2)).astype(np.float32)

# 0 will be Red and 1 will be Blue family
responses = np.random.randint(0,2,(25,1)).astype(np.float32)


# Get and plot Red family
red = trainDataset[responses.ravel()==0]
plt.scatter(red[:,0],red[:,1],80,'r','^')
#plt.show()

# Get and plot Blue family
blue = trainDataset[responses.ravel()==1]
plt.scatter(blue[:,0],blue[:,1],80,'b','s')


#plt.show()

# Generate the new-comer
newcomer = np.random.randint(0,100,(1,2)).astype(np.float32)
plt.scatter(newcomer[:,0],newcomer[:,1],80,'g','o')

# Fitting K-NN to the training set
nbrs = KNeighborsClassifier(n_neighbors = 3,
    metric = 'minkowski', p = 2).fit(trainDataset, responses)

pred = nbrs.predict(newcomer)


distances, indices = nbrs.kneighbors(newcomer)
print "number of Red",  len(red)
print "number of Blue",  len(blue)

print pred.item()
if pred.item == 0.0:
    print 'New-comer will be Red'
else: print "New-comer will be Blue"


print "neighbours: "
for val, dist in zip (indices, distances):

    for num, dist_ in zip (val, dist):
        if num in red:
            print "Red, {0}. Distance {1} ".format(num, dist_)
        else:

            print "Blue, {0}. Distance {1} ".format(num, dist_)


plt.show()
