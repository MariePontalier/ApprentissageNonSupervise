from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import arff
import time
from sklearn import cluster

path = '../artificial'

# EASY CLUSTERING
# databrut = arff.loadarff(open(path+"/smile1.arff", 'r'))
# dist = 0.035
# k = 5

# databrut = arff.loadarff(open(path+"/banana.arff", 'r'))
# dist = 0.035
# k = 5

# COMPLEX CLUSTERING
# databrut = arff.loadarff(open(path+"/xclara.arff", 'r'))
# dist = 7
# k = 7

# databrut = arff.loadarff(open(path + "/aggregation.arff", 'r'))
# dist = 1.8
# k = 5

databrut = arff.loadarff(open(path + "/diamond9.arff", 'r'))
dist = 0.225
k = 5

# Donnees dans X
X = [[x[0], x[1]] for x in databrut[0]]
f0 = [f[0] for f in X]
f1 = [f[1] for f in X]

# Distances k plus proches voisins

neigh = NearestNeighbors(n_neighbors=k)
neigh.fit(X)
distances, indices = neigh.kneighbors(X)
# retirer le point " origine "
newDistances = np.asarray([np.average(distances[i][1:]) for i in range(0, distances.shape[0])])
trie = np.sort(newDistances)
plt.title(" Plus proches voisins (5) ")
plt.plot(trie)
plt.show()

tps1 = time.time()
model = cluster.DBSCAN(eps=dist, min_samples=k)
model = model.fit(X)
tps2 = time.time()

labels = model.labels_

plt.scatter(f0, f1, c=labels)
plt.title("DBSCAN result")
plt.show()
