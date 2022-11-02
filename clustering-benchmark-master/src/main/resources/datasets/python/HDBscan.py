import matplotlib.pyplot as plt
from scipy.io import arff
import time
import hdbscan

path = '../artificial'

# EASY CLUSTERING
# databrut = arff.loadarff(open(path+"/smile1.arff", 'r'))
# dist = 0.035
# k = 5

# databrut = arff.loadarff(open(path+"/banana.arff", 'r'))
# dist = 0.035
# k = 5

# COMPLEX CLUSTERING
databrut = arff.loadarff(open(path+"/diamond9.arff", 'r'))

# databrut = arff.loadarff(open(path + "/aggregation.arff", 'r'))
# dist = 1.8
# k = 5

# Donnees dans X
X = [[x[0], x[1]] for x in databrut[0]]
f0 = [f[0] for f in X]
f1 = [f[1] for f in X]

tps1 = time.time()
clusterer = hdbscan.HDBSCAN(min_cluster_size=10)
cluster_labels = clusterer.fit_predict(X)
tps2 = time.time()

plt.scatter(f0, f1, c=cluster_labels)
plt.title("HDBSCAN result")
plt.show()
