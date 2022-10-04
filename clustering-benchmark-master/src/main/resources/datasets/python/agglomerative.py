from operator import mod
import scipy.cluster.hierarchy as shc
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn import cluster
from sklearn.metrics import davies_bouldin_score, silhouette_score, rand_score
from scipy.io import arff
import kmedoids
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import manhattan_distances

path = '../artificial'

# EASY CLUSTERING
databrut = arff.loadarff(open(path+"/xclara.arff", 'r'))
# k = 3

# databrut = arff.loadarff(open(path+"/square2.arff", 'r'))
# k=4

# COMPLEX CLUSTERING

# databrut = arff.loadarff(open(path+"/smile1.arff", 'r'))
# k=4

# databrut = arff.loadarff(open(path+"/complex8.arff", 'r'))
# k=8

#Donnees dans datanp
datanp = [[x[0],x[1]] for x in databrut[0]]
datanp1 = [[x[2]] for x in databrut[0]]

f0 = [f[0] for f in datanp]
f1 = [f[1] for f in datanp]
true_labels = [f[0] for f in datanp1]

print("Dendogramme 'single' donnees initiales")

linked_mat = shc.linkage(datanp, 'single')

plt.figure(figsize=(12,12))
shc.dendrogram(linked_mat, orientation='top', distance_sort='descending', show_leaf_counts=False)
plt.show()

tps1=time.time()
model=cluster.AgglomerativeClustering(distance_threshold=10,
linkage='single', n_clusters=None)
model=model.fit(datanp)
tps2=time.time()

labels=model.labels_
k=model.n_clusters_
leaves=model.n_leaves_
plt.scatter(f0, f1, c=labels, s=8)
plt.title("Resultat du clustering")
plt.show()
print("nb clusters = ", k, " , nb feuilles = ", leaves, "runtime = ", round((tps2 -tps1)*1000,2),"ms")

k=4
tps1=time.time()
model = cluster.AgglomerativeClustering(linkage = 'single',n_clusters=k)
model = model.fit(datanp)
tps2 = time.time()

labels = model.labels_
kres = model.n_clusters_
leaves = model.n_leaves_

plt.scatter(f0, f1, c=labels)
plt.title("Donnees apres clustering single algorithm")
plt.show()