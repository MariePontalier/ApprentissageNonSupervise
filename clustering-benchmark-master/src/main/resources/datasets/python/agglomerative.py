from operator import mod
import scipy.cluster.hierarchy as shc
import numpy as np
import matplotlib.pyplot as plt
import time
import numpy as np
from sklearn import cluster
from sklearn.metrics import davies_bouldin_score, silhouette_score, rand_score
from scipy.io import arff
import kmedoids
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import manhattan_distances

path = '../artificial'

# EASY CLUSTERING but hard to find good parameters with metrics

# databrut = arff.loadarff(open(path+"/smile1.arff", 'r'))
# k=4
# dist = 0.05

# databrut = arff.loadarff(open(path+"/banana.arff", 'r'))
# k=2
# dist = 0.05

# EASY CLUSTERING and easy to find good parameters
# databrut = arff.loadarff(open(path + "/hypercube.arff", 'r'))
# k=4

# COMPLEX CLUSTERING

# databrut = arff.loadarff(open(path+"/xclara.arff", 'r'))
# k = 3

# databrut = arff.loadarff(open(path+"/square2.arff", 'r'))
# k=4

# Donnees dans datanp
datanp = [[x[0], x[1]] for x in databrut[0]]
datanp1 = [[x[2]] for x in databrut[0]]

f0 = [f[0] for f in datanp]
f1 = [f[1] for f in datanp]
true_labels = [f[0] for f in datanp1]

# print("Dendogramme 'single' donnees initiales")
# linked_mat = shc.linkage(datanp, 'single')

# Search for best dist and nb clusters
results_dist = {}
cluster_link_method = ['single', 'average', 'complete', 'ward']

# for method in cluster_link_method:
#     for dist in np.arange(0,0.2,0.025):
#         tps1=time.time()
#         model=cluster.AgglomerativeClustering(distance_threshold=dist,
#         linkage=method, n_clusters=None)
#         model=model.fit(datanp)
#         tps2=time.time()

#         labels=model.labels_
#         k=model.n_clusters_
#         leaves=model.n_leaves_
#         if k > 1 and k < len(datanp):
#             results_dist[(method, dist)] = (k, davies_bouldin_score(datanp, labels))

#     for dist in range(1,20):
#         tps1=time.time()
#         model=cluster.AgglomerativeClustering(distance_threshold=dist,
#         linkage=method, n_clusters=None)
#         model=model.fit(datanp)
#         tps2=time.time()

#         labels=model.labels_
#         k=model.n_clusters_
#         leaves=model.n_leaves_
#         if k > 1 and k < len(datanp):
#             results_dist[(method, dist)] = (k, davies_bouldin_score(datanp, labels), round((tps2 -tps1)*1000,2))


# print(results_dist)
# # getting maximum value
# max_val = min(results_dist.values(), key=lambda sub: sub[1])

# # getting key with maximum value using comparison
# res = [key for key, val in results_dist.items() if val == max_val][0]

# best_Davies = max_val[1]
# nb_clusters = max_val[0]
# cut_dist = res[1]
# method = res[0]
# print("Best result of distance clustering has Davies score of ", best_Davies, " with ", nb_clusters,
#  " clusters and a dist cut of ", cut_dist, "and method ", method) 

# # plt.figure(figsize=(12,12))
# # shc.dendrogram(linked_mat, orientation='top', distance_sort='descending', show_leaf_counts=False)
# # plt.show()

# tps1=time.time()
# model=cluster.AgglomerativeClustering(distance_threshold=cut_dist,
# linkage=method, n_clusters=None)
# model=model.fit(datanp)
# tps2=time.time()

# labels=model.labels_
# k=model.n_clusters_
# leaves=model.n_leaves_
# plt.scatter(f0, f1, c=labels, s=8)
# plt.title("Resultat du clustering en coupant le dendogramme avec une distance")
# plt.show()
# print("nb clusters = ", k, " , nb feuilles = ", leaves, "runtime = ", round((tps2 -tps1)*1000,2),"ms")

# Find best K
# results_k = {}
# for method in cluster_link_method:
#     for k in range(2, 50):
#         tps1 = time.time()
#         model = cluster.AgglomerativeClustering(linkage=method, n_clusters=k)
#         model = model.fit(datanp)
#         tps2 = time.time()
#
#         labels = model.labels_
#         k = model.n_clusters_
#         leaves = model.n_leaves_
#         if k < len(datanp):
#             results_k[(method, k)] = (davies_bouldin_score(datanp, labels), round((tps2 - tps1) * 1000, 2))
#
# print(results_k)
# getting maximum value
# max_val = min(results_k.values(), key=lambda sub: sub[0])

# getting key with maximum value using comparison
# res = [key for key, val in results_k.items() if val == max_val][0]

# best_Davies = max_val[0]
# nb_clusters = res[1]
# method = res[0]
# print("Best result of distance clustering has Davies score of ", best_Davies, " with ", nb_clusters,
#       " clusters and method ", method)

# k=2
# tps1 = time.time()
# model = cluster.AgglomerativeClustering(linkage=method, n_clusters=nb_clusters)
# model = model.fit(datanp)
# tps2 = time.time()
#
# labels = model.labels_
# kres = model.n_clusters_
# leaves = model.n_leaves_
#
# plt.scatter(f0, f1, c=labels)
# plt.title("Donnees apres clustering en choisissant le nombre de clusters")
# plt.show()


model = cluster.AgglomerativeClustering(linkage="single", n_clusters=k)
model = model.fit(datanp)
plt.scatter(f0, f1, c=model.labels_)
plt.title("Donnees apres clustering en choisissant le nombre de clusters")
plt.show()