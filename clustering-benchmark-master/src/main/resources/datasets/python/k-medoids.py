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
# databrut = arff.loadarff(open(path+"/xclara.arff", 'r'))
# k = 3

databrut = arff.loadarff(open(path + "/square2.arff", 'r'))
k=4

# COMPLEX CLUSTERING

# databrut = arff.loadarff(open(path+"/smile1.arff", 'r'))
# k=4

# databrut = arff.loadarff(open(path+"/complex8.arff", 'r'))
# k=8

datanp = [[x[0], x[1]] for x in databrut[0]]
datanp1 = [[x[2]] for x in databrut[0]]

f0 = [f[0] for f in datanp]
f1 = [f[1] for f in datanp]
true_labels = [f[0] for f in datanp1]

print("Appel KMeans pour une valeur fixee de k")

old_Davies = -1
old_Silhouette = -2
distmatrix = manhattan_distances(datanp)
# for k in range(2, 8):
#     tps1 = time.time()
#     # distmatrix = euclidean_distances(datanp)
#     fp = kmedoids.fasterpam(distmatrix, k)
#     tps2 = time.time()
#     iter_kmed = fp.n_iter
#     labels_kmed = fp.labels
#     new_Davies = davies_bouldin_score(distmatrix, labels_kmed)
#     new_Silhouette = silhouette_score(distmatrix, labels_kmed)
#     if (old_Davies < 0 or new_Davies < old_Davies):
#         best_k_for_Davies = k
#         old_Davies = new_Davies
#         best_labels = labels_kmed
#     if (old_Silhouette < -1 or new_Silhouette > old_Silhouette):
#         best_k_for_Silhouette = k
#         old_Silhouette = new_Silhouette

fp = kmedoids.fasterpam(distmatrix, k)
print("Loss with FatsterPAM:", fp.loss)
best_labels = fp.labels
plt.scatter(f0, f1, c=best_labels, s=8)
plt.title("Donnees apres clustering KMedoids")
plt.show()
# print("nb cluster =", k, ", nb iter =", iter_kmed," , runtime = ", round((tps2-tps1)*1000,2), "ms")
# print("Davies Boouldin Score = ", best_k_for_Davies)
# print("Silhouette Score = ", best_k_for_Silhouette)
# print(databrut)
# print(best_labels)
model = cluster.KMeans(n_clusters=k, init='k-means++')
model.fit(datanp)
print("Rand Score = ", rand_score(model.labels_, best_labels))
