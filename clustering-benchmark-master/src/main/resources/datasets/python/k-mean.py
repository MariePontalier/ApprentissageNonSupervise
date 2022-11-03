import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn import cluster
from sklearn.metrics import davies_bouldin_score, silhouette_score, rand_score
from scipy.io import arff

path = '../artificial'

# EASY CLUSTERING
# databrut = arff.loadarff(open(path+"/xclara.arff", 'r'))
# k = 3

databrut = arff.loadarff(open(path + "/square2.arff", 'r'))
k=4

# COMPLEX CLUSTERING

# databrut = arff.loadarff(open(path + "/smile1.arff", 'r'))
# k = 4

# databrut = arff.loadarff(open(path+"/complex8.arff", 'r'))
# k=8

# print(databrut[0])
datanp = [[x[0], x[1]] for x in databrut[0]]
datanp1 = [[x[2]] for x in databrut[0]]

f0 = [f[0] for f in datanp]
f1 = [f[1] for f in datanp]
true_labels = [f[0] for f in datanp1]

print("Appel KMeans pour une valeur fixee de k")

# old_Davies = -1
# old_Silhouette = -2
# for k in range(2, 50):
#     tps1 = time.time()
#     model = cluster.KMeans(n_clusters=k, init='k-means++')
#     model.fit(datanp)
#     tps2 = time.time()
#     labels = model.labels_
#     iteration = model.n_iter_
#     new_Davies = davies_bouldin_score(datanp, labels)
#     new_Silhouette = silhouette_score(datanp, labels)
#     if (old_Davies < 0 or new_Davies < old_Davies):
#         best_k_for_Davies = k
#         old_Davies = new_Davies
#         best_labels = labels
#     if (old_Silhouette < -1 or new_Silhouette > old_Silhouette):
#         best_k_for_Silhouette = k
#         old_Silhouette = new_Silhouette

model = cluster.KMeans(n_clusters=k, init='k-means++')
model.fit(datanp)
tps2 = time.time()
best_labels = model.labels_

plt.scatter(f0, f1, c=best_labels)
plt.title("Donnees apres clustering Kmeans")
plt.show()
# print("nb cluster =", k, ", nb iter =", iteration, " , runtime = ", round((tps2 - tps1) * 1000, 2), "ms")
# print("Davies Boouldin best K = ", best_k_for_Davies)
# print("Silhouette best K = ", best_k_for_Silhouette)
print("Rand Score = ", rand_score(true_labels, best_labels))
