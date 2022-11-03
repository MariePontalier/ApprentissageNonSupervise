import time
import kmedoids
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import cluster
from sklearn.metrics import davies_bouldin_score, silhouette_score, rand_score
from sklearn.metrics.pairwise import manhattan_distances
from scipy.io import arff


# Return a matrix from txt dataset file
def loadCSV(filename: str):
    path = "../dataset-rapport/dataset-rapport/"
    databrut = pd.read_csv(path + filename, sep=" ", encoding="ISO-8859-1", skipinitialspace=True)
    return databrut.to_numpy()


# Return matrix from arff file
def loadArff(filename):
    databrut = arff.loadarff(open("../artificial/" + filename, 'r'))
    datanp = [[x[0], x[1]] for x in databrut[0]]
    return datanp


# plot data
def plotDATA(data):
    f0 = [f[0] for f in data]
    f1 = [f[1] for f in data]
    plt.scatter(f0, f1, s=8)
    plt.title("Donnees initiales")
    plt.show()


def plotResults(data, title, labels):
    f0 = [f[0] for f in data]
    f1 = [f[1] for f in data]
    plt.scatter(f0, f1, c=labels)
    plt.title(title)
    plt.show()


def plotsDatas(data, titles, labels):
    f0 = [f[0] for f in data]
    f1 = [f[1] for f in data]
    for i in range(len(labels)):
        print(titles[i])
        plt.scatter(f0, f1, c=labels[i])
        plt.title(titles[i])
    plt.show()


# Return a dict with all info for the best k (K, labels, Davies Score, time of exec)
def kMeanAuto(data):
    results = dict()
    tps0 = time.time()
    old_Davies = -1
    for k in range(2, 50):
        tps1 = time.time()
        model = cluster.KMeans(n_clusters=k, init='k-means++')
        model.fit(data)
        tps2 = time.time()
        labels = model.labels_
        new_Davies = davies_bouldin_score(data, labels)
        if (old_Davies < 0 or new_Davies < old_Davies):
            execution_time = round((tps2 - tps1) * 1000, 2)
            best_k_for_Davies = k
            old_Davies = new_Davies
            best_labels = labels
    tps3 = time.time()
    results['K'] = best_k_for_Davies
    results["labels"] = best_labels
    results["Davies Score"] = old_Davies
    results["time"] = execution_time
    results["total time"] = round((tps3 - tps0) * 1000, 2)
    return results


# Return a dict with all info for the best k (K, labels, Davies Score, time of exec)
def kMedoidsAuto(data):
    results = dict()
    old_Davies = -1
    tps0 = time.time()
    distmatrix = manhattan_distances(data)
    for k in range(2, 50):
        tps1 = time.time()
        model = kmedoids.fasterpam(distmatrix, k)
        tps2 = time.time()
        labels = model.labels
        new_Davies = davies_bouldin_score(data, labels)
        if (old_Davies < 0 or new_Davies < old_Davies):
            execution_time = round((tps2 - tps1) * 1000, 2)
            best_k_for_Davies = k
            old_Davies = new_Davies
            best_labels = labels
    tps3 = time.time()
    results['K'] = best_k_for_Davies
    results["labels"] = best_labels
    results["Davies Score"] = old_Davies
    results["time"] = execution_time
    results["total time"] = round((tps3 - tps0) * 1000, 2)
    return results


def aggloAutoDist(data):
    results = {}
    cluster_link_method = ['single', 'average', 'complete', 'ward']
    old_Davies = -1
    tps0 = time.time()
    for method in cluster_link_method:
        for dist in np.arange(0, 0.2, 0.025):
            tps1 = time.time()
            model = cluster.AgglomerativeClustering(distance_threshold=dist,
                                                    linkage=method, n_clusters=None)
            model = model.fit(data)
            tps2 = time.time()
            labels = model.labels_
            k = model.n_clusters_
            if k > 1 and k < len(data):
                new_Davies = davies_bouldin_score(data, labels)
                if (old_Davies < 0 or new_Davies < old_Davies):
                    old_Davies = new_Davies
                    results['K'] = k
                    results["labels"] = labels
                    results["Davies Score"] = old_Davies
                    results["time"] = round((tps2 - tps1) * 1000, 2)
                    results["method"] = method

        for dist in range(1, 20):
            tps1 = time.time()
            model = cluster.AgglomerativeClustering(distance_threshold=dist,
                                                    linkage=method, n_clusters=None)
            model = model.fit(data)
            tps2 = time.time()
            labels = model.labels_
            k = model.n_clusters_
            if k > 1 and k < len(data):
                new_Davies = davies_bouldin_score(data, labels)
                if (old_Davies < 0 or new_Davies < old_Davies):
                    old_Davies = new_Davies
                    results['K'] = k
                    results["labels"] = labels
                    results["Davies Score"] = old_Davies
                    results["time"] = round((tps2 - tps1) * 1000, 2)
                    results["method"] = method
    tps3 = time.time()
    results["total time"] = round((tps3 - tps0) * 1000, 2)
    return results


def aggloAutoK(data):
    results = {}
    cluster_link_method = ['single', 'average', 'complete', 'ward']
    old_Davies = -1
    tps0 = time.time()
    for method in cluster_link_method:
        for k in range(2, 50):
            tps1 = time.time()
            model = cluster.AgglomerativeClustering(linkage=method, n_clusters=k)
            model = model.fit(data)
            tps2 = time.time()

            labels = model.labels_
            k = model.n_clusters_
            if k < len(data):
                new_Davies = davies_bouldin_score(data, labels)
                if (old_Davies < 0 or new_Davies < old_Davies):
                    old_Davies = new_Davies
                    results['K'] = k
                    results["labels"] = labels
                    results["Davies Score"] = old_Davies
                    results["time"] = round((tps2 - tps1) * 1000, 2)
                    results["method"] = method
    tps3 = time.time()
    results["total time"] = round((tps3 - tps0) * 1000, 2)
    return results


# Return labels using agglo method chosing k
def AgloBasicK(data, k, method):
    model = cluster.AgglomerativeClustering(linkage=method, n_clusters=k)
    model = model.fit(data)
    return model.labels_


# Return labels using agglo method chosing dist
def AgloBasicDist(data, dist, method):
    model = cluster.AgglomerativeClustering(distance_threshold=dist,
                                            linkage=method, n_clusters=None)
    model = model.fit(data)
    return model.labels_


# datanp = loadArff("smile1.arff")
# model = cluster.AgglomerativeClustering(distance_threshold=0.05,
#                                         linkage="single", n_clusters=None)
# model = model.fit(datanp)
# plotResults(datanp, "Agglomerative clustering", model.labels_)


# datanp = loadCSV("x1.txt")
# datanp = loadArff("xclara.arff")
# resultsK = aggloAutoK(datanp)

# results = aggloAutoK(datanp)
# labels = AgloBasicK(datanp, 3, "single")
# labels1 = AgloBasicK(datanp, 3, "average")
# f0 = [f[0] for f in datanp]
# f1 = [f[1] for f in datanp]

# plt.scatter(f0, f1, c=results["labels"])
# plt.title("Metric evaluation")
# plt.show()

# fig, (ax1, ax2) = plt.subplots(1, 2)
# ax1.scatter(f0, f1, c=labels)
# ax1.set_title("Manual evaluation single method")
# ax2.scatter(f0, f1, c=labels1)
# ax2.set_title("Manual evaluation average method")
# plt.show()
# plotsDatas(datanp, ["Metric evaluation", "Manual evaluation"], [results["labels"], labels])
# plotResults(datanp, "Agglomerative clustering", results["labels"])
# print(results["method"])

datanp1 = loadCSV("x1.txt")
datanp2 = loadCSV("x2.txt")
datanp3 = loadCSV("x3.txt")
datanp4 = loadCSV("x4.txt")
datanp5 = loadCSV("y1.txt")
datanp6 = loadCSV("zz1.txt")
datanp7 = loadCSV("zz2.txt")
model = cluster.KMeans(n_clusters=5, init='k-means++')
model.fit(datanp7)
plotResults(datanp7, "K-mean k=5", model.labels_)
# plotDATA(datanp1)
# plotDATA(datanp2)
# plotDATA(datanp3)
# plotDATA(datanp4)
# plotDATA(datanp5)
# plotDATA(datanp6)
# plotDATA(datanp7)

# kmean1 = kMeanAuto(datanp1)
# kmean2 = kMeanAuto(datanp2)
# kmean3 = kMeanAuto(datanp3)
# kmean4 = kMeanAuto(datanp4)
# kmean5 = kMeanAuto(datanp5)
# kmean6 = kMeanAuto(datanp6)
# kmean7 = kMeanAuto(datanp7)
# print("Kmean1 : ", kmean1)
# print("Kmean2 : ", kmean2)
# print("Kmean3 : ", kmean3)
# print("Kmean4 : ", kmean4)
# print("Kmean5 : ", kmean5)
# print("Kmean6 : ", kmean6)
# print("Kmean7 : ", kmean7)

# kMedoid1 = kMedoidsAuto(datanp1)
# kMedoid2 = kMedoidsAuto(datanp2)
# kMedoid3 = kMedoidsAuto(datanp3)
# kMedoid4 = kMedoidsAuto(datanp4)
# # kMedoid5 = kMedoidsAuto(datanp5)
# kMedoid6 = kMedoidsAuto(datanp6)
# kMedoid7 = kMedoidsAuto(datanp7)
# print("Kmedoid1 : ", kMedoid1)
# print("Kmedoid2 : ", kMedoid2)
# print("Kmedoid3 : ", kMedoid3)
# print("Kmedoid4 : ", kMedoid4)
# # print("Kmedoid5 : ", kMedoid5)
# print("Kmedoid6 : ", kMedoid6)
# print("Kmedoid7 : ", kMedoid7)

# aggloDist1 = aggloAutoDist(datanp1)
# aggloDist2 = aggloAutoDist(datanp2)
# aggloDist3 = aggloAutoDist(datanp3)
# aggloDist4 = aggloAutoDist(datanp4)
# # aggloDist5 = aggloAutoDist(datanp5)
# aggloDist6 = aggloAutoDist(datanp6)
# aggloDist7 = aggloAutoDist(datanp7)
# print("AggloDist1 : ", aggloDist1)
# print("AggloDist2 : ", aggloDist2)
# print("AggloDist3 : ", aggloDist3)
# print("AggloDist4 : ", aggloDist4)
# # print("AggloDist5 : ", aggloDist5)
# print("AggloDist6 : ", aggloDist6)
# print("AggloDist7 : ", aggloDist7)

# aggloK1 = aggloAutoK(datanp1)
# aggloK2 = aggloAutoK(datanp2)
# aggloK3 = aggloAutoK(datanp3)
# aggloK4 = aggloAutoK(datanp4)
# aggloK6 = aggloAutoK(datanp6)
# aggloK7 = aggloAutoK(datanp7)
# print("AggloK1 : ", aggloK1)
# print("AggloK2 : ", aggloK2)
# print("AggloK3 : ", aggloK3)
# print("AggloK4 : ", aggloK4)
# print("AggloK6 : ", aggloK6)
# print("AggloK7 : ", aggloK7)

# aggloK5 = aggloAutoK(datanp5)
# print("AggloK5 : ", aggloK5)

# fig, axs = plt.subplots(2, 2)
# axs[0, 0].scatter([f[0] for f in datanp7], [f[1] for f in datanp7], c=kmean7["labels"], s=8)
# axs[0, 0].set_title("K-Means")
# axs[0, 1].scatter([f[0] for f in datanp7], [f[1] for f in datanp7], c=kMedoid7["labels"], s=8)
# axs[0, 1].set_title("K-Medoids")
# axs[1, 0].scatter([f[0] for f in datanp7], [f[1] for f in datanp7], c=aggloDist7["labels"], s=8)
# axs[1, 0].set_title("Agglo Dist")
# axs[1, 1].scatter([f[0] for f in datanp7], [f[1] for f in datanp7], c=aggloK7["labels"], s=8)
# axs[1, 1].set_title("Agglo K")
# plt.show()

# print("Rand kmean1 kmed1 : ", rand_score(kmean1["labels"], kMedoid1["labels"]))
# print("Rand kmean1 aggloDist1 : ", rand_score(kmean1["labels"], aggloK1["labels"]))
# print("Rand aggloDist1 Kmed1 : ", rand_score(aggloK1["labels"], kMedoid1["labels"]))
# print("Rand kmean5 kmed5 : ", rand_score(kmean6, kMedoid6))
# print("Rand kmean5 aggloDist5 : ", rand_score(kmean6, aggloK6))
# print("Rand aggloDist5 Kmed5 : ", rand_score(aggloK6, kMedoid6))
# print("Rand kmean7 kmed7 : ", rand_score(kmean7, kMedoid7))
# print("Rand kmean7 aggloDist7 : ", rand_score(kmean7, aggloK7))
# print("Rand aggloDist7 Kmed7 : ", rand_score(aggloK7, kMedoid7))
