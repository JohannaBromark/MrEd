from utils import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from sklearn.neighbors import KNeighborsClassifier


def euclidean_dist(v1, v2):
    return np.linalg.norm(v1 - v2)


def histogramish(dist):
    #the histogram of the data
    n, bins, patches = plt.hist(dist[78, :], bins=100, facecolor='green')

    plt.axis([0, 15, 0, 70])

    plt.show()


def averageDist(dist):
    avg_dist = []
    for i in range(len(test_set)):
        avg_dist = np.append(avg_dist, np.average(dist[i, :]))

    plt.hist(avg_dist, bins=20)
    plt.show()



def distance_measure():
    train_set, test_set = get_test_train_sets("features_targets/afe_feat_and_targ.txt", partition_num=0, seed=1)

    # Normalise the data
    train_set_norm, mean, std = normalise(train_set[:, 2:])
    test_set_norm = (test_set[:, 2:] - mean) / std

    dist_norm = np.zeros((len(test_set_norm), len(train_set_norm)))

    dist_threshold = 2.0

    for i in range(len(test_set_norm)):
        for k in range(len(train_set_norm)):
            dist_norm[i, k] = euclidean_dist(test_set_norm[i, :], train_set_norm[k, :])

    indx_test, indx_train = np.where(dist_norm < dist_threshold)

    unique_test, counted_test = np.unique(indx_test, return_counts=True)
    unique_train, counted_train = np.unique(indx_train, return_counts=True)

    # Pick the tracks in the training set that are close to at least 4 other tracks
    # The indices return will map to the index in unique_train, which maps to the song
    indx_frequent_train = np.where(counted_train > 3)
    track_idx_train = unique_train[indx_frequent_train]
    counted_tracks = counted_train[indx_frequent_train]
    train_tracks_close = train_set[track_idx_train, :]

    # The test tracks that are close to the train tracks with many close train
    test_sets_close = []
    for i in track_idx_train:
        idx_close_train = np.where(indx_train == i)
        idx_tets_close = indx_test[idx_close_train]
        test_sets_close.append([test_set[idx_tets_close, :]])

    print("help")


def knn_neighbor_count():
    for i in range(10):
        train_set, test_set = get_test_train_sets("features_targets/afe_feat_and_targ.txt", partition_num=i, seed=1)

        # Normalise the data
        train_set_norm, mean, std = normalise(train_set[:, 2:])
        test_set_norm = (test_set[:, 2:] - mean) / std

        knn = KNeighborsClassifier(n_neighbors=1)
        knn.fit(train_set_norm, train_set[:, 1].astype("int64"))
        neighbors_dist, neighbors_idx = knn.kneighbors(test_set_norm)

        test_targets = test_set[:, 1]

        colorDict = createColorDict()

        for i in range(len(test_set)):
            plt.plot(i, neighbors_dist[i], "o", c=colorDict[int((test_targets[i])+1)*2])

    plt.show()


    print("")


def MaxMinDist(dist):
    maxx = 0
    minn = 999
    for i in range(len(test_set)):
        if maxx < max(dist[i, :]):
            maxx = max(dist[i, :])
        if minn > min(dist[i, :]):
            minn = min(dist[i, :])
    print("Max distance")
    print(maxx)
    print("Min distance")
    print(minn)


def closeByTracks(dist):
    min_dist = []  # Get a sorted list of all indexes. Can spot if 2 or more songs are closest to the same training track.
    for i in range(len(test_set)):
        min_dist = np.append(min_dist, np.argmin(dist[i, :]))

    print(np.sort(min_dist))


if __name__ == '__main__':
    knn_neighbor_count()
    distance_measure()
    train_set, test_set = get_test_train_sets("features_targets/afe_feat_and_targ.txt")

    dist = []
    for i in range(len(test_set)):
        for k in range(len(train_set)):
            dist = np.append(dist, euclidean_dist(test_set[i, 2:], train_set[k, 2:]))

    dist = dist.reshape(len(test_set),len(train_set))

    MaxMinDist(dist)
    averageDist(dist)
    histogramish(dist)





