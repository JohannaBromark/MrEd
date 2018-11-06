from utils import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab


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
    train_set, test_set = get_test_train_sets("features_targets/afe_feat_and_targ.txt")

    # Normalise the data
    train_set_norm, mean, std = normalise(train_set[:, 2:])
    test_set_norm = (test_set[:, 2:] - mean) / std

    dist_norm = np.zeros((len(test_set_norm), len(train_set_norm)))

    dist_threshold = 1.0

    for i in range(len(test_set_norm)):
        for k in range(len(train_set_norm)):
            dist_norm[i, k] = euclidean_dist(test_set_norm[i, :], train_set_norm[k, :])

    something = np.where(dist_norm < dist_threshold)
    print("help")


if __name__ == '__main__':
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

def MaxMinDist(dist):

    maxx = 0
    minn = 999
    for i in range(len(test_set)):
        if maxx < max(dist[i,:]):
            maxx = max(dist[i,:])
        if minn > min(dist[i,:]):
            minn = min(dist[i,:])
    print("Max distance")
    print(maxx)
    print("Min distance")
    print(minn)


def closeByTracks(dist):

    min_dist = [] #Get a sorted list of all indexes. Can spot if 2 or more songs are closest to the same training track.
    for i in range(len(test_set)):
        min_dist = np.append(min_dist, np.argmin(dist[i,:]))
    
    print(np.sort(min_dist)) 




