from utils import *
import numpy as np
import matplotlib.pyplot as plt


def euclidean_dist(v1, v2):
    return np.linalg.norm(v1 - v2)


if __name__ == '__main__':
    train_set, test_set = get_test_train_sets("features_targets/afe_feat_and_targ.txt")

    dist = []
    for i in range(len(test_set)):
        for k in range(len(train_set)):
            dist = np.append(dist, euclidean_dist(test_set[i,2:],train_set[k,2:]))

    dist = dist.reshape(len(test_set),len(train_set))
    print(dist.shape)

    min_dist = [] #Get a sorted list of all indexes. Can spot if 2 or more songs are closest to the same training track.
    for i in range(len(test_set)):
        min_dist = np.append(min_dist, np.argmin(dist[i,:]))
    
    print(np.sort(min_dist)) 
    
