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

def averageHist(dist, test_set):
    avg_hist = []
    maxx, minn = MaxMinDist(dist, test_set)
    nr_buck = 100
    bucket_size = maxx / nr_buck
    bucket = 0
    counter = 0
    number_of_weights = []
    dist2 = np.array(dist)
    # print(dist2.shape)
    for a in range(1,nr_buck,1):
        # print(a)
        for i in range(dist2.shape[0]):
            for k in range(dist2.shape[1]):
                if((dist2[i,k] > bucket) and (dist2[i,k] < (bucket_size*a))):
                    counter = counter +1

        number_of_weights = np.append(number_of_weights, counter)
        bucket = (bucket_size*a)
        counter = 0
    
    print(number_of_weights)
    print(sum(number_of_weights))
    return number_of_weights




def MaxMinDist(dist, test_set):
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
    return maxx, minn


def closeByTracks(dist):
    min_dist = []  # Get a sorted list of all indexes. Can spot if 2 or more songs are closest to the same training track.
    for i in range(len(test_set)):
        min_dist = np.append(min_dist, np.argmin(dist[i, :]))

    print(np.sort(min_dist))


def distance_measure():
    partition_num = 6
    seed = 1
    colorDict = createColorDict()
    for partition_num in range(10):
        train_set, test_set = get_test_train_sets("features_targets/afe_feat_and_targ.txt", partition_num=partition_num, seed=seed)

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

        # Find the test tracks that are close to the train tracks with many close train
        test_sets_close = []
        for i in track_idx_train:
            idx_close_train = np.where(indx_train == i)
            idx_tets_close = indx_test[idx_close_train]
            test_sets_close.append([test_set[idx_tets_close, :]])

        # Save to file
        save_to_file = False
        verbose = True

        if save_to_file:
            folder = "analysis_docs"
            if verbose:
                filename = "verbose_max_distance_"
            else:
                filename = "max_difference"

            createCsv(folder+"/"+filename+str(dist_threshold)+
                    "_part_nr_"+str(partition_num)+"("+str(seed)+").csv",
                    train_tracks_close, test_sets_close, [seed, partition_num], verbose)

        # Plot the mean distances to the test tracks for each train track
        for idx, train_track in enumerate(train_tracks_close):
            dist_diff = np.zeros(train_track[2:].shape)
            for test_track in test_sets_close[idx][0]:
                dist_diff += abs(train_track[2:] - test_track[2:])
            dist_diff /= len(test_sets_close[idx][0])
            plt.plot([int(x)+1 for x in range(len(dist_diff))], dist_diff, c=colorDict[idx+1])
            plt.plot([int(x)+1 for x in range(len(dist_diff))], dist_diff, "o", c=colorDict[idx+1])

        plt.show()

    print("help")


def createCsv(filename, train_tracks, corresponding_test_tracks, settings, verbose = False):
    with open(filename, "w") as file:
        file.write("Random seed "+str(settings[0])+","+"Test set partition "+str(settings[1])+"\n")
        file.write("Train track nr,"+ "Class train,"+"Test track nr,"+"Class test")
        if verbose:
            file.write(",Feature distances from train track to each test track")
        file.write("\n")
        for idx, train_track in enumerate(train_tracks):
            file.write(str(int(train_track[0]))+","+str(int(train_track[1]))+"("+get_label(int(train_track[1]))+")")
            if verbose:
                file.write(","+","+","+",".join(list(map(lambda x: str(x), train_track[2:]))))
            file.write("\n")
            for test_track in corresponding_test_tracks[idx][0]:
                file.write(","+","+str(int(test_track[0]))+","+str(int(test_track[1]))+"("+get_label(int(test_track[1]))+"),")
                if verbose:
                    file.write(",".join(list(map(lambda x: str(x), test_track[2:]))))
                    file.write(","+","+",".join(list(map(lambda x: str(x), abs(train_track[2:] - test_track[2:])))))
                file.write("\n")
            file.write("\n")



def knn_neighbor_count():
    # Checks the distance for each partition as testset and plots the result

    colorDict = createColorDict()

    for i in range(10):
        train_set, test_set = get_test_train_sets("features_targets/afe_feat_and_targ.txt", partition_num=i, seed=1)

        # Normalise the data
        train_set_norm, mean, std = normalise(train_set[:, 2:])
        test_set_norm = (test_set[:, 2:] - mean) / std

        knn = KNeighborsClassifier(n_neighbors=1)
        knn.fit(train_set_norm, train_set[:, 1].astype("int64"))
        neighbors_dist, neighbors_idx = knn.kneighbors(test_set_norm)

        test_targets = test_set[:, 1]


        # Creates a plot with all samples in the same plot
        #for j in range(len(test_set)):
        #    plt.plot(j, neighbors_dist[j], "o", c=colorDict[int((test_targets[i])+1)*2])



    print("")

def partdata():
    train_samples, train_targets = read_stored_data('features_targets/afe_feat_and_tarF.txt')

    test_samples, test_targets = read_stored_data('features_targets/afe_feat_and_tarFT.txt')

    vali_samples, vali_targets = read_stored_data('features_targets/afe_feat_and_tarFV.txt')


    train_samples = mean_by_song(train_samples)
    train_targets = train_samples[:,1]
    train_samples = train_samples[:,2:]

    vali_samples = mean_by_song(vali_samples)
    vali_samples = vali_samples[:,2:]
    vali_targets = vali_samples[:,1]

    test_samples = mean_by_song(test_samples)
    test_targets = test_samples[:,1]
    test_samples = test_samples[:,2:]


    train_samples = np.concatenate([train_samples,vali_samples],0)
    train_targets = np.concatenate([train_targets,vali_targets],0)

    test_targets = np.array(test_targets)
    train_targets = np.array(train_targets)

    return train_samples, test_samples

def distfunc(train_set, test_set, remove = 2):
    dist = []

    for i in range(len(test_set)):
        for k in range(len(train_set)):
            dist = np.append(dist, euclidean_dist(test_set[i,remove:], train_set[k,remove:]))

    return dist.reshape(len(test_set),len(train_set))

def allDistance(train_set,test_set):
    train_set = np.concatenate((train_set,test_set), axis=0)
    train_set = (train_set[train_set[:,0].argsort()])
    print(train_set[:,1])
    dist = []
    targets = train_set[:,1]
    train_set, mean, std = normalise(train_set[:, 2:])
    print(len(train_set))

    for i in range(len(train_set)):
        print(i)
        dist = np.append(dist, targets[i])
        dist = np.append(dist, i)

        for k in range(len(train_set)):
            
            dist = np.append(dist, euclidean_dist(train_set[i,:], train_set[k,:]))

    return dist.reshape(len(train_set),len(train_set)+2)
    
  
    




if __name__ == '__main__':
    #knn_neighbor_count()
    # distance_measure()
    
    train_set, test_set = get_test_train_sets("features_targets/afe_feat_and_targ.txt",0,42)
    train_setP, test_setP = partdata()

    
    allDist, a = read_stored_data('features_targets/Alldistances.txt')
    allDist = np.array(allDist)
    print(allDist[:,0:2])
    # write_features_to_file(allDist, 'AllDistances.txt')


    # train_set_norm, mean, std = normalise(train_set[:, 2:])
    # test_set_norm = (test_set[:, 2:] - mean) / std
    # train_setP_norm, mean, std = normalise(train_setP[:, :])
    # test_setP_norm = (test_setP[:, :] - mean) / std

    # dist = distfunc(train_set_norm, test_set_norm,0)
    # distP = distfunc(train_setP_norm, test_setP_norm,0)
  
    
    # # Plottar ett histogram för en test sample till all tränings_samples. Random
    # histogramish(dist)


    # # Plottar 2 kurvor som representerar alla test samples distanser till träning. Random vs Fault
    # X = averageHist(dist, test_set_norm)
    # Y = averageHist(distP, test_setP_norm)
    # plt.subplot(1,2,1)
    # plt.plot(X)
    # plt.subplot(1,2,2)
    # plt.plot(Y)
    # plt.show()


