from utils import *

def knn_distance_measure():
    partition_num = 0
    seed = 1
    colorDict = createColorDict()
    for partition_num in range(10):
        train_set, test_set = get_test_train_sets("features_targets/afe_feat_and_targ.txt", partition_num=partition_num, seed=seed)

        # Normalise the data
        train_set_norm, mean, std = normalise(train_set[:, 2:])
        test_set_norm = (test_set[:, 2:] - mean) / std

        dist_norm = np.zeros((len(test_set_norm), len(train_set_norm)))

        dist_threshold = 1.5
        min_num_neghbors = 3

        for i in range(len(test_set_norm)):
            for k in range(len(train_set_norm)):
                dist_norm[i, k] = euclidean_dist(test_set_norm[i, :], train_set_norm[k, :])

        indx_test, indx_train = np.where(dist_norm < dist_threshold)

        unique_test, counted_test = np.unique(indx_test, return_counts=True)
        unique_train, counted_train = np.unique(indx_train, return_counts=True)

        # Pick the tracks in the training set that are close to at least 4 other tracks
        # The indices return will map to the index in unique_train, which maps to the song
        indx_frequent_train = np.where(counted_train >= min_num_neghbors)
        track_idx_train = unique_train[indx_frequent_train]
        counted_tracks = counted_train[indx_frequent_train]
        train_tracks_close = train_set[track_idx_train, :]
        train_tracks_close_norm = train_set_norm[track_idx_train, :]

        # Find the test tracks that are close to the train tracks with many close train
        test_sets_close = []
        test_sets_close_norm = []
        for i in track_idx_train:
            idx_close_train = np.where(indx_train == i)
            idx_tets_close = indx_test[idx_close_train]
            test_sets_close.append([test_set[idx_tets_close, :]])
            test_sets_close_norm.append([test_set_norm[idx_tets_close, :]])

        # Save to file
        save_to_file = False
        verbose = True

        if save_to_file:
            folder = "analysis_docs"
            if verbose:
                filename = "verbose_max_distance_"
            else:
                filename = "max_difference_"
            filename += ">"+str(min_num_neghbors)+"_"
            createCsv(folder+"/"+filename+str(dist_threshold)+
                    "_part_nr_"+str(partition_num)+"("+str(seed)+").csv",
                    train_tracks_close, test_sets_close, [seed, partition_num], verbose)

        # Plot the mean distances to the test tracks for each train track using normalised data
        """
        for idx, train_track in enumerate(train_tracks_close_norm):
            dist_diff = np.zeros(train_track.shape)
            for test_track in test_sets_close_norm[idx][0]:
                dist_diff += abs(train_track - test_track)
            dist_diff /= len(test_sets_close[idx][0])
            plt.plot([int(x)+1 for x in range(len(dist_diff))], dist_diff, c=colorDict[idx+1])
            plt.plot([int(x)+1 for x in range(len(dist_diff))], dist_diff, "o", c=colorDict[idx+1])

        plt.show()
        """


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


    pass


def save_track_features_to_file():

    track_pairs_similar = [[57, 66], [53, 75], [108, 137], [123, 165], [126, 160], [612, 177], [748, 767],
                           [28, 86], [158, 147], [204, 746], [346, 360], [355, 466], [419, 61], [701, 204]]

    features = get_songs_feature_set("features_targets/afe_feat_and_targ.txt")
    norm_features = normalise(features[:, 2:])[0]
    all_norm_features = np.zeros(features.shape)
    all_norm_features[:, 0:2] = features[:, 0:2]
    all_norm_features[:, 2:] = norm_features

    # Save to file
    with open("similar_tracks_norm.csv", "w") as file:
        for track_pair in track_pairs_similar:
            file.write(",".join(list(map(lambda x: str(x), all_norm_features[track_pair[0], :]))))
            file.write("\n")
            file.write(",".join(list(map(lambda x: str(x), all_norm_features[track_pair[1], :]))))
            file.write("\n\n\n")


def compare_popular_song_neighbors():
    features = get_songs_feature_set("features_targets/afe_feat_and_targ.txt")
    nearest_neighbors = get_nearest_neighbors()

    idx = 296 # The popular song index

    neighbors = np.where(nearest_neighbors[:, 0] == idx)[0]

    pop_song_features = features[idx:idx+1, :]
    neighbor_features = features[neighbors, :]

    neighbor_batch = np.concatenate((features[idx:idx+1, :], features[neighbors, :]))

    # Save the neighbors to file
    pass

def find_k_nearest_neighbors():
    """
    Find the k nearest neighbors for each sample
    :return:
    """

    k = 3
    features = get_songs_feature_set("features_targets/afe_feat_and_targ.txt")
    alldist, _ = read_stored_data("features_targets/AllDistances.txt")
    alldist = alldist[:, 2:]

    nearest_neighbors = np.zeros((1000, 2*k))

    for idx, distances in enumerate(alldist):
        neighbor_idxs = np.argpartition(distances, k+1)[:k+1]  # The smallest will be to the elemnet itself, compensating

        k_near_neighbors = np.concatenate((neighbor_idxs.reshape(-1, 1), np.take(distances, neighbor_idxs).reshape(-1, 1)), axis=1)
        sorted_neighbors = k_near_neighbors[k_near_neighbors[:, 1].argsort()]
        neighbor_idxs = sorted_neighbors[1:, 0]

        for nidx, neighbor in enumerate(neighbor_idxs):
            nidx *= 2
            nearest_neighbors[idx, nidx] = int(features[int(neighbor), 0])
            nearest_neighbors[idx, nidx+1] = distances[int(neighbor)]

    # Save to file
    filename = "analysis_docs/"+str(k)+"-nearest_neighbor_dist.csv"
    with open(filename, "w") as file:
        file.write("Track,,Nearest Neighbor,,Distance")
        file.write("\n")
        file.write("Track nr," + "Class," + "Track nr," + "Class,"+"Distance")
        file.write("\n")
        for idx, track in enumerate(features):
            file.write(str(int(track[0])) + "," + str(int(track[1])) + " (" + get_label(
                int(track[1])) + ")")
            nearest_neighbor = nearest_neighbors[idx, :]
            for nidx in range(0, len(nearest_neighbor), 2):
                neighbor = int(features[int(nearest_neighbor[nidx]), 0])
                file.write("," + str(int(features[neighbor, :][0])) + "," + str(
                    int(features[neighbor, :][1])) + " (" + get_label(
                    int(features[neighbor, :][1])) + "),")
                file.write(str(nearest_neighbor[nidx+1]))
            file.write("\n")

    pass
