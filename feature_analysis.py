from utils import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.mlab as mlab
from sklearn.neighbors import KNeighborsClassifier
import networkx as nx


def euclidean_dist(v1, v2):
    return np.linalg.norm(v1 - v2)

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



def get_nearest_neighbors():
    nearest_neghbors = np.zeros((1000, 2))

    features = get_songs_feature_set("features_targets/afe_feat_and_targ.txt")

    allDist, a = read_stored_data('features_targets/AllDistances.txt')
    allDist = np.array(allDist)[:, 2:]

    for idx, distances in enumerate(allDist):
        neighbor_dist = min([dist for dist in distances if dist > 0])
        neighbor_index = np.argwhere(distances ==neighbor_dist)
        nearest_neghbors[int(features[idx, 0]), 0] = int(features[neighbor_index, 0])
        nearest_neghbors[int(features[idx, 0]), 1] = neighbor_dist

    return nearest_neghbors


def view_wrongly_classified():
    features = get_songs_feature_set("features_targets/afe_feat_and_targ.txt")
    nearest_neighbors = get_nearest_neighbors()


    i = 1
    for idx, nearest_neghbor in enumerate(nearest_neighbors[:, 0]):
        sample = features[idx, :]
        neighbor = features[int(nearest_neghbor), :]
        if sample[1] != neighbor[1]:
            diff = abs(sample - neighbor)[2:]
            pic = np.append(diff, np.mean(diff)).reshape(4, 5)
            plt.subplot(20, 20, i)
            plt.imshow(pic, cmap="gray")
            plt.axis("off")
            i += 1
    plt.show()


def plot_features():
    """Plots the feature values for each sample"""
    colors = createColorDict()
    features = get_songs_feature_set("features_targets/afe_feat_and_targ.txt")
    norm_features = normalise(features[:, 2:])[0]

    sub = 1
    for i, feature_vector in enumerate(norm_features):
        if i%100 == 0:
            plt.subplot(5, 2, sub)
            plt.title(get_label(features[i, 1]))
            sub += 1
        for f, feature in enumerate(feature_vector):
            plt.plot(i, feature, "o", c=colors[f+1])

        #if i%100 == 0:
        #    # Draw a line between the classes
        #    plt.plot([i-0.5]*20, [y for y in range(20)], c="r")

    plt.show()

    # Describe the colors
    patches = []
    for g in range(19):
        patch = mpatches.Patch(color=colors[g+1], label="Feature"+str(g+1))
        patches.append(patch)

    plt.legend(handles=patches)
    plt.show()


def knn_distance_measure_correct():
    """Measure the distance between one sample and all other samples"""

    # Computes a vector of nearest neighbors
    nearest_neghbors = np.zeros((1000, 2))

    features = get_songs_feature_set("features_targets/afe_feat_and_targ.txt")

    allDist, a = read_stored_data('features_targets/AllDistances.txt')
    allDist = np.array(allDist)[:, 2:]

    for idx, distances in enumerate(allDist):
        neighbor_dist = min([dist for dist in distances if dist > 0])
        neighbor_index = np.argwhere(distances ==neighbor_dist)
        nearest_neghbors[int(features[idx, 0]), 0] = int(features[neighbor_index, 0])
        nearest_neghbors[int(features[idx, 0]), 1] = neighbor_dist

    # Save to file
    #filename = "analysis_docs/nearest_neighbor_dist.csv"
    #with open(filename, "w") as file:
    #    file.write("Track,,Nearest Neighbor,,Distance")
    #    file.write("\n")
    #    file.write("Track nr," + "Class," + "Track nr," + "Class,"+"Distance")
    #    file.write("\n")
    #    for idx, track in enumerate(features):
    #        file.write(str(int(track[0])) + "," + str(int(track[1])) + "(" + get_label(
    #            int(track[1])) + ")")
    #        neighbor = int(nearest_neghbor[idx, 0])
    #        file.write("," + str(int(features[neighbor, :][0])) + "," + str(
    #            int(features[neighbor, :][1])) + "(" + get_label(
    #            int(features[neighbor, :][1])) + "),")
    #        file.write(str(nearest_neghbor[idx, 1]))
    #        file.write("\n")
    #    file.write("\n")


    # Divide the into two sets with correct and incorrect tracks
    correct_set = []
    incorrect_set = []
    for idx, nearest_neghbor in enumerate(nearest_neghbors[:, 0]):
        if features[idx, 1] == features[int(nearest_neghbor), 1]:
            correct_set.append(idx)
        else:
            incorrect_set.append(idx)

    correct_features = features[correct_set, :]
    incorrect_features = features[incorrect_set, :]

    write_features_to_file(correct_features, "all_correct_features.txt")
    write_features_to_file(incorrect_features, "all_incorrect_features.txt")

    # Find of some sample occures more often than others
    unique_samples, sample_count = np.unique(nearest_neghbors[:, 0], return_counts=True)

    max_sample_idx = np.argmax(sample_count)
    max_sample_song_idx = int(unique_samples[max_sample_idx])

    popular_track = features[max_sample_song_idx, :]

    # Find the samples that are close to 6 other neigbors
    popular_tracks_6 = get_popular_tracks(features, unique_samples, sample_count, 6)
    popular_tracks_5 = get_popular_tracks(features, unique_samples, sample_count, 5)
    popular_tracks_4 = get_popular_tracks(features, unique_samples, sample_count, 4)

    plot_track_distances(284, allDist[284, :], features)

    pass


def plot_track_distances(track_id, distance_vector, features):
    """Plots the distances to all the neighbors of specified track. The different classes are colorcoded"""
    colors = createColorDict()

    for i, distance in enumerate(distance_vector):
        plt.plot(i, distance, "o", c=colors[int(features[i, 1])+1])

    # Describe the colors
    patches = []
    for i in range(10):
        patch = mpatches.Patch(color=colors[i+1], label=get_label(i))
        patches.append(patch)

    plt.legend(handles=patches)

    plt.title("Neighbor distances for "+str(track_id))
    plt.show()

def plot_all_track_dist_to_origo():
    colors = createColorDict()
    features = get_songs_feature_set("features_targets/afe_feat_and_targ.txt")
    norm_features = normalise(features[:, 2:])[0]
    origo = np.zeros(19)

    for i, feature in enumerate(norm_features):
        dist_to_origo = euclidean_dist(origo, feature)
        plt.plot(i, dist_to_origo, "o", c=colors[int(features[i, 1]) + 1])

    patches = []
    for i in range(10):
        patch = mpatches.Patch(color=colors[i + 1], label=get_label(i))
        patches.append(patch)

    plt.legend(handles=patches)
    plt.title("Track distances to origo")
    plt.show()


def get_popular_tracks(features, unique_samples, sample_count, num_close):
    """
    Find the tracks that are close to the specified number of tracks
    :param features: feature set
    :param unique_samples: the unique samples in the nearest_neighbors
    :param sample_count: the number of times the track is neighbor to other samples
    :param num_close: the number of tracks the track should have to get selected
    :return: The tracks that are close ot num_close other tracks
    """
    num_neighbor = np.where(sample_count == num_close)
    popular_tracks_idx = unique_samples[num_neighbor]
    popular_tracks = np.zeros((len(popular_tracks_idx), 21))
    for idx, track_idx in enumerate(popular_tracks_idx):
        popular_tracks[int(idx), :] = features[int(track_idx), :]

    return popular_tracks


def histogramish(dist):
    #the histogram of the data
    n, bins, patches = plt.hist(dist[740, :], bins=100, facecolor='green')

    plt.axis([0, 15, 0, 70])

    plt.show()


def averageDist(dist):
    avg_dist = []
    for i in range(len(test_set)):
        avg_dist = np.append(avg_dist, np.average(dist[i, :]))

    plt.hist(avg_dist, bins=20)
    plt.show()

def averageHist(dist,bucket_size=100):
    avg_hist = []
    maxx, minn = MaxMinDist(dist)
    nr_buck = bucket_size
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
    
    # print(number_of_weights)
    # print(sum(number_of_weights))
    return number_of_weights


def MaxMinDist(dist):
    maxx = 0
    minn = 999
    for i in range(len(dist[:,0])):
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
def classDistance(alldist):
    a = []
    classes = [7,6,3,0,8,1,9,4,2,5]
    for i in range(1,11,1):
        a = np.append(a,np.average(alldist[(i-1)*100:i*100,:]))
    
    adict = dict(zip(classes, a))

    return a, adict

def classHistograms(alldist):
    a = 0
    classes = [7,6,3,0,8,1,9,4,2,5]
  
    for i in range(10):
        print(a)
        plt.subplot(2,5,a+1)
        plt.ylim(0,6000)
        plt.ylabel('Number of tracks')
        plt.xlabel('Bucket number')
        plt.plot(averageHist(alldist[0+a*100:100+a*100,:]))
        plt.title(get_label(classes[a]))
        a += 1


    plt.show()
def classInternalDistance(alldist):
    classes = [7,6,3,0,8,1,9,4,2,5]
    a = 0

    for i in range(10):
        a = classes[i]
        plt.subplot(2,5,i+1)
        plt.ylim(0,2000)
        plt.ylabel('Number of tracks')
        plt.xlabel('Bucket number')
        plt.plot(averageHist(alldist[0+a*100:100+a*100,0+a*100:100+a*100],30)) #KAN VARA FEL, KOLLA OM CLASSES TEKNIKEN INTE ÄR BONKERS
        plt.title(get_label(classes[a]))

    plt.show()

def allCorrectPlotDist(alldist):
    allCorrect, a = read_stored_data('features_targets/all_correct_features.txt')

    allCorrectDist = []
    start = 0
    for i in range(allCorrect.shape[0]):
        for k in range(start, alldist.shape[0],1):
            if allCorrect[i,0] == alldist[k,1]:
                # print('WOp')
                allCorrectDist = np.append(allCorrectDist, alldist[k,:])
                start = int(allCorrect[i,0])

                break
    allCorrectDist = allCorrectDist.reshape(allCorrect.shape[0],alldist.shape[1])
    return allCorrectDist

def allInCorrectPlotDist(alldist):
    allInCorrect, a = read_stored_data('features_targets/all_incorrect_features.txt')

    allInCorrectDist = []
    start = 0
    for i in range(allInCorrect.shape[0]):
        for k in range(start,alldist.shape[0],1):
            if allInCorrect[i,0] == alldist[k,1]:
                # print('WOp')
                allInCorrectDist = np.append(allInCorrectDist, alldist[k,:])
                start = int(allInCorrect[i,0])
                break
    allInCorrectDist = allInCorrectDist.reshape(allInCorrect.shape[0],alldist.shape[1])
    return allInCorrectDist

def correct_incorrectDistPlot(allCorrect, allInCorrect):
    plt.subplot(1,2,1)
    plt.plot(averageHist(allCorrect[:,2:]))
    plt.ylim(0,)
    plt.xlim(0,)
    plt.ylabel('Number of tracks')
    plt.xlabel('Bucket number')
    plt.title('All Correct Dist')

    plt.subplot(1,2,2)
    plt.plot(averageHist(allInCorrect[:,2:]))
    plt.ylim(0,)
    plt.xlim(0,)
    plt.ylabel('Number of tracks')
    plt.xlabel('Bucket number')
    plt.title('All Incorrect Dist')

    plt.show()

def create_neighbor_graph():
    nearest_neighbors = get_nearest_neighbors()
    features = get_songs_feature_set("features_targets/afe_feat_and_targ.txt")
    G = nx.DiGraph()
    colors = createColorDict()
    for idx, neighbor_i in enumerate(nearest_neighbors[:, 0]):
        if idx not in G:
            G.add_node(idx, color=colors[int(features[idx, 1])+1])
        if int(neighbor_i) not in G:
            G.add_node(int(neighbor_i), color=colors[int(features[int(neighbor_i), 1])+1])
        G.add_edge(idx, int(neighbor_i))

    G = nx.drawing.nx_agraph.to_agraph(G)
    G.node_attr.update(style="filled")
    # G.draw('analysis_docs/knn_v2.png', format='png', prog='neato')

def remove_diagonal(alldist):
    for i in range(alldist.shape[0]):
        alldist[i,i+2] = 999
    return alldist
    
def get_nearest_neighbors_dist(alldist):
    nearest = alldist[:,1:3]

    for i in range(alldist.shape[0]):
        nearest[i,1] = np.argmin(alldist[i,2:])

    return nearest

def get_nearest_correct_neighbors(alldist):
    nearest = alldist[:,1:3]

    a = 0
    for i in range(alldist.shape[0]):
        if((i%100 == 0) and not(i == 0)):
            a += 1

        nearest[i,1] = np.argmin(alldist[i,2+(a*100):102+(a*100)])+a*100

    return nearest

def get_both_nearest_and_correct_neighbors(alldist):
    nearest = alldist[:,1:4]

    for i in range(alldist.shape[0]):
        nearest[i,1] = np.argmin(alldist[i,2:])

    a = 0
    for i in range(alldist.shape[0]):
        if((i%100 == 0) and not(i == 0)):
            a += 1

        nearest[i,2] = np.argmin(alldist[i,2+(a*100):102+(a*100)])+a*100

    return nearest

if __name__ == '__main__':
    #knn_neighbor_count()
    #save_track_features_to_file()
    #plot_all_track_dist_to_origo()
    #knn_distance_measure_correct()
    #view_wrongly_classified()
    # plot_features()
    #compare_popular_song_neighbors()
    #train_set, test_set = get_test_train_sets("features_targets/afe_feat_and_targ.txt",0,42)
    #train_setP, test_setP = partdata()
    # create_neighbor_graph()

    train_set, test_set = get_test_train_sets("features_targets/afe_feat_and_targ.txt",0,42)
    train_setP, test_setP = partdata()


    allDist, a = read_stored_data('features_targets/Alldistances.txt')
    allDist = np.array(allDist)

    alldistNoDiag = remove_diagonal(allDist)
    nearest = get_nearest_neighbors_dist(alldistNoDiag)
    nearest_correct = get_nearest_correct_neighbors(alldistNoDiag)
    both = get_both_nearest_and_correct_neighbors(alldistNoDiag)

    for i in range(1000):
        print(both[i,:])

    # nearesttracks = np.concatenate((nearest,nearest_correct[:,1]),axis=0) #Får inte skiten att funka. 



    
    # a = allCorrectPlotDist(allDist)
    # b = allInCorrectPlotDist(allDist)
    # CorrectAvg = np.average(a)
    # InCorrectAvg = np.average(b)
    # print(CorrectAvg)
    # print(InCorrectAvg)
    # correct_incorrectDistPlot(a,b)
   
    # a = allCorrectPlotDist(allDist)
    # b = allInCorrectPlotDist(allDist)
    # correct_incorrectDistPlot(a,b)

    # classInternalDistance(allDist[:,2:])
    # classHistograms(allDist[:,2:])
    # values, ClassDict = classDistance(allDist)

    # print(allDist[:,0:2])
    # write_features_to_file(allDist, 'AllDistances.txt')
 

    # train_set_norm, mean, std = normalise(train_set[:, 2:])
    # test_set_norm = (test_set[:, 2:] - mean) / std
    # train_setP_norm, mean, std = normalise(train_setP[:, :])
    # test_setP_norm = (test_setP[:, :] - mean) / std

    # dist = distfunc(train_set_norm, test_set_norm,0)
    # distP = distfunc(train_setP_norm, test_setP_norm,0)


    # # Plottar ett histogram för en test sample till all tränings_samples. Random
    # histogramish(allDist)


    # # Plottar 2 kurvor som representerar alla test samples distanser till träning. Random vs Fault
    # X = averageHist(dist, test_set_norm)
    # Y = averageHist(distP, test_setP_norm)
    # plt.subplot(1,2,1)
    # plt.plot(averageHist(allDist[:,2:]))
    # plt.subplot(1,2,2)
    # plt.plot(Y)
    # plt.show()


