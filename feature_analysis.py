from utils import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.mlab as mlab
import networkx as nx
import sys
import re
# import pygraphviz


########################## Save stuff to file ##################################


def save_neighbors_to_csv(filename, neighbors, neighbor_distances):
	features = get_songs_feature_set("features_targets/all_vectors.txt")

	# Save to file
	with open(filename, "w") as file:
		file.write("Track,,Nearest Neighbor,,Distance")
		file.write("\n")
		file.write("Track nr," + "Class," + "Track nr," + "Class,"+"Distance")
		file.write("\n")
		for idx, track_neighbors in enumerate(neighbors):
			file.write(str(int(neighbor_distances[idx, 0])) + "," + str(int(neighbor_distances[idx, 1])) + " (" + get_label(
				int(neighbor_distances[idx, 1])) + ")")
			for nidx in track_neighbors:
				neighbor = int(features[int(nidx), 0])
				file.write("," + str(int(features[neighbor, 0])) + "," + str(
					int(features[neighbor, 1])) + " (" + get_label(
					int(features[neighbor, 1])) + "),")
				file.write(str(neighbor_distances[idx, int(nidx)+2]))
			file.write("\n")


def save_angle_neighbors_to_file():

	angles = read_stored_data("features_targets/all_angles.txt")
	distance_matrix = get_k_nearest_neighbors(angles[:, 2:], 1)

	save_neighbors_to_csv("analysis_docs/nearest_neighbor_angle.csv",
						  distance_matrix,
						  angles)


def compute_angles():
	"""Compute the angles between each feature and save to file"""

	features = get_songs_feature_set("features_targets/all_vectors.txt")
	only_mfcc = True
	if only_mfcc:
		mfcc_filter = [0,1,10,11,12,13,14,15,16,17,18,19]
		features = features[:, mfcc_filter]
	angles = np.zeros((len(features), len(features) +2))
	norm_features = normalise(features[:, 2:])[0]
	angles[:, :2] = features[:, :2]

	for idx, feature in enumerate(norm_features):
		for o_idx, other_feature in enumerate(norm_features):
			angles[idx, o_idx+2] = angle(feature, other_feature)

	save_matrix("features_targets/all_angles_mfcc.txt", angles)

def compute_distances():
	"""Compute the distance between each feature and save to file"""
	only_mfcc = True
	features = get_songs_feature_set("features_targets/all_vectors.txt")
	if only_mfcc:
		mfcc_filter = [0,1,10,11,12,13,14,15,16,17,18,19]
		features = features[:, mfcc_filter]
	distances = np.zeros((len(features), len(features) + 2))
	norm_features = normalise(features[:, 2:])[0]
	distances[:, :2] = features[:, :2]
	for idx, feature in enumerate(norm_features):
		for o_idx, other_feature in enumerate(norm_features):
			distances[idx, o_idx + 2] = euclidean_dist(feature, other_feature)
	save_matrix("features_targets/all_distances_mfcc.txt", distances)


########################### Neighbor computations ###############################

def get_k_nearest_neighbors(matrix, k):
	"""
	Computes the k-nearest neighbor from the matrix.
	The matrix should not include track number and label!
	"""

	nearest_neghbors = np.zeros((1000, k))

	for idx, distances in enumerate(matrix):
		neighbor_idxs_temp = np.argpartition(distances, k + 1)[:k + 1]

		# Remove itself from the neighbor list
		neighbor_idxs = np.delete(neighbor_idxs_temp, np.argwhere(neighbor_idxs_temp == idx))

		# Sort neighbors based on distance
		k_near_neighbors = np.concatenate(
			(neighbor_idxs.reshape(-1, 1), np.take(distances, neighbor_idxs).reshape(-1, 1)), axis=1)
		sorted_neighbors = k_near_neighbors[k_near_neighbors[:, 1].argsort()]
		nearest_neghbors[idx, :] = sorted_neighbors[:, 0]

	return nearest_neghbors


def get_nearest_neighbors():
	nearest_neghbors = np.zeros((1000, 2))

	features = get_songs_feature_set("features_targets/all_vectors.txt")

	allDist = read_stored_data('features_targets/AllDistances.txt')
	allDist = np.array(allDist)[:, 2:]

	for idx, distances in enumerate(allDist):
		neighbor_dist = min([dist for dist in distances if dist > 0])
		neighbor_index = np.argwhere(distances ==neighbor_dist)
		nearest_neghbors[int(features[idx, 0]), 0] = int(features[neighbor_index, 0])
		nearest_neghbors[int(features[idx, 0]), 1] = neighbor_dist

	return nearest_neghbors


################################# Other #############################

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
	nearest_neighbors = get_nearest_neighbors()

	# Create subplots
	#sub = 1
	#for i, feature_vector in enumerate(norm_features):
	#    if i%100 == 0:
	#        plt.subplot(5, 2, sub)
	#        plt.title(get_label(features[i, 1]))
	#        sub += 1
	#    for f, feature in enumerate(feature_vector):
	#        plt.plot(i, feature, "x", c=colors[f+1])
		#if i%100 == 0:
		#    # Draw a line between the classes
		#    plt.plot([i-0.5]*20, [y for y in range(20)], c="r")
	#plt.show()


	# Create one plot per genre
	#for g in range(10):
	#    for i, feature_vector in enumerate(norm_features[g*100:(g+1)*100, :]):
	#        for f, feature in enumerate(feature_vector):
	#            plt.plot(i, feature, "o", c=colors[f+1])
	#        if features[i, 1] != features[int(nearest_neighbors[i, 0]), 1]:
	#            max_y_value = max(feature_vector)
	#            min_y_value = min(feature_vector)
	#            y = np.linspace(min_y_value, max_y_value)
	#            plt.plot([i]*len(y), y, c="r")
	#    plt.title(get_label(int(features[g*100, 1])))
	#    plt.show()

	# Plot the feature of a wrongly classified sample and its neighbor
	for i, feature in enumerate(norm_features):
		neighbor = int(nearest_neighbors[i, 0])
		if features[i, 1] != features[neighbor, 1]:
			diff = abs(feature - norm_features[neighbor, :])
			for f in range(len(feature)):
				plt.plot(1, feature[f], "o", c=colors[f+1])
				plt.plot(1.5, norm_features[neighbor, f], "o", c=colors[f+1])
				if diff[f] < 0.1:
					plt.plot([1, 1.5], [feature[f], norm_features[neighbor, f]], c=colors[f+1])
				plt.title("Sample " + str(i) + " (" + get_label(int(features[i, 1])) + ") classified by " + str(neighbor) + " ("+get_label(int(features[neighbor, 1]))+")")
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

############################# Plots #############################

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


################################ Distance computations ##################


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
	features = get_songs_feature_set("features_targets/all_vectors.txt")
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
	#G.draw('analysis_docs/knn_v3.png', format='png', prog='neato')


def remove_diagonal(alldist):
	for i in range(alldist.shape[0]):
		alldist[i, i + 2] = 999
	return alldist


def get_nearest_neighbors_dist(alldist):
	nearest = alldist[:, 1:3]

	for i in range(alldist.shape[0]):
		nearest[i, 1] = np.argmin(alldist[i, 2:])

	return nearest


def get_nearest_correct_neighbors(alldist):
	nearest = alldist[:, 1:3]

	a = 0
	for i in range(alldist.shape[0]):
		if ((i % 100 == 0) and not (i == 0)):
			a += 1

		nearest[i, 1] = np.argmin(alldist[i, 2 + (a * 100):102 + (a * 100)]) + a * 100

	return nearest


def get_both_nearest_and_correct_neighbors(alldist):
	nearest = alldist[:, 0:4]

	for i in range(alldist.shape[0]):
		nearest[i, 2] = np.argmin(alldist[i, 2:])

	a = 0
	for i in range(alldist.shape[0]):
		if ((i % 100 == 0) and not (i == 0)):
			a += 1

		nearest[i, 3] = np.argmin(alldist[i, 2 + (a * 100):103 + (a * 100)]) + a * 100
	
	return nearest

def get_missclassified(alldistt):
	missclassified = np.array([])
	nearest = get_both_nearest_and_correct_neighbors(np.copy(alldistt))
	
	for i in range(alldistt.shape[0]):
		if(nearest[i,1] != nearest[i,2]):
			missclassified = np.append(missclassified, alldistt[i,:])
	

   
	missclassified = missclassified.reshape(int(len(missclassified)/alldistt.shape[1]),int(alldistt.shape[1]))

	return missclassified

def get_missclassified_with_neighbors_nearest_and_correct(alldistt):
	missclassified = np.array([])
	nearest = get_both_nearest_and_correct_neighbors(np.copy(alldistt))

	for i in range(alldistt.shape[0]):
		if(nearest[i,2] != nearest[i,3]):
			missclassified = np.append(missclassified, nearest[i,:])



	missclassified = missclassified.reshape(int(len(missclassified)/4),4)

	return missclassified




def compute_knn_adjecency_matrix():

	all_distances = read_stored_data("features_targets/all_distances_mfcc.txt")
	all_distances = all_distances[:, 2:]
	genre_adjecency_matrix = np.zeros((10, 10))

	for g in range(10):
		for g2 in range(10):
			tot_dist = 0
			for s in range(100):
				tot_dist += np.mean(all_distances[g*100 + s, g2*100:(g2*100)+19])
			genre_adjecency_matrix[g, g2] = tot_dist/100

	return genre_adjecency_matrix


def create_knn_graph():
	""" RESULTS IN A CRAPPY DRAWING!!! don't know why :'("""
	adjecency_matrix = compute_knn_adjecency_matrix()
	class_order = [7,6,3,0,8,1,9,4,2,5]

	dt = [('len', float)]
	tuple_distances = np.array([tuple(dist) for dist in adjecency_matrix])
	G = nx.from_numpy_matrix(tuple_distances.view(dt))

	name_mapping = dict(zip(G.nodes, [get_label(label) for label in class_order]))
	G = nx.relabel_nodes(G, name_mapping)
	G = nx.drawing.nx_agraph.to_agraph(G)
	G.node_attr.update(color="red", style="filled")

	# G.draw('analysis_docs/knn_mfcc_distances_visualized.png', format='png', prog='neato')

########################### Angles ####################################

def create_angle_neighbor_graph():
	angles = read_stored_data("features_targets/all_angles.txt")
	nearest_neighbors = get_k_nearest_neighbors(angles[:, 2:], 1)
	features = get_songs_feature_set("features_targets/all_vectors.txt")
	G = nx.DiGraph()
	colors = createColorDict()
	for idx, neighbor_i in enumerate(nearest_neighbors):
		if idx not in G:
			G.add_node(idx, color=colors[int(features[idx, 1])+1])
		if int(neighbor_i) not in G:
			G.add_node(int(neighbor_i), color=colors[int(features[int(neighbor_i), 1])+1])
		G.add_edge(idx, int(neighbor_i))

	G = nx.drawing.nx_agraph.to_agraph(G)
	G.node_attr.update(style="filled")
	G.draw('analysis_docs/knn_angle_neighbor_visualized.png', format='png', prog='neato')


def create_class_graph_angle():
	angles_data = read_stored_data("features_targets/all_angles.txt")
	features = get_songs_feature_set("features_targets/all_vectors.txt")
	angles = angles_data[:, 2:]

	# Compute the mean angle from one class to all other classes
	adjecency_matrix = np.zeros((10, 10))
	for genre in range(10):
		for neighbor in range(10):
			sub_angles = angles[genre*100:(genre+1)*100, neighbor*100: (neighbor+1)*100]
			mean = np.mean(sub_angles, axis=0)
			adjecency_matrix[genre, neighbor] = np.sum(mean)/100

	G = nx.from_numpy_matrix(adjecency_matrix*10000)
	G = nx.drawing.nx_agraph.to_agraph(G)
	G.draw('analysis_docs/knn_genre_angle_visualized.png', format='png', prog='neato')


# Find tracks that gets wrongly classified with distance and correct with angles and vice versa.
def distance_vs_angle():
	angles_data = read_stored_data("features_targets/all_angles.txt")
	distance_data = read_stored_data("features_targets/all_distances.txt")
	features = get_songs_feature_set()

	nearest_neighbor_angle = get_k_nearest_neighbors(angles_data[:, 2:], 1)
	nearest_neighbor_distance = get_k_nearest_neighbors(distance_data[:, 2:], 1)

	# Angle classifies correct, distance classifies wrong
	angle_correct = []
	# Angle classifies wrong, distance classifies correct
	distance_correct = []
	# Both angle and distance classifies wrong
	both_wrong = []
	# Both angle and distance classifies correct
	both_correct = []

	for idx, neighbors in enumerate(zip(nearest_neighbor_angle, nearest_neighbor_distance)):
		angle_neighbor = int(neighbors[0][0])
		distance_neighbor = int(neighbors[1][0])
		if angles_data[idx, 1] == features[int(nearest_neighbor_angle[idx]), 1] \
				and distance_data[idx, 1] != features[int(nearest_neighbor_distance[idx]), 1]:
			angle_correct.append(features[idx, :])

		elif angles_data[idx, 1] != features[int(nearest_neighbor_angle[idx]), 1] \
				and distance_data[idx, 1] == features[int(nearest_neighbor_distance[idx]), 1]:
			distance_correct.append(features[idx, :])

		elif angles_data[idx, 1] != features[int(nearest_neighbor_angle[idx]), 1] \
				and distance_data[idx, 1] != features[int(nearest_neighbor_distance[idx]), 1]:
			both_wrong.append(features[idx, :])

		elif angles_data[idx, 1] == features[int(nearest_neighbor_angle[idx]), 1] \
				and distance_data[idx, 1] == features[int(nearest_neighbor_distance[idx]), 1]:
			both_correct.append(features[idx, :])

	# Could save this to file as well..
	# Could count how many of each class there is in each list

	pass

def plot_missclassified_with_neighbor_by_feature(id,normalize=True):
	allDist = read_stored_data('features_targets/all_distances.txt')
	allDist = np.array(allDist)

	alldistNoDiag = remove_diagonal(np.copy(allDist))
	content = read_content('features_targets/index_of_content.txt') #TODO sätt namnet på låten i labels
	missclassified = get_missclassified_with_neighbors_nearest_and_correct(alldistNoDiag)
	features = get_songs_feature_set('features_targets/all_vectors.txt')
	if normalize == True:
		features, a, b = normalise(features)
	# print(missclassified[0,0])
	# for i in range(150):
	# 	print(features[i,:])
	
	song1 = features[int(missclassified[id,0]),2:]
	song2 = features[int(missclassified[id,2]),2:]
	song3 = features[int(missclassified[id,3]),2:]
	songlabel1 = content[int(missclassified[id,0])]
	songlabel2 = content[int(missclassified[id,2])]
	songlabel3 = content[int(missclassified[id,3])]

	
	f, ax = plt.subplots()
	ax.set_xticks([i for i in range(19)])
	x_ticks_labels = [
		'Centroid mean', 
		'Centroid var',
		'Rolloff mean', 
		'Rolloff var',
		'Flux mean', 
		'Flux var',
		'Zero-Crossing mean', 
		'Zero-Crossing var',
		'MFCC0 mean', 
		'MFCC0 var',
		'MFCC1 mean', 
		'MFCC1 var',
		'MFCC2 mean', 
		'MFCC2 var',
		'MFCC3 mean', 
		'MFCC3 var',
		'MFCC4 mean', 
		'MFCC4 var',
		'Energy'
		]

	ax.set_xticklabels(x_ticks_labels, rotation='80', fontsize=18)

# plt.plot(i, dist_to_origo, "o", c=colors[int(features[i, 1]) + 1])
	print(missclassified[id,0]) 
	print(missclassified[id,2])
	print(missclassified[id,3])

	song = plt.plot(song1,"o-", markersize=12, label=("Missclassified Song "+str(missclassified[id,0])+str(songlabel1))) #blå
	nearest = plt.plot(song2, "o--",markersize=12,label=("Nearest to song "+str(missclassified[id,2])+str(songlabel2)))#gul
	correct = plt.plot(song3, "o--",markersize=12, label=("Nearest correct to song "+str(missclassified[id,3])+str(songlabel3))) #grön
	# plt.legend(()=[song,nearest,correct],loc='upper left')
	# song = plt.plot(song1,"o-", markersize=8, label=("Missclassified song: "+str(songlabel1[2]) + ' by ' + str(songlabel1[1]))) #blå
	# nearest = plt.plot(song2, "o--",markersize=8,label=("Nearest to song: "+str(songlabel2[2]) + ' by ' + str(songlabel2[1])))#gul
	# correct = plt.plot(song3, "o--",markersize=8, label=("Nearest correct to song: "+str(songlabel3[2]) + ' by ' + str(songlabel3[1]))) #grön

	ax.legend(prop={'size': 16})
	# plt.savefig('bob_marley_misclassified.png')
	plt.show()

def plot_missclassified_with_neighbor_by_feature_mfcc(id,normalize=True):
	allDist = read_stored_data('features_targets/all_distances.txt')
	allDist = np.array(allDist)

	alldistNoDiag = remove_diagonal(np.copy(allDist))
	content = read_content('features_targets/index_of_content.txt') #TODO sätt namnet på låten i labels
	missclassified = get_missclassified_with_neighbors_nearest_and_correct(alldistNoDiag)
	features = get_songs_feature_set('features_targets/all_vectors.txt')
	if normalize == True:
		features, a, b = normalise(features)
	# print(missclassified[0,0])
	# for i in range(150):
	# 	print(features[i,:])
	for i in range(len(missclassified)):
		print("ID: "+str(i))
		print(int(missclassified[i,0]))
	
	song1 = features[int(missclassified[id,0]),10:20]
	song2 = song1-features[int(missclassified[id,2]),10:20]
	# song2 = features[int(missclassified[id,2]),10:20]
	# song3 = features[int(missclassified[id,3]),10:20]
	song3 = song1-features[int(missclassified[id,3]),10:20]
	songlabel1 = content[int(missclassified[id,0])]
	songlabel2 = content[int(missclassified[id,2])]
	songlabel3 = content[int(missclassified[id,3])]

	
	f, ax = plt.subplots()
	ax.set_xticks([i for i in range(10)])
	x_ticks_labels = [
		'MFCC0 mean', 
		'MFCC0 var',
		'MFCC1 mean', 
		'MFCC1 var',
		'MFCC2 mean', 
		'MFCC2 var',
		'MFCC3 mean', 
		'MFCC3 var',
		'MFCC4 mean', 
		'MFCC4 var',
		]

	ax.set_xticklabels(x_ticks_labels, rotation='80', fontsize=12)
# plt.plot(i, dist_to_origo, "o", c=colors[int(features[i, 1]) + 1])
	print(missclassified[id,0]) 
	print(missclassified[id,2])
	print(missclassified[id,3])

	# song = plt.plot(song1,"o-", markersize=12, label=("Missclassified Song "+str(missclassified[id,0])+str(songlabel1))) #blå
	nearest = plt.plot(song2, "o-",markersize=12,label=("Nearest to song "+str(missclassified[id,2])+str(songlabel2)))#gul
	correct = plt.plot(song3, "o-",markersize=12, label=("Nearest correct to song "+str(missclassified[id,3])+str(songlabel3))) #grön
	# plt.legend(()=[song,nearest,correct],loc='upper left')
	ax.legend()

	plt.show()

def save_to_file(data, filename):
	with open(filename, "w") as f:
		for row in data:
			for col in row:
				f.write(str(col) + ' ')
			f.write('\n')

def feature_mean_by_class():
	features = get_songs_feature_set('features_targets/all_vectors.txt')
	classes = features[:,1]
	# features, mu, va = normalise(features[:,2:])
	features = features[:,2:]
	print(features[0,:])
	for i in range(10):
		print(get_label(classes[0+i*100]))
		print(np.average(features[0+i*100:100+i*100,10]))

def prominent_features():
	prominent_array = np.zeros((11,20), dtype=int)
	prominent_array[0,:] = range(20)
	prominent_array[1:,0] = range(10)
	for j in range(1,20):
		data = read_confusion_matrix("analysis_docs/gmm1_with_single_features/feature_"+str(j-1)+".csv")
		for i in range(1,11):
			prominent_array[i,j] = data[i-1,i-1] 
	# prominent_array= ",".join(str(prominent_array))
	# prominent_array = ",".join(list(map(lambda r: str(r), prominent_array)))
	# prominent_array = re.sub("\s" , "," , str(prominent_array).strip())
	print(prominent_array)
	save_confusion_matrix("Prominent_features_gmm1.csv",prominent_array)

	# read()


if __name__ == '__main__':
	prominent_features()
	# feature_mean_by_class()
	# allDist = read_stored_data('features_targets/all_distances.txt')
	# allDist = np.array(allDist)

	# alldistNoDiag = remove_diagonal(np.copy(allDist))

	# alldistNoDiag = remove_diagonal(np.copy(allDist))
	# save_to_file(get_missclassified_with_neighbors_nearest_and_correct(alldistNoDiag), "features_targets/nearest_and_correct_nearest.txt")
	plot_missclassified_with_neighbor_by_feature(302)
	# plot_missclassified_with_neighbor_by_feature_mfcc(200,False)

	# np.set_printoptions(threshold=sys.maxsize)
	# content = read_content('features_targets/index_of_content.txt')

	# print(np.array(content))

	# print(get_both_nearest_and_correct_neighbors(alldistNoDiag))
	# print(get_missclassified_with_neighbors_nearest_and_correct(alldistNoDiag))
	# create_knn_graph()

	# create_angle_neighbor_graph()
	# create_knn_graph()
	#save_angle_neighbors_to_file()
	#compute_angles()
	#get_k_nearest_neighbors(read_stored_data("features_targets/AllDistances.txt")[:, 2:], 3)
	#create_neighbor_graph()
	#knn_neighbor_count()
	#save_track_features_to_file()
	#plot_all_track_dist_to_origo()
	#knn_distance_measure_correct()
	# view_wrongly_classified()
	# plot_features()
	#compare_popular_song_neighbors()
	#train_set, test_set = get_test_train_sets("features_targets/afe_feat_and_targ.txt",0,42)
	#train_setP, test_setP = partdata()
	# create_neighbor_graph()
	#create_neighbor_graph()

	#train_set, test_set = get_test_train_sets("features_targets/afe_feat_and_targ.txt",0,42)
	#train_setP, test_setP = partdata()


	#allDist, a = read_stored_data('features_targets/Alldistances.txt')
	#allDist = np.array(allDist)


	# alldistNoDiag = remove_diagonal(allDist)
	# nearest = get_nearest_neighbors_dist(alldistNoDiag)
	# nearest_correct = get_nearest_correct_neighbors(alldistNoDiag)
	# both = get_both_nearest_and_correct_neighbors(alldistNoDiag)

	# for i in range(1000):
	#     print(both[i,:])

	# nearesttracks = np.concatenate((nearest,nearest_correct[:,1]),axis=0) #Får inte skiten att funka. 



	
	# compute_distances()
	#train_set, test_set = get_test_train_sets("features_targets/afe_feat_and_targ.txt",0,42)
	#train_setP, test_setP = partdata()

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



	# allDist = read_stored_data('features_targets/Alldistances.txt')
	# allDist = np.array(allDist)

	# alldistNoDiag = remove_diagonal(np.copy(allDist))

	# nearest = get_nearest_neighbors_dist(alldistNoDiag)
	# nearest_correct = get_nearest_correct_neighbors(alldistNoDiag)
	# both = get_both_nearest_and_correct_neighbors(alldistNoDiag)

	# for i in range(1000):
	#     print(both[i, :])
	# for i in range(1000):
	#     print(allDist[i,:7])

	# print(allDist.shape)
	# for i in range(allDist.shape[1]-900):
	#     print(allDist[0,i])
	# nearesttracks = np.concatenate((nearest,nearest_correct[:,1]),axis=0) #Får inte skiten att funka.



