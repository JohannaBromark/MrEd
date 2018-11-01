from sklearn.neighbors import KNeighborsClassifier
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from timbral_texture import get_label
import matplotlib.pyplot as plt
import numpy as np


def read_stored_data():
  """Return feature vectors and corr labels from stored txt file"""
  with open('features_targets/features.txt') as f:
    lines = f.readlines()
    features = [[0]] * len(lines)
    for i in range(len(lines)):
      features[i] = [float(i) for i in lines[i].split()]
    features = np.array(features)

  with open('features_targets/targets.txt') as f:
    targets = np.array([int(i) for i in f.readlines()])
    # with brackets
    # targets = np.array([[int(i)] for i in f.readlines()])

  return features, targets


def normalise(features):
  for i in range(len(features[1])):
    features[:,i] = np.interp(features[:,i], (features[:,i].min(), features[:,i].max()), (0, 1))
  return features

def group_by_song(features, targets):
  songs = []
  grouped_targets = []
  for i in range(len(features)//30):
    songs.append(features[i*30:(i+1)*30])
    grouped_targets.append(targets[i*30])
  return np.array(songs), grouped_targets

def mean_var_by_song(features, targets):
  features_mean = np.zeros((int(len(features)//30), features.shape[1]))
  grouped_targets = []
  features_matrix = np.array(features)
  for i in range(len(features)//30):
    features_mean[i, :] = np.mean(features_matrix[i*30:(i+1)*30, :], 0)
    # SHOULD THERE BE VARIANCE AS WELL --> 38 dimensions?
    grouped_targets.append(targets[i*30])

  return features_mean, grouped_targets

def ungroup(grouped_features, grouped_targets):
  targets_noflat = np.array([[i]*30 for i in grouped_targets])
  targets = targets_noflat.flatten()

  features_flat = grouped_features.flatten()
  features = features_flat.reshape(-1, grouped_features.shape[2])

  return features, targets

  

if __name__ == '__main__':
	features, targets = read_stored_data()
	features = normalise(features)
	
	features_mean, grouped_targets = mean_var_by_song(features, targets)

	grouped_targets = np.array(grouped_targets)

	train_samples, test_samples, train_targets, test_targets = train_test_split(features_mean, grouped_targets, test_size=0.33, random_state=42)

	knn = KNeighborsClassifier(n_neighbors=1)

	knn.fit(train_samples,train_targets)

	predictions = knn.predict(test_samples)
	score = knn.score(test_samples,test_targets)
	print(score)

