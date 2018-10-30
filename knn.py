from sklearn.neighbors import KNeighborsClassifier
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from timbral_texture import get_label
import matplotlib.pyplot as plt
import numpy as np


def read_stored_data():
  """Return feature vectors and corr labels from stored txt file"""
  with open('features.txt') as f:
    lines = f.readlines()
    features = [[0]] * len(lines)
    for i in range(len(lines)):
      features[i] = [float(i) for i in lines[i].split()]
    features = np.array(features)

  with open('targets.txt') as f:
    targets = np.array([int(i) for i in f.readlines()])
    # with brackets
    # targets = np.array([[int(i)] for i in f.readlines()])

  return features, targets


def normalise(features):
  for i in range(len(features[1])):
    features[:,i] = np.interp(features[:,i], (features[:,i].min(), features[:,i].max()), (0, 1))
  return features

  

if __name__ == '__main__':
	features, targets = read_stored_data()
	features = normalise(features)


	train_samples, test_samples, train_targets, test_targets = train_test_split(features, targets, test_size=0.15, random_state=42)

	knn = KNeighborsClassifier(n_neighbors=1)

	knn.fit(train_samples,train_targets)

	predictions = knn.predict(test_samples)
	score = knn.score(test_samples,test_targets)
	print(score)

