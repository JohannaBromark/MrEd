from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
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

def group_by_song(features, targets):
  songs = []
  grouped_targets = []
  for i in range(len(features)//30):
    songs.append(features[i*30:(i+1)*30])
    grouped_targets.append(targets[i*30])
  return np.array(songs), grouped_targets

def ungroup(grouped_features, grouped_targets):
  targets_noflat = np.array([[i]*30 for i in grouped_targets])
  targets = targets_noflat.flatten()

  features_flat = grouped_features.flatten()
  features = features_flat.reshape(-1, grouped_features.shape[2])

  return features, targets


def normalise(features):
  for i in range(len(features[1])):
    features[:,i] = np.interp(features[:,i], (features[:,i].min(), features[:,i].max()), (0, 1))
  return features

if __name__ == '__main__':
  features, targets = read_stored_data()

  grouped_features, grouped_targets = group_by_song(features, targets)

  # features = normalise(features)

  #train_samples, test_samples, train_targets, test_targets = train_test_split(
  #  features,
  #  targets,
  #  test_size=0.33,
  #  random_state=42)

  grouped_train_samples, grouped_test_samples, grouped_train_targets, grouped_test_targets = train_test_split(
    grouped_features,
    grouped_targets,
    test_size=0.33,
    random_state=42)

  train_samples, train_targets = ungroup(grouped_train_samples, grouped_train_targets)
  test_samples, test_targets = ungroup(grouped_test_samples, grouped_test_targets)

  # gmm = GaussianMixture(n_components=10)
  # gmm.fit(train_samples)

  # res = gmm.predict(train_samples)
  # print(np.count_nonzero(res==train_targets)/len(train_samples))

  score = np.empty((test_samples.shape[0], 10))
  predictor_list = []
  for i in range(10):
    predictor = GaussianMixture(n_components=1)
    test = train_samples[train_targets==i]
    predictor.fit(train_samples[train_targets==i])
    predictor_list.append(predictor)
    score[:, i] = predictor.score_samples(test_samples)
  # print(score)
  # print(score.shape)

  Y_predicted = np.argmax(score, axis=1)
  
  # print(Y_predicted.size)
  print([i for i in(Y_predicted)])
  a = np.count_nonzero(Y_predicted == test_targets)/len(test_targets)
  print(a)