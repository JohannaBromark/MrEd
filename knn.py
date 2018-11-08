from sklearn.neighbors import KNeighborsClassifier
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from timbral_texture import get_label
import matplotlib.pyplot as plt
import numpy as np
from utils import *

def run_knn_random():
  features, targets = read_stored_data('features_targets/afe_feat_and_targ.txt')
  # features = normalise(features)
  features_mean = mean_by_song(features)
  grouped_targets = features_mean[:, 1]
  features_mean = features_mean[:, 2:]

  grouped_targets = np.array(grouped_targets)

  b = 0

  for i in range(100):
    train_samples, test_samples, train_targets, test_targets = train_test_split(features_mean, grouped_targets,
                                                                                test_size=0.1, random_state=i)

    knn = KNeighborsClassifier(n_neighbors=1)

    knn.fit(train_samples, train_targets)

    predictions = knn.predict(test_samples)
    score = knn.score(test_samples, test_targets)
    b = b + score
    print(score)
  print(b / 100)


def run_knn_k_fold():
  features, _ = read_stored_data('features_targets/afe_feat_and_targ.txt')
  features_mean = mean_by_song(features)

  k = 10
  num_genres = 10
  num_iterations = 100

  iteration_accuracies = []
  heat_map = np.zeros((num_genres, num_genres))
  for e in range(num_iterations):
    feature_partition = make_k_fold_partition(features_mean, 10)
    print("Iteration: ", e)
    s = 0
    predictions_matrix = np.zeros((num_genres, num_genres))
    for i in range(k):
      train_samples, train_targets, test_samples, test_targets = get_k_fold_partitions(feature_partition, i)
      train_samples, mu, std = normalise(train_samples)
      test_samples = (test_samples - mu)/std

      knn = KNeighborsClassifier(n_neighbors=1)

      knn.fit(train_samples, train_targets)

      predictions = knn.predict(test_samples)
      for j in range(len(predictions)):
        predictions_matrix[predictions[j], test_targets[j] ] += 1
      score = knn.score(test_samples, test_targets)
      s += score
    heat_map += (predictions_matrix / k)
    iteration_accuracies.append(s/k)
    print(s/k)
  heat_map /= num_iterations
  print(np.mean(iteration_accuracies))

def run_knn_find_neighbors():
  train_set, test_set = get_test_train_sets('features_targets/afe_feat_and_targ.txt')

  train_samples = train_set[:, 2:]
  train_targets = train_set[:, 1]
  test_samples = test_set[:, 2:]
  test_targets = test_set[:, 1]

  knn = KNeighborsClassifier(n_neighbors=1)
  knn.fit(train_samples, train_targets)
  neighbors_dist, neighbors_idx = knn.kneighbors(test_samples)

  # Create pair of matching training and test data

  neighbor_pairs = []

  for i in range(len(neighbors_idx)):
    neighbor_pairs.append([test_set[i, :2], train_set[neighbors_idx[i][0], :2]])

  neighbor_pairs_matrix = np.array(neighbor_pairs)
  """
  This is essentially what a confution matrix does
  num_correct = [0]*10
  num_incorrect = [0]*10

  for i in range(len(neighbor_pairs)):
    test_target = int(neighbor_pairs[i][0][1])
    if test_target == int(neighbor_pairs[i][1][1]):
      num_correct[test_target] += 1
    else:
      num_incorrect[test_target] += 1

  incorrect = [0]*10
  correct = [0]*10
  for i in range(len(num_incorrect)):
    incorrect[i] = (num_incorrect[i]/(num_incorrect[i]+num_correct[i]))
    correct[i] = (num_correct[i] / (num_incorrect[i]+num_correct[i]))

  """





if __name__ == '__main__':


  run_knn_k_fold()
  #run_knn_find_neighbors()



