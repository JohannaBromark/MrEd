from sklearn.neighbors import KNeighborsClassifier
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from timbral_texture import get_label
import matplotlib.pyplot as plt
import numpy as np
from utils import *

def run_knn_k_fold():
  features, _ = read_stored_data('features_targets/afe_feat_and_targ.txt')
  features_mean = mean_by_song(features)

  k = 10
  num_genres = 10
  num_iterations = 100

  iteration_accuracies = []
  heat_map = np.zeros((num_genres, num_genres))
  for e in range(num_iterations):
    feature_partition = make_k_fold_partition_equal(features_mean, 10)
    print("Iteration: ", e)
    s = 0
    predictions_matrix = np.zeros((num_genres, num_genres))
    for i in range(k):
      train_samples, train_targets, test_samples, test_targets = get_k_fold_partitions(feature_partition, i)
      train_samples = normalise(train_samples)
      test_samples = normalise(test_samples)

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


if __name__ == '__main__':
  features, targets = read_stored_data('features_targets/afe_feat_and_targ.txt')
  # features = normalise(features)
  features_mean = mean_by_song(features)
  grouped_targets = features_mean[:,1]
  features_mean = features_mean [:,2:]


  grouped_targets = np.array(grouped_targets)

  b = 0

  for i in range(100):
    train_samples, test_samples, train_targets, test_targets = train_test_split(features_mean, grouped_targets, test_size=0.1, random_state=i)

    knn = KNeighborsClassifier(n_neighbors=1)

    knn.fit(train_samples,train_targets)

    predictions = knn.predict(test_samples)
    score = knn.score(test_samples, test_targets)
    b = b + score
    print(score)
  print(b/100)


  #run_knn_k_fold()

