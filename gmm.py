from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from timbral_texture import get_label
import matplotlib.pyplot as plt
import numpy as np

from utils import *

def runRandomGMM():
  features, targets = read_stored_data()
  features = features[:,1:]
  # features = normalise(features)
  features_mean, grouped_targets = mean_var_by_song(features, targets)

  grouped_targets = np.array(grouped_targets)
  

  b = 0
  for o in range(100):
    train_samples, test_samples, train_targets, test_targets = train_test_split(features_mean, grouped_targets, test_size=0.05, random_state=o)
    score = np.empty((test_samples.shape[0], 10))
    predictor_list = []
    for i in range(10):
      predictor = GaussianMixture(
        n_components=3,
        covariance_type='full',
        tol=0.000001,
        max_iter=500,
        n_init=2,
        init_params='kmeans'
        )
      predictor.fit(train_samples[train_targets==i])
      predictor_list.append(predictor)
      score[:, i] = predictor.score_samples(test_samples)
    # print(score)
    # print(score.shape)

    Y_predicted = np.argmax(score, axis=1)
    
    # print(Y_predicted.size)
    # print([i for i in(Y_predicted)])
    a = np.count_nonzero(Y_predicted == test_targets)/len(test_targets)
    # if(a > 0.57):
    #   print(o)
    print('Prediction')
    print(a)
    b = b + a 
  print(b/100)

def runFaultFilteredGMM():
  train_samples, train_targets = read_stored_data('features_targets/featuresF.txt','features_targets/targetsF.txt')
  train_samples = normalise(train_samples)
  test_samples, test_targets = read_stored_data('features_targets/featuresFT.txt','features_targets/targetsFT.txt')
  test_samples = normalise(test_samples)
  vali_samples, vali_targets = read_stored_data('features_targets/featuresFV.txt','features_targets/targetsFV.txt')
  # print(train_samples.shape)
  # print(vali_samples.shape)
  train_samples = np.concatenate([train_samples,vali_samples],0)
  train_targets = np.concatenate([train_targets,vali_targets],0)


  # grouped_features, grouped_targets = group_by_song(features, targets)


  train_samples = normalise(train_samples)
  test_samples = normalise(test_samples)

  train_samples, train_targets = mean_var_by_song(train_samples, train_targets)
  test_samples, test_targets = mean_var_by_song(test_samples, test_targets)
  test_targets = np.array(test_targets)
  train_targets = np.array(train_targets)
  score = np.empty((test_samples.shape[0], 10))
  predictor_list = []
  for i in range(10):
    predictor = GaussianMixture(
      n_components=3,
      covariance_type='full',
      tol=0.000001,
      max_iter=500,
      n_init=2,
      init_params='kmeans'
      )
    predictor.fit(train_samples[train_targets==i])
    predictor_list.append(predictor)
    score[:, i] = predictor.score_samples(test_samples)
  # print(score)
  # print(score.shape)

  Y_predicted = np.argmax(score, axis=1)
  
  # print(Y_predicted.size)
  # print([i for i in(Y_predicted)])
  a = np.count_nonzero(Y_predicted == test_targets)/len(test_targets)

  print('Prediction')
  print(a)

def run_gmm_k_fold():
  features, targets = read_stored_data()
  features = features[:,1:]
  # features = normalise(features)
  features_mean, grouped_targets = mean_var_by_song(features, targets)
  feature_partition, target_partition = k_fold_initialization(features_mean, grouped_targets, 10)

  k = 10
  num_genres = 10
  num_iterations = 100

  iteration_accuracies = []
  for e in range(num_iterations):
    print("Iteration: ", e)
    accuracy_per_partition = []
    for i in range(k):
      train_samples, train_targets, test_samples, test_targets = get_cross_validate_partitions(feature_partition, target_partition, i)
      score = np.empty((test_samples.shape[0], 10))
      for j in range(num_genres):
        predictor = GaussianMixture(
          n_components=3,
          covariance_type='full',
          tol=0.000001,
          max_iter=500,
          n_init=2,
          init_params='kmeans'
          )
        predictor.fit(train_samples[train_targets == j])
        score[:, j] = predictor.score_samples(test_samples)
      Y_predicted = np.argmax(score, axis=1)
      a = np.count_nonzero(Y_predicted == test_targets) / len(test_targets)
      accuracy_per_partition.append(a)
    iteration_accuracy = np.mean(accuracy_per_partition)
    iteration_accuracies.append(iteration_accuracy)
    print("Accuracy: ", iteration_accuracy)

  final_accuracy = np.mean(iteration_accuracies)
  final_accuracy_variance = np.var(iteration_accuracies)
  print("Final accuracy: ", final_accuracy)
  print("Final variance: ", final_accuracy_variance)



if __name__ == '__main__':

  #runFaultFilteredGMM()
  runRandomGMM()
  # run_gmm_k_fold()


# Partition all the samples into 10 equally sized partition, resulting in a 3D matrix
  # (each 3D layer correspond to a partition)
  # partitioned_samples, partitioned_targets = k_fold_initialization(features_mean, grouped_targets, 10)


  #train_samples, train_targets = ungroup(grouped_train_samples, grouped_train_targets)
  #test_samples, test_targets = ungroup(grouped_test_samples, grouped_test_targets)



  # res = gmm.predict(train_samples)
  # print(np.count_nonzero(res==train_targets)/len(train_samples))

