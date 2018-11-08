from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from timbral_texture import get_label
import matplotlib.pyplot as plt
import numpy as np

from utils import *

def runRandomGMM():
  features, targets = read_stored_data()
  # features = normalise(features)
  features_mean = mean_by_song(features)
  grouped_targets = features_mean[:,1]
  features_mean = features_mean[:,2:]
  grouped_targets = np.array(grouped_targets)
  

  b = 0
  for k in range(100):
    train_samples, test_samples, train_targets, test_targets = train_test_split(features_mean, grouped_targets, test_size=0.1, random_state=k)
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
  features, _ = read_stored_data('features_targets/afe_feat_and_targ.txt')
  # features = normalise(features)
  #features = features[:, 1:]
  features_mean = mean_by_song(features)

  k = 10
  num_genres = 10
  num_iterations = 10

  iteration_accuracies = []
  confusion_matrix = np.zeros((num_genres, num_genres))
  for e in range(num_iterations):
    feature_partition = make_k_fold_partition(features_mean, 10)
    print("Iteration: ", e)
    accuracy_per_partition = []
    for i in range(k):
      train_samples, train_targets, test_samples, test_targets = get_k_fold_partitions(feature_partition, i)
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
      for p in range(len(Y_predicted)):
        confusion_matrix[Y_predicted[p], test_targets[p]] += 1
      a = np.count_nonzero(Y_predicted == test_targets) / len(test_targets)
      accuracy_per_partition.append(a)
    iteration_accuracy = np.mean(accuracy_per_partition)
    iteration_accuracies.append(iteration_accuracy)
    print("Accuracy: ", iteration_accuracy)

  final_accuracy = np.mean(iteration_accuracies)
  final_accuracy_variance = np.var(iteration_accuracies)
  print("Final accuracy: ", final_accuracy)
  print("Final variance: ", final_accuracy_variance)
  confusion_matrix /= 10

def gmm_props(m):
  return m.means_, m.covariances_, m.weights_

def avg_mean(means, weights):
  matrices = [means[i]*weights[i] for i in range(len(means))]
  avg = means[0]+means[1]
  for i in range(2, len(weights)):
    avg += means[i]
  return avg

def mean_covar_weights_for_classes():
  # read features
  read_feats, _ = read_stored_data('features_targets/afe_feat_and_targ.txt')
  # norm_read_feats = normalise(read_feats[:,2:])
  feats = read_feats[:,2:]
  targs = read_feats[:,1]
  n_genres = np.unique(targs).shape[0]
  all_props = []
  avg_means = []
  models = []
  models_props = []

  for i in range(n_genres):
    models.append(GaussianMixture(n_components=3))
    models[i].fit(feats[targs == i])
    all_props.append(gmm_props(models[i]))
    avg_means.append(avg_mean(all_props[i][0], all_props[i][2]))

  distances = np.array([
    [np.linalg.norm(avg_means[i]-avg_means[j]) for j in range(n_genres)] 
    for i in range(n_genres)])

  print(np.array(distances))



if __name__ == '__main__':

  # runFaultFilteredGMM()
  # runRandomGMM()
  # run_gmm_k_fold()
  mean_covar_weights_for_classes()


# Partition all the samples into 10 equally sized partition, resulting in a 3D matrix
  # (each 3D layer correspond to a partition)
  # partitioned_samples, partitioned_targets = k_fold_initialization(features_mean, grouped_targets, 10)


  #train_samples, train_targets = ungroup(grouped_train_samples, grouped_train_targets)
  #test_samples, test_targets = ungroup(grouped_test_samples, grouped_test_targets)



  # res = gmm.predict(train_samples)
  # print(np.count_nonzero(res==train_targets)/len(train_samples))

