from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from timbral_texture import get_label
from scipy.stats import entropy as entropy
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

def train_gmm_models(features, targets, n_components=3):
  """Train GMM models for each label"""
  all_models = []
  n_genres = np.unique(targets).shape[0]
  for i in range(n_genres):
    all_models.append(GaussianMixture(
      n_components=3))
    all_models[i].fit(features[targets == i])
  return np.array(all_models)


def kl_divergence(m0, cov0, m1, cov1, n_dim=19):
  """Compute the KL divergence of two gaussian distributions"""
  # https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Kullback%E2%80%93Leibler_divergence

  trace = np.trace(np.matmul(np.linalg.inv(cov1), cov0))
  alg = ((m1-m0).T * np.linalg.inv(cov1) * (m1-m0))[0][0]
  log = np.log(np.linalg.det(cov1)/np.linalg.det(cov0))

  return 1/2 * (trace + alg - n_dim + log)
    
def kl_distance_between(model1, model2):
  """Compute KL distance between two GMMs.
  Distance = D(GMM1 || GMM2) + D(GMM2 || GMM1) where
  D(GMM1 || GMM2) = 
  Aw1Aw2 D(A1 || A2) + Aw1Bw2 D(A1 || B2) + Aw1Cw2 D(A1 || C2) 
  Bw1Aw2 D(B1 || A2) + Bw1Bw2 D(B1 || B2) + Bw1Cw2 D(B1 || C2) 
  Cw1Aw2 D(C1 || A2) + Cw1Bw2 D(C1 || B2) + Cw1Cw2 D(C1 || C2)
  """
  means_m1, covars_m1, weights_m1 = gmm_props(model1)
  means_m2, covars_m2, weights_m2 = gmm_props(model2)
  n_comp = len(means_m1)

  dist = 0
  for i in range(n_comp):
    for j in range(n_comp):
      dist += weights_m1[i] * weights_m2[j] * kl_divergence(means_m1[i], covars_m1[i], means_m2[j], covars_m2[j])
      dist += weights_m2[i] * weights_m1[j] * kl_divergence(means_m2[i], covars_m2[i], means_m1[j], covars_m1[j])
  return dist

def kl_distances_matrix(models):
  """Compute matrix with KL distances between all models"""
  n_models = len(models)
  distance_matrix = np.zeros((n_models, n_models))
  for i in range(n_models):
    distances = np.zeros(n_models)
    for j in range(n_models):
      distances[j] = (kl_distance_between(models[i], models[j]))
    distance_matrix[i] = distances
  return np.array(distance_matrix)

def compare_gmms():
  read_feats, _ = read_stored_data('features_targets/afe_feat_and_targ.txt')
  feats = read_feats[:,2:]
  feats, _, _ = normalise(feats)
  targs = read_feats[:,1]

  models = train_gmm_models(feats, targs, n_components=3)
  kl_distances = kl_distances_matrix(models)
  plt.imshow(kl_distances, cmap='gray')
  
  plt.show()

if __name__ == '__main__':

  # runFaultFilteredGMM()
  # runRandomGMM()
  # run_gmm_k_fold()
  compare_gmms()

