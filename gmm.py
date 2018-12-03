from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from scipy.stats import entropy as entropy
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
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
  ### Prepare feature data
  features = read_stored_data('features_targets/all_vectors.txt')
  feature_vectors = mean_by_song(features)

  ### Normalisation with GMM3 gives bad results.. 
  # songs[:,2:] = normalise(songs[:,2:])[0]
  
  ### GMM and partition settings 
  num_genres = np.unique(features[:,1]).shape[0]
  num_iterations = 10
  n_components = 3
  n_folds = 10
  # MFCC 0 mean and var is 8 and 9
  filter_idxs = [8,9,12,13]

  # Store accuracies and create a confusion matrix 
  iteration_accuracies = []
  confusion_matrix = np.zeros((num_genres, num_genres))

  for e in range(num_iterations):
    # Create partitions
    partitions = make_k_fold_partition(feature_vectors, n_folds)

    print("Iteration: ", e)
    accuracy_per_partition = []

    for i in range(n_folds):
      # Get train and test sets
      train_samples, train_targets, test_samples, test_targets = get_k_fold_partitions(partitions, i)

      train_samples = train_samples[:,filter_idxs]
      test_samples = test_samples[:,filter_idxs]

      score = np.empty((test_samples.shape[0], 10))

      # Train models and classify test samples
      for j in range(num_genres):
        predictor = GaussianMixture(n_components=n_components)
        predictor.fit(train_samples[train_targets == j])
        score[:, j] = predictor.score_samples(test_samples)
      Y_predicted = np.argmax(score, axis=1)

      # Create confusion matrix
      for p in range(len(Y_predicted)):
        confusion_matrix[Y_predicted[p], test_targets[p]] += 1

      # Compute accuracy
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
  confusion_matrix = (confusion_matrix / 10).astype("int64")

  #save_confusion_matrix("analysis_docs/confusion_matrix_gmm.csv", confusion_matrix)

def gmm_props(m):
  """Return model means, covariances and weights for a given GMM model"""
  return m.means_, m.covariances_, m.weights_

def train_gmm_models(features, targets, n_components=3):
  """Train GMM models for each label"""
  all_models = []
  n_genres = np.unique(targets).shape[0]
  for i in range(n_genres):
    all_models.append(GaussianMixture(n_components))
    all_models[i].fit(features[targets == i])
  return np.array(all_models)

def kl_divergence(m0, cov0, m1, cov1, n_dim):
  """Compute the KL divergence of two gaussian distributions"""
  # https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Kullback%E2%80%93Leibler_divergence

  trace = np.trace(np.matmul(np.linalg.inv(cov1), cov0))
  alg = ((m1-m0).T * np.linalg.inv(cov1) * (m1-m0))[0][0]
  log = np.log(np.linalg.det(cov1)/np.linalg.det(cov0))

  return 1/2 * (trace + alg - n_dim + log)
    
def kl_distance_between(model1, model2):
  """Compute KL distance between two GMMs.
  Distance = D(GMM1 || GMM2) + D(GMM2 || GMM1) where
  In GMM3, D(GMM1 || GMM2) = 
  Aw1Aw2 D(A1 || A2) + Aw1Bw2 D(A1 || B2) + Aw1Cw2 D(A1 || C2) 
  Bw1Aw2 D(B1 || A2) + Bw1Bw2 D(B1 || B2) + Bw1Cw2 D(B1 || C2) 
  Cw1Aw2 D(C1 || A2) + Cw1Bw2 D(C1 || B2) + Cw1Cw2 D(C1 || C2)
  """
  means_m1, covars_m1, weights_m1 = gmm_props(model1)
  means_m2, covars_m2, weights_m2 = gmm_props(model2)
  n_dim = means_m1.shape[1]
  n_comp = len(means_m1)

  dist = 0
  for i in range(n_comp):
    for j in range(n_comp):
      dist += weights_m1[i] * weights_m2[j] * kl_divergence(means_m1[i], covars_m1[i], means_m2[j], covars_m2[j], n_dim)
      dist += weights_m2[i] * weights_m1[j] * kl_divergence(means_m2[i], covars_m2[i], means_m1[j], covars_m1[j], n_dim)
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

def read_and_combine_fault_filtered_train_and_val():
  training = read_stored_data('features_targets/fault_filtered_vectors_train.txt')
  val = read_stored_data('features_targets/fault_filtered_vectors_valid.txt')
  return np.concatenate((training,val))

def mfcc_only(vectors):
  filtered = [0,1,10,11,12,13,14,15,16,17,18,19]
  return vectors[:,filtered]

def write_gmm_distance_to_csv(kl_distances, file_name):
  with open(file_name, 'w') as f:
    f.write(',')
    for i in range(10):
      f.write(get_label(i) + ',')
    f.write('\n')

    c = 0
    for i, distance in enumerate(kl_distances):
      f.write(get_label(i) + ',')
      for j in distance:
        f.write(str(j) + ',')
      f.write('\n')
    f.write('\n')

    f.write('Nearest classes\nGenre,Nearest,2nd nearest,3rd nearest\n')
    for j, dist in enumerate(kl_distances):
      sort = np.argsort(dist)
      f.write(get_label(j) + ',')
      for i in range(3):
        f.write(get_label(sort[i]) + ',')
      f.write('\n')

def plot_distances(kl_distances, f_name):

  # Set up adjacency matrix
  dt = [('len', float)]
  tuple_distances = np.array([tuple(dist) for dist in kl_distances]) / 5
  A = tuple_distances.view(dt)

  # Create and draw graph
  G = nx.from_numpy_matrix(A)
  labels = dict(zip([i for i in range(10)], [get_label(i) for i in range(10)]))
  G = nx.relabel_nodes(G, labels)
  G = nx.drawing.nx_agraph.to_agraph(G)

  # Save fig
  G.node_attr.update(color="red", style="filled")
  # G.draw(f_name, format='png', prog='neato')


def compare_gmms():
  ### Read vectors
  # f_vectors = read_stored_data()
  # f_vectors = read_and_combine_fault_filtered_train_and_val()
  # f_vectors = mfcc_only(f_vectors)
  # features = f_vectors[:,2:]
  # features, _, _ = normalise(features)
  # targets = f_vectors[:,1]
  
  ### Read songs
  train_songs = get_songs_feature_set('features_targets/fault_filtered_vectors_train.txt')
  val_songs = get_songs_feature_set('features_targets/fault_filtered_vectors_valid.txt')
  
  all_songs = np.concatenate((train_songs, val_songs))
  f_songs = mfcc_only(all_songs)
  f_songs = normalise(f_songs[:,2:])[0]
  t_songs = all_songs[:,1]


  ### Train model and get distances
  n_comp = 3
  models = train_gmm_models(f_songs, t_songs, n_comp)
  kl_distances = kl_distances_matrix(models)

  path = 'analysis_docs/gmm_comparisons/song_mean/mfcc_only/fault_filtered/'

  ### Save distances to .csv
  # write_gmm_distance_to_csv(kl_distances, path+'gmm'+str(n_comp)+'_distances.csv')

  ### Save graph to .png
  # plot_distances(kl_distances, path+'gmm'+str(n_comp)+'_distances_vis.png')

if __name__ == '__main__':

  # runFaultFilteredGMM()
  # runRandomGMM()
  run_gmm_k_fold()
  # compare_gmms()

  
