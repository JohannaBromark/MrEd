from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from timbral_texture import get_label
import matplotlib.pyplot as plt
import numpy as np

def read_stored_data(filename1 = 'featuresO.txt',filename2 = 'targetsO.txt'):
  """Return feature vectors and corr labels from stored txt file"""
  with open(filename1) as f:
    lines = f.readlines()
    features = [[0]] * len(lines)
    for i in range(len(lines)):
      features[i] = [float(i) for i in lines[i].split()]
    features = np.array(features)

  with open(filename2) as f:
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

def mean_by_song(features, targets):
  features_mean = np.zeros((int(len(features)//30), features.shape[1]))
  grouped_targets = []
  features_matrix = np.array(features)
  for i in range(len(features)//30):
    features_mean[i, :] = np.mean(features_matrix[i*30:(i+1)*30, :], 0)
    grouped_targets.append(targets[i*30])

  return features_mean, grouped_targets

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

def plot_feature_vectors(features, targets):
  features = normalise(features)
  c = 0
  for i in range(20):
    plt.subplot(5,4,i+1)
    plt.axis('off')
    if i % 2 == 0:
      plt.title(get_label(targets[3000*c]))
      plt.imshow(np.append(features[3000*c], [0.5]).reshape(4,5))
    else:
      plt.title(get_label(targets[3001*c]))
      plt.imshow(np.append(features[3001*c], [0.5]).reshape(4,5))
      c += 1
  plt.show()


def k_fold_initialization(samples, targets, k):

  """

  :param samples: All samples that are used for training and testing
  :param targets: Targets corresponding to each sample
  :param k: Integer to decide number of partitions
  :return: partitions: 3D matrix (3D layers correspond to the partitions),
           partition_targets: matrix where each row is target for each partitions
  """
  partition_size = samples.shape[0]//k
  partitions = np.zeros((partition_size, samples.shape[1], k))
  partition_targets = np.zeros((k, partition_size))
  for i in range(k):
    partitions[:, :, i] = samples[i*partition_size:(i+1)*partition_size, :]
    partition_targets[i, :] = targets[i*partition_size:(i+1)*partition_size]
  return partitions, partition_targets

def get_cross_validate_partitions(partitioned_samples, partitioned_targets, partition_num):
  """

  :param paritioned_samples: All samples partitioned into equal sized partitions (stored as 3D matrix)
  :param partition_num: The partition to be training set
  :return: training set and test set
  """
  k = partitioned_samples.shape[2]
  N = partitioned_samples.shape[0]
  test_samples = partitioned_samples[:, : partition_num]
  train_samples_ = np.zeros((N*(k-1), partitioned_samples.shape[1]))

  test_targets = partitioned_targets[partition_num, :]
  train_targets = np.delete(partitioned_targets, np.s_[partition_num*N:(partition_num+1)*N], None)
  j = 0
  for i in range(k):
    if i != partition_num:
      partitioned = partitioned_samples[:, :, i]
      train_samples_[j*N:(j+1)*N, :] = partitioned
      j += 1

  return train_samples_, train_targets, test_samples, test_targets



def k_fold_initialization(samples, targets, k):

  """

  :param samples: All samples that are used for training and testing
  :param targets: Targets corresponding to each sample
  :param k: Integer to decide number of partitions
  :return: partitions: 3D matrix (3D layers correspond to the partitions),
           partition_targets: matrix where each row is target for each partitions
  """
  partition_size = samples.shape[0]//k
  partitions = np.zeros((partition_size, samples.shape[1], k))
  partition_targets = np.zeros((k, partition_size))
  for i in range(k):
    partitions[:, :, i] = samples[i*partition_size:(i+1)*partition_size, :]
    partition_targets[i, :] = targets[i*partition_size:(i+1)*partition_size]
  return partitions, partition_targets

def get_cross_validation_sets(partitioned_samples, partitioned_targets, partition_num):
  """
  Sets one partition as test set and the remaining as training set.
  :param paritioned_samples: All samples partitioned into equal sized partitions (stored as 3D matrix)
  :param partition_num: The partition to be the testing set
  :return: training set and test set
  """
  k = partitioned_samples.shape[2]
  N = partitioned_samples.shape[0]
  test_samples = partitioned_samples[:, : partition_num]
  train_samples_ = np.zeros((N*(k-1), partitioned_samples.shape[1]))

  test_targets = partitioned_targets[partition_num, :]
  train_targets = np.delete(partitioned_targets, np.s_[partition_num*N:(partition_num+1)*N], None)
  j = 0
  for i in range(k):
    if i != partition_num:
      partitioned = partitioned_samples[:, :, i]
      train_samples_[j*N:(j+1)*N, :] = partitioned
      j += 1

  return train_samples_, train_targets, test_samples, test_targets


if __name__ == '__main__':
  features, targets = read_stored_data()
  train_samples, train_targets = read_stored_data('featuresF.txt','targetsF.txt')
  test_samples, test_targets = read_stored_data('featuresFT.txt','targetsFT.txt')
  print(features.size)
  print(train_samples.size)
  print(train_targets.size)
  print(test_samples.size)
  print(test_targets.size)


  #grouped_features, grouped_targets = group_by_song(features, targets)

  # features_mean, grouped_targets = mean_by_song(features, targets)

  # features = normalise(features)

  #train_samples, test_samples, train_targets, test_targets = train_test_split(
  #  features,
  #  targets,
  #  test_size=0.33,
  #  random_state=42)


  #grouped_train_samples, grouped_test_samples, grouped_train_targets, grouped_test_targets = train_test_split(
  #  grouped_features,
  #  grouped_targets,
  #  test_size=0.33,
  #  random_state=42)
#
  #train_samples, train_targets = ungroup(grouped_train_samples, grouped_train_targets)
  #test_samples, test_targets = ungroup(grouped_test_samples, grouped_test_targets)


  # Partition all the samples into 10 equally sized partition, resulting in a 3D matrix
  # (each 3D layer correspond to a partition)
  # partitioned_samples, partitioned_targets = k_fold_initialization(features_mean, grouped_targets, 10)

  # grouped_targets = np.array(grouped_targets)
  # train_samples, test_samples, train_targets, test_targets = train_test_split(features_mean, grouped_targets, test_size=0.33, random_state=42)

  # gmm = GaussianMixture(n_components=10)
  # gmm.fit(train_samples)

  # res = gmm.predict(train_samples)
  # print(np.count_nonzero(res==train_targets)/len(train_samples))

  score = np.empty((test_samples.shape[0], 10))
  predictor_list = []
  for i in range(10):
    predictor = GaussianMixture(n_components=3,n_init=1)
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
