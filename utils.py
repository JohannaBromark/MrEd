from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis import audioFeatureExtraction as audioFE
import matplotlib.pyplot as plt
import numpy as np
import numpy.fft as fft
import scipy.signal as signal
from sklearn.preprocessing import normalize
from sklearn.mixture import GaussianMixture
import os
import random

##################
### Read files ###

def read_file(file_name='genres/rock/rock.00093.wav'):
  """Return 22050 Hz sampling frequency and sample amplitudes"""
  # pop: genres/pop/pop.00000.wav
  return audioBasicIO.readAudioFile(file_name)

def read_directory(genre='rock'):
  path = 'genres/' + genre + '/'
  all_samples = [[0]] * len([f for f in os.listdir(path)])
  i = 0
  for filename in os.listdir(path):
    sample_rate, all_samples[i] = read_file(os.path.join(path, filename))
    i += 1
  all_samples = np.array(all_samples)
  labels = np.full((100,1), get_label(genre))
  return all_samples, labels

def read_directories():
  all_songs = []
  all_labels = []
  i = 0
  for name in os.listdir('genres/'):
    songs, labels = read_directory(name)
    all_songs[i*100:(i+1)*100] = songs
    all_labels[i*100:(i+1)*100] = labels
    i += 1
  all_songs = np.array(all_songs)
  all_labels = np.array(all_labels)
  return all_songs, all_labels

def get_path(txt):
  with open(txt, "r") as ins:
    paths = []
    for line in ins:
        paths.append(line)
  return paths

def read_stored_data(feat_name='features_targets/afe_features.txt', tar_name='features_targets/afe_targets.txt'):
  """Return feature vectors and corr labels from stored txt file"""
  with open(feat_name) as f:
    lines = f.readlines()
    features = [[0]] * len(lines)
    for i in range(len(lines)):
      features[i] = [float(i) for i in lines[i].split()]
    features = np.array(features)

  with open(tar_name) as f:
    targets = np.array([int(i) for i in f.readlines()])
    # with brackets
    # targets = np.array([[int(i)] for i in f.readlines()])

  return features, targets



##################
### Write files ###

def write_features_to_file(features, file_name='features.txt'):
  with open('features.txt','w') as file:
    for item in features:
      for element in item:
        file.write(str(element))
        file.write(' ')
      file.write('\n')

def write_targes_to_file(targets, file_name='targets.txt'):
  with open('targets.txt', 'w') as file:
    for item in targets:
      file.write(str(int(item[0])))
      file.write('\n')
  
def write_afe_to_file(songs, targets, f_name):
  with open('features_targets/' + f_name, 'w') as f_t:
    c = 0
    for song in songs:
      for vector in song:
        f_t.write(str(c) + ' ')
        f_t.write(str(targets[c][0]) + ' ')
        for feat in vector:
          f_t.write(str(feat) + ' ') 
        f_t.write('\n')
      c += 1



##################
### Labels ###

def get_label(target):
  labels = {
    'blues' : 0,
    'classical' : 1,
    'country' : 2,
    'disco' : 3,
    'hiphop' : 4,
    'jazz' : 5,
    'metal' : 6,
    'pop' : 7,
    'reggae' : 8,
    'rock' : 9
  }

  if isinstance(target, str):
    return labels.get(target)
  else:
    for key in labels:
      if labels.get(key) == target:
        return key



##################
### Partitioning ###

def read_partition(path):
  path = get_path(path)
  all_songs = []
  all_labels = np.zeros(len(path))
  # all_samples = np.zeros(len(path))
  all_samples = [[0]] * len([f for f in range(len(path))])

  i = 0
  for p in path:
    sample_rate, all_samples[i] = read_file('genres/' + p.strip())
    label = get_label(p.split('/')[0])
    all_labels[i] = label
    i += 1

  all_labels = np.array(all_labels)

  # all_samples = np.array(all_samples)
  return sample_rate, all_samples, all_labels



  ################
  ### Plotting ###

def plot_fft(samples, sample_rate):
  """Plot FFT of an audio sample"""
  # Perioden för ett sample
  T = 1./sample_rate
  # Antal samples
  N = len(samples)
  # Returnerar N st amplituder för varje frekvens från DFT
  fft = np.fft.fft(samples)
  # N st mappade frekvenser (bins) för varje amplitud
  # För varje N st amplituder med index i -> frekvens fi = i * sample_rate/N
  freqs = np.linspace(0, 1 / T, N)
  # Tar endast första halvan pga Nyqvist-frekvens
  plt.title('FFT Magnitude')
  plt.plot(freqs[:N // 2], np.abs(fft)[:N // 2] * 1 / N)  # 1 / N is a normalization factor
  plt.ylabel('Amplitude')
  plt.xlabel('Frequency [Hz]')
  plt.show()

def plot_whole_stft(samples, sample_rate):
  """Plot STFT of an audio sample"""
  f, t, Zxx = signal.stft(samples, fs=sample_rate, nperseg=512*2, noverlap=0)
  plt.pcolormesh(t, f, np.abs(Zxx))
  plt.title('STFT Magnitude')
  plt.ylabel('Frequency [Hz]')
  plt.xlabel('Time [sec]')
  plt.show()

def plot_an_window(an_wndw, freqs, include_centroid=True):
  """Plot an analysis window from STFT with optional spectral centroid"""
  plt.title('Window FFT')
  plt.ylabel('Amplitude')
  plt.xlabel('Frequency [Hz]')
  plt.plot(freqs, an_wndw)
  if include_centroid:
    plt.plot(spectral_centroid(an_wndw, freqs), 0, 'o', label='centroid')
  plt.show()


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



#################
### Normalise ###

def normalise(features):
  for i in range(len(features[1])):
    features[:,i] = np.interp(features[:,i], (features[:,i].min(), features[:,i].max()), (0, 1))
  return features



###############################
### Grouping and ungrouping ###

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



##################
### Song means ###

def mean_var_by_song(features, targets):
  features_mean = np.zeros((int(len(features)//30), features.shape[1]))
  grouped_targets = []
  features_matrix = np.array(features)
  for i in range(len(features)//30):
    features_mean[i, :] = np.mean(features_matrix[i*30:(i+1)*30, :], 0)
    # SHOULD THERE BE VARIANCE AS WELL --> 38 dimensions?
    grouped_targets.append(targets[i*30])

  return features_mean, grouped_targets



##############
### k fold ###

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
  partition_targets = np.zeros((k, partition_size), dtype="int64")
  shuffle_array = [x for x in random.sample(range(samples.shape[0]), samples.shape[0])]

  # Shuffling the data
  shuffled_samples = np.take(samples, shuffle_array, 0)
  shuffled_targets = np.take(targets, shuffle_array)

  """ Tests
  print(samples[shuffle_array[0]] == shuffled_samples[0])
  print(samples[shuffle_array[440]] == shuffled_samples[440])
  print(samples[shuffle_array[500]] == shuffled_samples[500])
  print(samples[shuffle_array[337]] == shuffled_samples[337])
  print(samples[shuffle_array[993]] == shuffled_samples[993])
  print(targets[shuffle_array[0]] == shuffled_targets[0])
  print(targets[shuffle_array[440]] == shuffled_targets[440])
  print(targets[shuffle_array[500]] == shuffled_targets[500])
  print(targets[shuffle_array[337]] == shuffled_targets[337])
  print(targets[shuffle_array[993]] == shuffled_targets[993])
  """

  for i in range(k):
    partitions[:, :, i] = shuffled_samples[i*partition_size:(i+1)*partition_size, :]
    partition_targets[i, :] = shuffled_targets[i*partition_size:(i+1)*partition_size]
  return partitions, partition_targets


def get_cross_validate_partitions(partitioned_samples, partitioned_targets, partition_num):
  """
  :param paritioned_samples: All samples partitioned into equal sized partitions (stored as 3D matrix)
  :param partition_num: The partition to be training set
  :return: training set and test set
  """
  k = partitioned_samples.shape[2]
  N = partitioned_samples.shape[0]
  test_samples = partitioned_samples[:, :, partition_num]
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