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
from sklearn.model_selection import train_test_split

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
  return all_samples, labels, np.array(os.listdir(path))

def read_directories():
  all_songs = []
  all_labels = []
  all_names = []
  i = 0
  for name in os.listdir('genres/'):
    songs, labels, file_names = read_directory(name)
    all_songs[i*100:(i+1)*100] = songs
    all_labels[i*100:(i+1)*100] = labels
    all_names[i*100:(i+1)*100] = file_names
    i += 1
  all_songs = np.array(all_songs)
  all_labels = np.array(all_labels)
  all_names = np.array(all_names)
  return all_songs, all_labels, all_names

def get_path(txt):
  with open(txt, "r") as ins:
    paths = []
    for line in ins:
        paths.append(line)
  return paths

def read_stored_data(feat_name='features_targets/all_vectors.txt'):
  """Return feature vectors and corr labels from stored txt file"""
  with open(feat_name) as f:
    lines = f.readlines()
    features = [[0]] * len(lines)
    for i in range(len(lines)):
      features[i] = [float(i) for i in lines[i].split()]
      features[i][:2] = [int(i) for i in features[i][:2]]
    features = np.array(features)
  return features



##################
### Write files ###
  
def write_afe_to_file(songs, targets, song_names, f_name):
  with open('features_targets/' + f_name, 'w') as f_t:
    c = 0
    for song, name in zip(songs, song_names):
      for vector in song:
        print(targets[c][0])
        print(str(targets[c][0]) + str(name[-6:-4]))
        f_t.write(str(targets[c][0]) + str(name[-6:-4]) + ' ')
        f_t.write(str(targets[c][0]) + ' ')
        for feat in vector:
          f_t.write(str(feat) + ' ') 
        f_t.write('\n')
      c += 1

def save_confusion_matrix(filename, confusion_matrix):
  with open(filename, "w") as file:
    file.write("," + ",".join([get_label(i) for i in range(10)]))
    file.write("\n")
    for row_idx, row in enumerate(confusion_matrix):
      file.write(get_label(row_idx) + ",")
      file.write(",".join(list(map(lambda r: str(r), row))) + "\n")


def save_matrix(filename, matrix):
  with open(filename, "w") as file:
    for row in matrix:
      file.write(" ".join(list(map(lambda r: str(r), row))))
      file.write("\n")

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
  all_samples = [[0]] * len([f for f in range(len(path))])
  all_names = []

  i = 0
  for p in path:
    sample_rate, all_samples[i] = read_file('genres/' + p.strip())
    all_labels[i] = get_label(p.split('/')[0])
    all_names.append(p.strip().split('/')[1])
    i += 1
  
  all_labels = np.array(all_labels)
  all_labels = all_labels.reshape(all_labels.size,1)
  all_labels = [[int(i[0])] for i in all_labels]
  
  all_names = np.array(all_names)

  return sample_rate, all_samples, all_labels, all_names

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
  """Normalise by subtracting mean and dividing std for each dimension for whole set
  
  :return: features, means and stds for each dimension"""
  n_vec, n_feats = features.shape

  # if n_feats != 19:
  #   raise ValueError("Wrong number of feature dimensions for normalisation: " + str(n_feats))

  used_means, used_stds = np.zeros(n_feats), np.zeros(n_feats)
  norm_features = np.zeros(features.shape)
  for i in range(n_feats):
    used_means[i] = np.sum(features[:,i])/n_vec
    used_stds[i] = np.std(features[:,i])
    norm_features[:, i] = (features[:, i] - used_means[i]) / used_stds[i]

  return norm_features, used_means, used_stds


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

def mean_by_song(features):
  num_songs = len(np.unique(features[:, 0]))
  songs = np.zeros((num_songs, features.shape[1]))

  for song_num in range(num_songs):
    song_idx = np.where(features[:, 0] == song_num)[0]
    song_matrix = np.take(features, song_idx, 0)
    songs[song_num, :] = np.mean(song_matrix, 0)

  return songs



##############
### k fold ###

def make_k_fold_partition(samples, k, seed = None):
  """
  :param samples: All samples that are used for training and testing
  :param targets: Targets corresponding to each sample
  :param k: Integer to decide number of partitions
  :return: partitions: 3D matrix (3D layers correspond to the partitions),
           partition_targets: matrix where each row is target for each partitions
  """
  if seed is not None:
    random.seed(seed)
  partition_size = samples.shape[0]//k
  partitions = np.zeros((partition_size, samples.shape[1], k))
  shuffle_array = [x for x in random.sample(range(samples.shape[0]), samples.shape[0])]

  # Shuffling the data
  shuffled_samples = np.take(samples, shuffle_array, 0)

  for i in range(k):
    partitions[:, :, i] = shuffled_samples[i*partition_size:(i+1)*partition_size, :]
  return partitions


def make_k_fold_partition_equal(samples, k, seed = None):
  """
  :param samples: All samples that are used for training and testing
  :param k: Number of partitions
  :param seed: The seed for the random partition
  :return: partitions: 3D matrix (3D layers correspond to the partitions),
           partition_targets: matrix where each row is target for each partitions
  """

  if seed is not None:
    np.random.seed(seed)

  partition_size = samples.shape[0]//k

  if 100 % k != 0:
    raise IndexError("Cannot create even partitions!")

  genres = np.unique(samples[:, 0]).astype("int64")

  partitions = np.zeros((partition_size, samples.shape[1], k))
  partitions_final = np.zeros((partition_size, samples.shape[1], k))

  last_filled_idx = 0

  # Shuffle the genres and place in partition
  for genre_idx in genres:
    genre_samples_idx = np.where(samples[:, 1] == genre_idx)[0]
    genre_samples = samples[genre_samples_idx]
    np.random.shuffle(genre_samples)
    genre_partitions = np.split(genre_samples, k)

    for idx, partition in enumerate(genre_partitions):
      partitions[last_filled_idx: (last_filled_idx+partition.shape[0]), :, idx] = partition

    last_filled_idx += genre_partitions[0].shape[0]

  # Shuffle each partition
  for i in range(k):
    np.random.shuffle(partitions[:, :, i])

  """ Litet test bara
    for i in range(k):
      for j in range(10):
        num_genre_in_partition = len(np.where(partitions[:, 1, i] == j)[0])
        if num_genre_in_partition != 25:
          print("FELLLLLLLLLLL", num_genre_in_partition)
  """

  return partitions


def get_k_fold_partitions(partitioned_samples, partition_num):
  """
  :param paritioned_samples: All samples partitioned into equal sized partitions (stored as 3D matrix)
  :param partition_num: The partition to be testing set
  :return: training set and test set
  """

  k = partitioned_samples.shape[2]
  N = partitioned_samples.shape[0]


  # If the samples have more than 19 columns, take the last 19 columns
  train_samples = np.zeros((N * (k - 1), partitioned_samples.shape[1]))

  j = 0
  for i in range(k):
    if i != partition_num:
      train_samples[j*N:(j+1)*N, :] = partitioned_samples[:, :, i]
      j += 1

  if partitioned_samples.shape[1] == 21:
    test_targets = partitioned_samples[:, 1, partition_num].astype("int64")
    test_samples = partitioned_samples[:, 2:, partition_num]
    train_targets = train_samples[:, 1].astype("int64")
    train_samples = train_samples[:, 2:]
  elif partitioned_samples.shape[1] == 20:
    test_samples = partitioned_samples[:, 1:, partition_num]
    test_targets = partitioned_samples[:, 0, partition_num].astype("int64")
    train_targets = train_samples[:, 0].astype("int64")
    train_samples = train_samples[:, 1:]
  else:
    raise ValueError("Wrong dimension of input. Missing targets")

  return train_samples, train_targets, test_samples, test_targets


def get_set_from_partitions(partitioned_samples, partition_num = 0):
  """
  Keeps the track number and the target
  :param partitioned_samples: Samples divided into partitions
  :return: test set (one of the partitions) and training set (the rest of the partitions)
  """

  k = partitioned_samples.shape[2]
  N = partitioned_samples.shape[0]

  test_set = partitioned_samples[:, :, partition_num]

  train_set = np.zeros((N*(k-1), partitioned_samples.shape[1]))

  j = 0
  for i in range(k):
    if i != partition_num:
      train_set[j * N:(j + 1) * N, :] = partitioned_samples[:, :, i]
      j += 1

  return train_set, test_set


##################
### Get songs directly from import ###

def get_songs_feature_set(filename):
  """
  Takes a filename and returns the feature vectors for each song
  :param filename: file for the stored features
  :return: feature vectors for each song
  """
  features = read_stored_data(filename)

  return mean_by_song(features)


def get_test_train_sets(features, partition_num = 0, seed = None):
  """
  Takes filename and return training and test set (partitioned 90-10)
  :param features:
  :return:
  """
  if isinstance(features, str):
    songs = get_songs_feature_set(features)
  else:
    songs = features

  partitions = make_k_fold_partition(songs, 10, seed)

  train_set, test_set = get_set_from_partitions(partitions, partition_num)

  return train_set, test_set



##################
## Vector calculations ##

def euclidean_dist(v1, v2):
  return np.linalg.norm(v1 - v2)


def angle(v1, v2):
  """ Computes the angle between two vectors. Returns the angle in radians"""
  v1_norm = v1/np.linalg.norm(v1)
  v2_norm = v2/np.linalg.norm(v2)
  dot_product = np.dot(v1_norm, v2_norm)
  if -1 <= dot_product <= 1:
    angle = np.arccos(dot_product)
  else:
    if dot_product > 0:
      angle = np.arccos(1.0)
    else:
      angle = np.arccos(-1.0)
  return angle


# Function for create dictionary with colors
def createColorDict():
    colorDict = {}
    colorDict[1] = "#0000ff"  # Blue
    colorDict[2] = "#000099"  # Dark blue
    colorDict[3] = "#0099cc"  # Light blue
    colorDict[4] = "#33cc33"  # Green
    colorDict[5] = "#006600"  # Dark Green
    colorDict[6] = "#99ff33"  # Light green
    colorDict[7] = "#9900ff"  # Purple
    colorDict[8] = "#6600cc"  # Dark Purple
    colorDict[9] = "#9966ff"  # Light purple
    colorDict[10] = "#cc00cc"  # Pink
    colorDict[11] = "#990099"  # Dark pink
    colorDict[12] = "#ff66ff"  # Light pink
    colorDict[13] = "#ff0000"  # Red
    colorDict[14] = "#800000"  # Dark red
    colorDict[15] = "#ff5050"  # Light red
    colorDict[16] = "#ffff00"  # Yellow
    colorDict[17] = "#996600"  # light brown
    colorDict[18] = "#ffff66"  # Light Yellow
    colorDict[19] = "#666666"  # Grey
    colorDict[20] = "#262626"  # Dark grey
    colorDict[21] = "#cccccc"  # Light Grey
    colorDict[22] = "#4d2600"  # Brown
    colorDict[23] = "#ff9900"  # Orange
    colorDict[24] = "#ff9933"  # Light orange
    colorDict[25] = "#00e6e6"  # Turquoise
    colorDict[26] = "#004d4d"  # Dark turquoise
    colorDict[27] = "#333300"  # Strange green
    colorDict[28] = "#cc3300"  # Orange/red
    colorDict[29] = "#000000"  # Black
    return colorDict
