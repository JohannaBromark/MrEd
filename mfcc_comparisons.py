from utils import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.mlab as mlab
import networkx as nx
# import pygraphviz


def mfcc_only(vectors):
  filtered = [0,1,10,11,12,13,14,15,16,17,18,19]
  return vectors[:,filtered]

def filter_on_genre(vectors, genre):
  if (isinstance(genre, str)):
    return np.array([v for v in vectors if get_label(v[1]) == genre])
  else:
    return np.array([v for v in vectors if v[1] == genre])

def scatter_plot(f_genre, start_at, n_genres, label):
  f = f_genre[:,2:]

  mean_idx = [0,2,4,6,8]
  var_idx = [1,3,5,7,9]
  y = np.concatenate((f[:,mean_idx].flatten('F'), f[:,var_idx].flatten('F')))

  c = start_at+1
  x = []
  for j in range(f.shape[1]):
    for i in range(f.shape[0]):
      x.append(c)
    c += n_genres

  plt.plot(x, y, '.', label=label)

def box_plot(genres, ax):
  n_genres = len(genres)
  f = []
  for i in range(n_genres):
    f.append(genres[i][:][2:])

  # print(f)
  idxs = [0,2,4,6,8,1,3,5,7,9]
  # idxs = [0]

  data = []
  for i in range(len(idxs)):
    for j in range(n_genres):
      g = genres[j]
      data.append(g[:,idxs[i]+2])

  ax.boxplot(data, whis=[5, 95]) ### SET WHISKERS HERE or ELSE [(Q1-1.5 IQR), (Q3+1.5 IQR)]
  
def compare_mfccs(genres, compare_songs, use_box_plot):

  n_genres = len(genres)

  if compare_songs:
    f_vectors = get_songs_feature_set("features_targets/all_vectors.txt")
  else:
    f_vectors = read_stored_data()
    
  f_genres = []
  for i in range(n_genres):
    f_genres.append(filter_on_genre(mfcc_only(f_vectors), genres[i]))

  x_ticks_labels = []
  for i in range(5):
    x_ticks_labels.append(np.array(['Coeff ' + str(i) + ' mean']*n_genres))
  for i in range(5):
    x_ticks_labels.append(np.array(['Coeff ' + str(i) + ' var']*n_genres))
  
  x_ticks_labels = np.array(x_ticks_labels).flatten()
  
  fig, ax = plt.subplots()

  if use_box_plot:
    box_plot(f_genres, ax)
  else:
    for i, g in enumerate(f_genres):
      scatter_plot(g, i, n_genres, label=genres[i])

  ax.set_xticks([i for i in range(1,len(x_ticks_labels)+1)])
  ax.set_xticklabels(x_ticks_labels, rotation='80', fontsize=8)

  if compare_songs:
    title = 'MFCCs of songs for '
  else:
    title = 'MFCCs of feature vectors for '
  
  for g in genres:
    title += g + ' and '
    
  plt.title(title[:-5])

  plt.xlabel('MFCC features')
  plt.ylabel('Feature values')

  plt.tight_layout()
  plt.legend()
  plt.show()


def read_nearest_and_correct_nearest():
  data = []
  with open("features_targets/nearest_and_correct_nearest.txt", 'r') as f:
    lines = f.readlines()
    for l in lines:
      data.append(np.array([float(i) for i in l.split()]))
  return np.array(data)


def compare_nearest_and_correct_nearest(sample_n):

  data = read_nearest_and_correct_nearest()

  all_songs = get_songs_feature_set("features_targets/all_vectors.txt")
  all_texture_windows = read_stored_data()

  sample_windows = all_texture_windows[np.where(all_texture_windows[:,0] == data[sample_n,0])]

  sample = all_songs[np.where(all_songs[:,0] == data[sample_n,0])]
  nearest = all_songs[np.where(all_songs[:,0] == data[sample_n,2])]
  correct_nearest = all_songs[np.where(all_songs[:,0] == data[sample_n,3])]

  sample_windows = mfcc_only(sample_windows)
  sample = mfcc_only(sample)
  nearest = mfcc_only(nearest)
  correct_nearest = mfcc_only(correct_nearest)

  idxs = [0,2,4,6,8,1,3,5,7,9]

  fig, ax = plt.subplots()
  box_plot([sample_windows], ax)

  song_names = read_content()
  sample_name = song_names[int(data[sample_n,0])]
  nearest_name = song_names[int(data[sample_n,2])]
  correct_nearest_name = song_names[int(data[sample_n,3])]

  # scatter_plot(sample, 0, 1, label='Sample: ' + str(int(sample[0,0])) + ' (' + get_label(sample[0,1]) + ')')
  # scatter_plot(nearest, 0, 1, label='Nearest: ' + str(int(nearest[0,0])) + ' (' + get_label(nearest[0,1]) + ')')
  # scatter_plot(correct_nearest, 0, 1, label='Correct nearest: ' + str(int(correct_nearest[0,0])) + ' (' + get_label(correct_nearest[0,1]) + ')')

  scatter_plot(sample, 0, 1, label='Sample: ' + sample_name[0] + ' ' + sample_name[1] + ' ' + sample_name[2])
  scatter_plot(nearest, 0, 1, label='Nearest: ' + nearest_name[0] + ' ' + nearest_name[1] + ' ' + nearest_name[2])
  scatter_plot(correct_nearest, 0, 1, label='Correct nearest: ' + correct_nearest_name[0] + ' ' + correct_nearest_name[1] + ' ' + correct_nearest_name[2])
  

  x_ticks_labels = []
  for i in range(5):
    x_ticks_labels.append('Coeff ' + str(i) + ' mean')
  for i in range(5):
    x_ticks_labels.append('Coeff ' + str(i) + ' var')
  ax.set_xticks([i for i in range(1,len(x_ticks_labels)+1)])
  ax.set_xticklabels(x_ticks_labels, rotation='80', fontsize=8)
  
  plt.title('MFCC values for texture windows of song ' + sample_name[0])
  plt.tight_layout()
  plt.legend()
  plt.show()

def plot_single_mfccs(genres, mfcc_number):
  data = get_songs_feature_set("features_targets/all_vectors.txt")
  g_idx = 1

  n_genres = np.unique(data[:,g_idx]).shape[0]

  for i in range(n_genres):
    if get_label(i) in genres or 'all' in genres:
      genre_idxs = np.where(data[:,g_idx] == i)[0]
      plt.plot(data[genre_idxs,10+mfcc_number], data[genre_idxs,11+mfcc_number], '.', label=get_label(i))
    
  plt.title('MFCC ' + str(mfcc_number) + ' of songs')
  plt.xlabel('Mean')
  plt.ylabel('Variance')
  plt.legend()
  plt.show()
  
  

if __name__ == "__main__":

  # box plot explanation
  # https://stackoverflow.com/questions/17725927/boxplots-in-matplotlib-markers-and-outliers

  ### Compare MFCC box plots
  compare_mfccs(genres=['classical', 'hiphop'], compare_songs=True, use_box_plot=True)

  ### MFCC box plots of a misclassified song with neighbour and the nearest correct
  # compare_nearest_and_correct_nearest(sample_n = 0)

  ### Plot single MFCC values for all songs
  # plot_single_mfccs(genres=['reggae', 'hiphop', 'all'], mfcc_number=1)
  