from utils import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.mlab as mlab
import networkx as nx
import pygraphviz


def mfcc_only(vectors):
  filtered = [0,1,10,11,12,13,14,15,16,17,18,19]
  return vectors[:,filtered]

def filter_on_genre(vectors, genre):
  # hiphop 4, reggae 8, classical 1
  if (isinstance(genre, str)):
    return np.array([v for v in vectors if get_label(v[1]) == genre])
  else:
    return np.array([v for v in vectors if v[1] == genre])

def scatter_plot(f_genre_1, f_genre_2, compare_songs):
  f_1 = f_genre_1[:,2:]
  f_2 = f_genre_2[:,2:]
  
  mean_idx = [0,2,4,6,8]
  var_idx = [1,3,5,7,9]

  y_1 = np.concatenate((f_1[:,mean_idx].flatten('F'), f_1[:,var_idx].flatten('F')))
  y_2 = np.concatenate((f_2[:,mean_idx].flatten('F'), f_2[:,var_idx].flatten('F')))

  c = 0
  x_1 = []
  for j in range(f_1.shape[1]):
    for i in range(f_1.shape[0]):
      x_1.append(c)
    c += 2

  c = 1
  x_2 = []
  for j in range(f_2.shape[1]):
    for i in range(f_2.shape[0]):
      x_2.append(c)
    c += 2

  
  f, ax = plt.subplots()
  ax.set_xticks([i for i in range(max(x_2)+1)])
  x_ticks_labels = [
    'Coeff 0 mean', 
    'Coeff 0 mean',
    'Coeff 1 mean', 
    'Coeff 1 mean',
    'Coeff 2 mean', 
    'Coeff 2 mean',
    'Coeff 3 mean', 
    'Coeff 3 mean',
    'Coeff 4 mean', 
    'Coeff 4 mean',
    'Coeff 0 var', 
    'Coeff 0 var',
    'Coeff 1 var', 
    'Coeff 1 var',
    'Coeff 2 var', 
    'Coeff 2 var',
    'Coeff 3 var', 
    'Coeff 3 var',
    'Coeff 4 var', 
    'Coeff 4 var']

  genre_1 = get_label(f_genre_1[0,1])
  genre_2 = get_label(f_genre_2[0,1])

  ax.set_xticklabels(x_ticks_labels, rotation='80', fontsize=8)
  if compare_songs:
    plt.title('MFCCs of songs for ' + genre_1 + ' and ' + genre_2)
  else:
    plt.title('MFCCs of feature vectors for ' + genre_1 + ' and ' + genre_2)
  plt.xlabel('MFCC features')
  plt.ylabel('Feature values')

  plt.plot(x_1, y_1, '.', label= genre_1)
  plt.plot(x_2, y_2, '.', label= genre_2)
  plt.legend(loc='upper left')
  plt.tight_layout()
  
  # if compare_songs:
  #   plt.savefig('analysis_docs/mfcc_comparisons/songs/'+genre_1+'-'+genre_2)
  # else:
  #   plt.savefig('analysis_docs/mfcc_comparisons/feature_vectors/'+genre_1+'-'+genre_2)

  # plt.show()


def box_plot(f_genre_1, f_genre_2, compare_songs):

  f_1 = f_genre_1[:,2:]
  f_2 = f_genre_2[:,2:]

  idxs = [0,2,4,6,8,1,3,5,7,9]

  data = []
  for i in range(len(idxs)):
    data.append(f_1[:,idxs[i]])
    data.append(f_2[:,idxs[i]])
  
  x_ticks_labels = [
    'Coeff 0 mean', 
    'Coeff 0 mean',
    'Coeff 1 mean', 
    'Coeff 1 mean',
    'Coeff 2 mean', 
    'Coeff 2 mean',
    'Coeff 3 mean', 
    'Coeff 3 mean',
    'Coeff 4 mean', 
    'Coeff 4 mean',
    'Coeff 0 var', 
    'Coeff 0 var',
    'Coeff 1 var', 
    'Coeff 1 var',
    'Coeff 2 var', 
    'Coeff 2 var',
    'Coeff 3 var', 
    'Coeff 3 var',
    'Coeff 4 var', 
    'Coeff 4 var']

  fig1, ax1 = plt.subplots()
  # ax1.boxplot(data)
  ax1.boxplot(data, whis=[0, 100]) ### SET WHISKERS HERE or ELSE [(Q1-1.5 IQR), (Q3+1.5 IQR)]
  ax1.set_xticklabels(x_ticks_labels, rotation='80', fontsize=8)

  genre_1 = get_label(f_genre_1[0,1])
  genre_2 = get_label(f_genre_2[0,1])

  if compare_songs:
    plt.title('MFCCs of songs for ' + genre_1 + ' and ' + genre_2)
  else:
    plt.title('MFCCs of feature vectors for ' + genre_1 + ' and ' + genre_2)

  plt.xlabel('MFCC features')
  plt.ylabel('Feature values')

  plt.tight_layout()
  plt.show()


def compare_mfccs(genre_1, genre_2, compare_songs, use_box_plot):

  if compare_songs:
    f_vectors = get_songs_feature_set()
  else:
    f_vectors = read_stored_data()
    
  f_genre_1 = filter_on_genre(mfcc_only(f_vectors), genre_1)
  f_genre_2 = filter_on_genre(mfcc_only(f_vectors), genre_2)

  if use_box_plot:
    box_plot(f_genre_1, f_genre_2, compare_songs)
  else:
    scatter_plot(f_genre_1, f_genre_2, compare_songs)
  

if __name__ == "__main__":
  # box plot explanation
  # https://stackoverflow.com/questions/17725927/boxplots-in-matplotlib-markers-and-outliers
  compare_mfccs('reggae', 'hiphop', compare_songs=False, use_box_plot=True)
  