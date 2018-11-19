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

def compare_mfccs(genre_1, genre_2):
  f_vectors = read_stored_data()
  f_genre_1 = filter_on_genre(mfcc_only(f_vectors), genre_1)
  f_genre_2 = filter_on_genre(mfcc_only(f_vectors), genre_2)

  f_1 = normalise(f_genre_1[:,2:])[0]
  f_2 = normalise(f_genre_2[:,2:])[0]
  
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
    'G1, coeff 0 means', 
    'G2, coeff 0 means',
    'G1, coeff 1 means', 
    'G2, coeff 1 means',
    'G1, coeff 2 means', 
    'G2, coeff 2 means',
    'G1, coeff 3 means', 
    'G2, coeff 3 means',
    'G1, coeff 4 means', 
    'G2, coeff 4 means',
    'G1, coeff 0 vars', 
    'G2, coeff 0 vars',
    'G1, coeff 1 vars', 
    'G2, coeff 1 vars',
    'G1, coeff 2 vars', 
    'G2, coeff 2 vars',
    'G1, coeff 3 vars', 
    'G2, coeff 3 vars',
    'G1, coeff 4 vars', 
    'G2, coeff 4 vars']

  ax.set_xticklabels(x_ticks_labels, rotation='80', fontsize=8)
  plt.title('MFCC comparison of feature vectors between ' + genre_1 + ' and ' + genre_2)
  plt.xlabel('MFCC features')
  plt.ylabel('Feature values')
  plt.plot(x_1, y_1, '.', label='G1: ' + genre_1 + ' feature vectors')
  plt.plot(x_2, y_2, '.', label='G2: ' + genre_2 + ' feature vectors')
  plt.legend(loc='upper left')
  plt.tight_layout()
  plt.show()

if __name__ == "__main__":
  compare_mfccs('metal', 'rock')
  