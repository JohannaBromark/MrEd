import numpy as np
import os
from pyAudioAnalysis import audioFeatureExtraction as afe
from pyAudioAnalysis import audioBasicIO
from utils import *

def feature_vector(frames):
  feat_idx = [3,7,6,0,8,9,10,11,12]
  vec = []
  mean = np.sum(frames, axis=1)/frames.shape[1]
  var = np.var(frames, axis=1)

  for i in feat_idx:
    vec.append(mean[i])
    vec.append(var[i])
  vec.append(rms_energy(frames))

  return np.array(vec)

def all_feature_vectors(all_frames, size=43):
  vectors = []
  for i in range(all_frames.shape[1]//size):
    vectors.append(feature_vector(all_frames[:,i*size:(i+1)*size]))
  return np.array(vectors)

def rms_energy(frames):
  e_idx = 1
  mean = np.sum(frames[e_idx,:])/frames.shape[1]
  percentage = np.count_nonzero(np.where(frames[e_idx,:] > mean, 1, 0))/frames.shape[1]
  return percentage

if __name__ == '__main__':
  samples, targets = read_directories()
  seg_size = 512
  sample_rate = 22050
  f_vectors = []

  for i in range(len(samples)):
    print(i)
    frames, f_names = afe.stFeatureExtraction(samples[i], sample_rate, seg_size, seg_size)
    f_vectors.append(all_feature_vectors(frames))

  # write_afe_to_file(f_vectors, targets)
