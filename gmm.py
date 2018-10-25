from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import numpy as np

def read_stored_data():
  """Return feature vectors and corr labels from stored txt file"""
  with open('features.txt') as f:
    lines = f.readlines()
    features = [[0]] * len(lines)
    for i in range(len(lines)):
      features[i] = [float(i) for i in lines[i].split()]
    features = np.array(features)

  with open('targets.txt') as f:
    targets = np.array([[int(i)] for i in f.readlines()])

  return features, targets

def normalise(features):
  for i in range(len(features[1])):
    features[:,i] = np.interp(features[:,i], (features[:,i].min(), features[:,i].max()), (0, 1))
  return features

if __name__ == '__main__':
  features, targets = read_stored_data()

  norm_features = normalise(features)

  c = 1
  for i in range(10):
    plt.subplot(2,10,c)
    plt.axis('off')
    plt.imshow(features[100*i][:-1].reshape(6,3))
    c += 1
    plt.subplot(2,10,c)
    plt.axis('off')
    plt.imshow(features[101*i][:-1].reshape(6,3))
    c += 1
  # plt.show()

  gmm = GaussianMixture()
  gmm.fit(features)
  print(gmm.predict(features))
  