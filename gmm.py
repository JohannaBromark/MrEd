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

if __name__ == '__main__':
  features, targets = read_stored_data()
  print('done')
  # gmm = GaussianMixture()
  # gmm.fit(features[0:5])
  # print(gmm.predict(features[5:10]))

  for i in range(30):
    plt.subplot(6,5,i+1)
    plt.axis('off')
    plt.imshow(features[i][:-1].reshape(6,3))
  plt.show()
  