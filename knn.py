from sklearn.neighbors import KNeighborsClassifier
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from timbral_texture import get_label
import matplotlib.pyplot as plt
import numpy as np
from utils import *
  
if __name__ == '__main__':
  features, targets = read_stored_data('features_targets/afe_feat_and_targ.txt')
  # features = normalise(features)
  features_mean = mean_by_song(features)
  grouped_targets = features_mean[:,0]
  features_mean = features_mean [:,1:]


  grouped_targets = np.array(grouped_targets)

  b = 0

  for i in range(100):
    train_samples, test_samples, train_targets, test_targets = train_test_split(features_mean, grouped_targets, test_size=0.1, random_state=i)

    knn = KNeighborsClassifier(n_neighbors=1)

    knn.fit(train_samples,train_targets)

    predictions = knn.predict(test_samples)
    score = knn.score(test_samples,test_targets)
    b = b + score
    print(score)
  print(b/100)

