import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from utils import *

def main():
    dir = "features_targets/"

    # Read the data
    random_train_1_vec = read_stored_data(dir + "random_filtered_vectors_train.txt")
    random_train_2_vec = read_stored_data(dir + "random_filtered_vectors_valid.txt")
    random_test_vec = read_stored_data(dir + "random_filtered_vectors_test.txt")

    fault_train_1_vec = read_stored_data(dir + "fault_filtered_vectors_train.txt")
    fault_train_2_vec = read_stored_data(dir + "fault_filtered_vectors_valid.txt")
    fault_test_vec = read_stored_data(dir + "fault_filtered_vectors_test.txt")

    # Concat training set and validation set
    random_train_vec = np.concatenate((random_train_1_vec, random_train_2_vec))
    fault_train_vec = np.concatenate((fault_train_1_vec, fault_train_2_vec))

    # Create the song samples
    random_train = mean_by_song(random_train_vec)
    random_test = mean_by_song(random_test_vec)
    fault_train = mean_by_song(fault_train_vec)
    fault_test = mean_by_song(fault_test_vec)

    # Take only shared samples as test set
    random_song_nr = random_test[:, 0]
    fault_song_nr = fault_test[:, 0]
    shared_song_nr, random_indices, fault_indices = np.intersect1d(random_song_nr, fault_song_nr, return_indices=True)
    shared_test = random_test[random_indices]

    # Normalise the data
    random_train_norm, random_mean, random_std = normalise(random_train[:, 10:20])
    random_test_norm = ((shared_test[:, 10:20] - random_mean) / random_std)
    fault_train_norm, fault_mean, fault_std = normalise(fault_train[:, 10:20])
    fault_test_norm = ((shared_test[:, 10:20] - fault_mean) / fault_std)

    # Extract the features
    random_features_train = random_train[:, 1]
    random_features_test = random_test[:, 1]
    fault_features_train = fault_train[:, 1]
    fault_features_test = fault_test[:, 1]
    shared_features_test = shared_test[:, 1]

    # Train two gmm models and predict the test samples
    random_scores = np.zeros((random_test_norm.shape[0], 10))
    fault_scores = np.zeros((fault_test_norm.shape[0], 10))
    for i in range(10):
        random_predictor = GaussianMixture(
            n_components=1,
            covariance_type='full',
            tol=0.000001,
            max_iter=500,
            n_init=2,
            init_params='kmeans')

        fault_predictor = GaussianMixture(
            n_components=1,
            covariance_type='full',
            tol=0.000001,
            max_iter=500,
            n_init=2,
            init_params='kmeans')

        random_predictor.fit(random_train_norm[random_features_train == i])
        fault_predictor.fit(fault_train_norm[fault_features_train == i])

        random_scores[:, i] = random_predictor.score_samples(random_test_norm)
        fault_scores[:, i] = fault_predictor.score_samples(fault_test_norm)

    # Find the model prediction
    gmm_random_prediction = np.argmax(random_scores, axis=1)
    gmm_fault_prediction = np.argmax(fault_scores, axis=1)

    # Train a knn model

    random_knn = KNeighborsClassifier(n_neighbors=1)
    fault_knn = KNeighborsClassifier(n_neighbors=1)

    random_knn.fit(random_train_norm, random_features_train)
    fault_knn.fit(fault_train_norm, fault_features_train)

    knn_random_prediction = random_knn.predict(random_test_norm)
    knn_fault_prediction = fault_knn.predict(fault_test_norm)

    # Save the result
    gmm_random_result = []
    gmm_fault_result = []
    knn_random_result = []
    knn_fault_result = []

    for i in range(len(gmm_random_prediction)):
        if gmm_random_prediction[i] == shared_test[i, 1]:
            gmm_random_result.append("True "+str(int(gmm_random_prediction[i])))
        else:
            gmm_random_result.append("False "+str(int(gmm_random_prediction[i])))
        if gmm_fault_prediction[i] == shared_test[i, 1]:
            gmm_fault_result.append("True "+str(int(gmm_fault_prediction[i])))
        else:
            gmm_fault_result.append("False "+str(int(gmm_fault_prediction[i])))

        if knn_random_prediction[i] == shared_test[i, 1]:
            knn_random_result.append("True "+str(int(knn_random_prediction[i])))
        else:
            knn_random_result.append("False "+str(int(knn_random_prediction[i])))

        if knn_fault_prediction[i] == shared_test[i, 1]:
            knn_fault_result.append("True "+str(int(knn_fault_prediction[i])))
        else:
            knn_fault_result.append("False "+str(int(knn_fault_prediction[i])))

    # Save the data to a csv so it can be imported to a sheet

    column_names = [str(song) for song in shared_song_nr]
    row_names = ["Random gmm", "Fault gmm", "Random knn", "Fault knn"]
    results = [gmm_random_result, gmm_fault_result, knn_random_result, knn_fault_result]

    with open("analysis_docs/knn1_gmm1_table_mfcc.csv", "w") as file:
        file.write(",")
        for column_name in column_names:
            file.write(column_name + ",")
        file.write("\n")
        for i in range(len(row_names)):
            file.write(row_names[i]+",")
            for j in range(len(column_names)):
                file.write(str(results[i][j])+",")
            file.write("\n")
    pass


if __name__ == '__main__':
    main()
