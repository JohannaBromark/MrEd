from utils import *
from feature_analysis import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.mlab as mlab

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
	allDist = read_stored_data('features_targets/Alldistances.txt')
	allDist = np.array(allDist)

	alldistNoDiag = remove_diagonal(np.copy(allDist))

    missclassified = get_missclassified_with_neighbors_nearest_and_correct(alldistNoDiag)

	f_vectors = missclassified
	
	song_idx = []
	for i in range(0,19,3)
		song_idx = np.append(song_idx,i)
		missclass_idx = np.append(missclass_idx,i+1)
		correctclass_idx = np.append(correctclass_idx,i+2)
	


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

	ax.set_xticklabels(x_ticks_labels, rotation='80', fontsize=8)
	if compare_songs:
		plt.title('MFCCs of songs for ' + genre_1 + ' and ' + genre_2)
	else:
		plt.title('MFCCs of feature vectors for ' + genre_1 + ' and ' + genre_2)
	plt.xlabel('MFCC features')
	plt.ylabel('Feature values')

	# plt.ylim(-10,10)

	plt.plot(x_1, y_1, '.', label= genre_1)
	plt.plot(x_2, y_2, '.', label= genre_2)
	plt.legend(loc='upper left')
	plt.tight_layout()
	
	if compare_songs:
		plt.savefig('analysis_docs/mfcc_comparisons/songs/'+genre_1+'-'+genre_2)
	else:
		plt.savefig('analysis_docs/mfcc_comparisons/feature_vectors/'+genre_1+'-'+genre_2)

	plt.show()

if __name__ == "__main__":
  	compare_mfccs('hiphop', 'reggae', compare_songs=True)
  