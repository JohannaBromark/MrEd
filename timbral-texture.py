from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis import audioFeatureExtraction as audioFE
import matplotlib.pyplot as plt
import numpy as np
import numpy.fft as fft
import scipy.signal as signal
from sklearn.preprocessing import normalize
from sklearn.mixture import GaussianMixture
import os

def read_file(file_name='genres/rock/rock.00093.wav'):
  """Return 22050 Hz sampling frequency and sample amplitudes"""
  # pop: genres/pop/pop.00000.wav
  return audioBasicIO.readAudioFile(file_name)
def read_file2(file_name='genres/pop/pop.00050.wav'):
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
  return all_samples, labels

def read_directories():
  all_songs = []
  all_labels = []
  i = 0
  for name in os.listdir('genres/'):
    songs, labels = read_directory(name)
    all_songs[i*100:(i+1)*100] = songs
    all_labels[i*100:(i+1)*100] = labels
    i += 1
  all_songs = np.array(all_songs)
  all_labels = np.array(all_labels)
  return all_songs, all_labels

def get_label(genre='rock'):
  if genre == 'blues':
    return 0
  elif genre == 'classical':
    return 1
  elif genre == 'country':
    return 2
  elif genre == 'disco':
    return 3
  elif genre == 'hiphop':
    return 4
  elif genre == 'jazz':
    return 5
  elif genre == 'metal':
    return 6
  elif genre == 'pop':
    return 7
  elif genre == 'reggae':
    return 8
  elif genre == 'rock':
    return 9
  else:
    return False
  
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

def spectral_centroid_idx(an_wndw):
  """Return index of spectral centroid frequency of an analysis window"""
  bin_sum = np.sum(an_wndw*[i for i in range(1, len(an_wndw)+1)])
  mag_sum = np.sum(an_wndw)
  return int(round(bin_sum/mag_sum, 0)) - 1

def spectral_centroid(an_wndw, freqs):
  """Return spectral centroid frequency of an analysis window"""
  return freqs[spectral_centroid_idx(an_wndw)]

def spectral_rolloff(an_wndw):
  """Return the spectral rolloff in an analysis window"""
  return 0.85*np.sum(np.abs(an_wndw))

def spectral_flux(an_wndw, prev_wndw):
  """Return the spectral flux of an analysis window
  
  :param prev_wndw: The analysis window one time step prior to an_wndw
  """
  an_wndw *= 1./np.max(an_wndw, axis=0)
  prev_wndw *= 1./np.max(prev_wndw, axis=0)
  return np.sum(np.power(an_wndw-prev_wndw, 2))

def time_zero_crossings(wndw_no, samples, seg_size):
  """Return time domain zero crossings for an analysis window"""
  signed = np.where(samples[wndw_no*seg_size:(wndw_no+1)*seg_size] > 0, 1, 0)
  return np.sum([np.abs(signed[i]-signed[i-1]) for i in range(1, len(signed))])

def mfcc_coeffs(an_wndw, sample_rate):
  """Return the five first mfcc coefficients"""
  an_wndw_size = an_wndw.shape[0]
  [filter_bank, _] = audioFE.mfccInitFilterBanks(sample_rate, an_wndw_size)
  return audioFE.stMFCC(an_wndw, filter_bank, 5)

def rms_energy(wndw_no, samples, seg_size):
  """Return the RMS energy of an analysis window"""
  energy = [np.power(i, 2) for i in samples[wndw_no*seg_size:(wndw_no+1)*seg_size]]
  return np.square(np.sum(energy) * 1/seg_size)

#Mean and variance of the centroids
def MVcentroid(an_wndws,freqs,t_wndw_size):
  centroids = []
  for i in range(t_wndw_size):
    centroids = np.append(centroids, spectral_centroid(an_wndws[i],freqs))

  mean = np.sum(centroids)/t_wndw_size
  var = 0
  for k in range(t_wndw_size):
    var = var + (centroids[k]-mean)
  var/(t_wndw_size-1)
  return mean, var

#Mean and variance of the rolloffs
def MVrolloffs(an_wndws,t_wndw_size):
  rolloffs = []
  for i in range(t_wndw_size):
    rolloffs = np.append(rolloffs, spectral_rolloff(an_wndws[i]))

  mean = np.sum(rolloffs)/t_wndw_size
  var = 0
  for k in range(t_wndw_size):
    var = var + (rolloffs[k]-mean)
  var/(t_wndw_size-1)
  return mean, var

#Mean and variance of the MVflux
def MVflux(an_wndws,t_wndw_size):
  flux = []
  for i in range(1,t_wndw_size+1,1):
    flux = np.append(flux, spectral_flux(an_wndws[i],an_wndws[i-1]))

  mean = np.sum(flux)/t_wndw_size
  var = 0
  for k in range(t_wndw_size):
    var = var + (flux[k]-mean)
  var/(t_wndw_size-1)
  return mean, var

#Mean and varaince of the zero_crossings
def MVzero_crossing(start,samples,seg_size,t_wndw_size):
  crossing = []
  for i in range(start,t_wndw_size+start,1):
    crossing = np.append(crossing, time_zero_crossings(i,samples,seg_size))

  mean = np.sum(crossing)/t_wndw_size
  var = 0
  for k in range(t_wndw_size):
    var = var + (crossing[k]-mean)
  var/(t_wndw_size-1)
  return mean, var

def MeVaCentroid(an_wndws,freqs,t_wndw_size,nr_wndws):
  mean_centroids = []
  var_centroids = []
  for i in range(0, nr_wndws, t_wndw_size):
    mean_centroid, var_centroid = MVcentroid(an_wndws[:,i:i+t_wndw_size],freqs,t_wndw_size)
    mean_centroids = np.append(mean_centroids, mean_centroid)
    var_centroids = np.append(var_centroids, var_centroid)
  return mean_centroids, var_centroids

def MeVaRolloffs(an_wndws,t_wndw_size,nr_wndws):
  mean_rolloffs = []
  var_rolloffs = []
  for i in range(0, nr_wndws, t_wndw_size):
    mean_rolloff, var_rolloff = MVrolloffs(an_wndws[:,i:i+t_wndw_size],t_wndw_size)
    mean_rolloffs = np.append(mean_rolloffs, mean_rolloff)
    var_rolloffs = np.append(var_rolloffs, var_rolloff)
  return mean_rolloffs, var_rolloffs

def MeVaFlux(an_wndws,t_wndw_size,nr_wndws):
  mean_fluxs = []
  var_fluxs = []
  for i in range(0, nr_wndws, t_wndw_size):
    mean_flux, var_flux = MVflux(an_wndws[:,i:i+t_wndw_size], t_wndw_size)
    mean_fluxs = np.append(mean_fluxs, mean_flux)
    var_fluxs = np.append(var_fluxs, var_flux)
  return mean_fluxs, var_fluxs

def MeVaZero_Crossings(samples,seg_size,t_wndw_size,nr_wndws):
  mean_crossings = []
  var_crossings = []
  for i in range(0, nr_wndws, t_wndw_size):
    mean_crossing, var_crossing = MVzero_crossing(i,samples,seg_size,t_wndw_size)
    mean_crossings = np.append(mean_crossings, mean_crossing)
    var_crossings = np.append(var_crossings, var_crossing)
  return mean_crossings, var_crossings
    
def MVmfcc(an_wndws, sample_rate, t_wndw_size):
  mfccs = []
  for i in range(t_wndw_size):
    mfccs = np.append(mfccs, mfcc_coeffs(an_wndws[i],sample_rate))
  mfccs = mfccs.reshape(t_wndw_size,5)
  # print(mfccs)
  mean = []
  for i in range(5):
    mean = np.append(mean, np.sum(mfccs[:,i])/t_wndw_size)
  var = []
  for k in range(5):
    variance = 0
    for a in range(t_wndw_size):
      variance = variance + (mfccs[a,k]-mean[k])
    var = np.append(var,variance)
  var/(t_wndw_size-1)
  return mean, var

def MeVaMfcc(an_wndws,sample_rate,t_wndw_size,nr_wndws):
  mean_mfccs = []
  var_mfccs = []
  for i in range(0, nr_wndws, t_wndw_size):
    mean_mfcc, var_mfcc = MVmfcc(an_wndws[:,i:i+t_wndw_size],sample_rate,t_wndw_size)
    mean_mfccs = np.append(mean_mfccs, mean_mfcc)
    var_mfccs = np.append(var_mfccs, var_mfcc)

  rshape = int(mean_mfccs.size/5)
  mean_mfccs = mean_mfccs.reshape(rshape,5) #TROR DET ÄR RÄTT RESHAPE
  var_mfccs = var_mfccs.reshape(rshape,5)
  
  return mean_mfccs, var_mfccs

def MVenergy(start,samples,seg_size,t_wndw_size):
  energy = []
  for i in range(start,t_wndw_size+start,1):
    energy = np.append(energy, rms_energy(i,samples,seg_size))

  mean = np.sum(energy)/t_wndw_size

  count = 0
  for i in range(t_wndw_size):
    if energy[i] < mean:
      count += 1
  return count/t_wndw_size

def MeVaEnergy(samples,seg_size,t_wndw_size,nr_wndws):
  mean_rms_energys = []
  
  for i in range(0, nr_wndws, t_wndw_size):
    mean_rms_energy = MVenergy(i,samples,seg_size,t_wndw_size)
    mean_rms_energys = np.append(mean_rms_energys, mean_rms_energy)
  return mean_rms_energys

def CreateFeatureVectors(seg_size,samples,sample_rate,an_wndws,freqs,t_wndw_size,nr_wndws):
  mean_centroids, var_centroids = MeVaCentroid(an_wndws, freqs, t_wndw_size,nr_wndws)
  mean_rolloffs, var_rolloffs = MeVaRolloffs(an_wndws,t_wndw_size,nr_wndws)
  mean_fluxs, var_fluxs = MeVaFlux(an_wndws, t_wndw_size,nr_wndws)
  mean_crossings, var_crossings = MeVaZero_Crossings(samples,seg_size, t_wndw_size,nr_wndws)
  mean_mfccs, var_mfccs = MeVaMfcc(an_wndws,sample_rate,t_wndw_size,nr_wndws) #31 texture windows, 5 olika MFCSS i varje rad.
  mean_rms_energy = MeVaEnergy(samples,seg_size,t_wndw_size,nr_wndws)

  featureVector = np.zeros(19)
  featureMatrix = []
  for i in range(mean_centroids.size):
    featureVector[0] = mean_centroids[i]
    featureVector[1] = var_centroids[i]
    featureVector[2] = mean_rolloffs[i]
    featureVector[3] = var_rolloffs[i]
    featureVector[4] = mean_fluxs[i]
    featureVector[5] = var_fluxs[i]
    featureVector[6] = mean_crossings[i]
    featureVector[7] = var_crossings[i]
    featureVector[8] = mean_mfccs[i,0]
    featureVector[9] = mean_mfccs[i,1]
    featureVector[10] = mean_mfccs[i,2]
    featureVector[11] = mean_mfccs[i,3]
    featureVector[12] = mean_mfccs[i,4]
    featureVector[13] = mean_mfccs[i,0]
    featureVector[14] = var_mfccs[i,1]
    featureVector[15] = var_mfccs[i,2]
    featureVector[16] = var_mfccs[i,3]
    featureVector[17] = var_mfccs[i,4]
    featureVector[18] = mean_rms_energy[i]
    featureMatrix = np.append(featureMatrix,featureVector)
  featureMatrix = featureMatrix.reshape(int(nr_wndws/43),19)
  # print(featureMatrix.shape)
  # print(featureMatrix)

  return featureMatrix
  
def createAll(all_samples,labels):
  seg_size = 512
  t_wndw_size = 43
  
  featureMatrix = np.zeros((2,19))
  labelsMatrix = []
  for i in range(1000):
    try:
      print(i)
      freqs, time_inits, stft_wndws = signal.stft(all_samples[i], fs=sample_rate, nperseg=seg_size, noverlap=0)
      an_wndws = np.abs(stft_wndws)
      nr_wndws = int(((samples.size/512)//43)*43)
      nr_t_wndws = int(nr_wndws/43)

      featureMatrix = np.concatenate((featureMatrix ,CreateFeatureVectors(seg_size,all_samples[i],sample_rate,an_wndws,freqs,t_wndw_size,nr_wndws)),axis =0)
      targets = np.zeros(nr_t_wndws)
      targets[0:nr_t_wndws] = labels[i][0]
      labelsMatrix = np.append(labelsMatrix,targets)
    except:
      print('Någt gick snett till')
      print(i)
      

    

  labelsMatrix = labelsMatrix.reshape(labelsMatrix.size,1)
  featureMatrix = featureMatrix[2:]
  return featureMatrix, labelsMatrix

if __name__ == '__main__':
  sample_rate, samples = read_file()
  all_samples, labels = read_directories()

  # Check if params are correct
  # Include overlap? Praxis is to use default overlap setting
  # nperseg -> length of each segment (also number of frequencies per seg) should be *2 for some reason?
  # freqs, time_inits, stft_wndws = signal.stft(samples, fs=sample_rate, nperseg=seg_size, noverlap=0)
  # an_wndws = np.abs(stft_wndws) # abs -> we only want freq amplitudes
  # an_wndw = an_wndws[:,wndw_no] # col -> analysis window
 
  features, targets = createAll(all_samples,labels)
  # with open('features.txt','w') as file:
  #   for item in features:
  #     for element in item:
  #       file.write(str(element))
  #       file.write(' ')
  #     file.write('\n')

  # with open('targets.txt', 'w') as file:
  #   for item in targets:
  #     file.write(str(int(item[0])))
  #     file.write('\n')
  
  # gmm = GaussianMixture(n_components=3)
  # gmm.fit(featureMatrix)
  # print(gmm.predict(featureMatrix))


  # print(mean_centroids.shape)
  # print(var_centroids)
  # print(mean_rolloffs)
  # print(var_rolloffs)
  # print(mean_fluxs)
  # print(var_fluxs)
  # print(mean_crossings)
  # print(var_crossings)

  # centroid = spectral_centroid(an_wndw, freqs)
  # rolloff = spectral_rolloff(an_wndw) # nåt lurt med denna, vafan betyder ens output 
  # flux = spectral_flux(an_wndw, an_wndws[:,wndw_no-1])
  # zero_crossings = time_zero_crossings(wndw_no, samples, seg_size)
  # mfcc = mfcc_coeffs(an_wndw, sample_rate)

  # print(mfcc.shape)

  # print(rms_energy(wndw_no, samples, seg_size))


