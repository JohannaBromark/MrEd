from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis import audioFeatureExtraction as audioFE
import matplotlib.pyplot as plt
import numpy as np
import numpy.fft as fft
import scipy.signal as signal
from sklearn.preprocessing import normalize

def read_file(file_name='genres/rock/rock.00093.wav'):
  """Return 22050 Hz sampling frequency and sample amplitudes"""

  # pop: genres/pop/pop.00000.wav
  return audioBasicIO.readAudioFile(file_name)

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
  """Return time domain zero crossings for an analysis window in the time domain"""
  
  signed = np.where(samples[wndw_no*seg_size:(wndw_no+1)*seg_size] > 0, 1, 0)
  return np.sum([np.abs(signed[i]-signed[i-1]) for i in range(1, len(signed))])

def mfcc_coeffs(an_wndw):
  pass

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
def MVzero_crossing(stft_wndws,t_wndw_size):
  crossing = []
  for i in range(t_wndw_size+1):
    crossing = np.append(crossing, time_zero_crossings(stft_wndws[i]))

  mean = np.sum(crossing)/t_wndw_size
  var = 0
  for k in range(t_wndw_size):
    var = var + (crossing[k]-mean)
  var/(t_wndw_size-1)
  return mean, var


if __name__ == '__main__':
  sample_rate, samples = read_file()
 
  # Check if params are correct
  # Include overlap? Praxis is to use default overlap setting
  # nperseg -> length of each segment (also number of frequencies per seg) should be *2 for some reason?
  seg_size = 512
  freqs, time_inits, stft_wndws = signal.stft(samples, fs=sample_rate, nperseg=seg_size, noverlap=0)
 
  wndw_no = 1
  an_wndws = np.abs(stft_wndws) # abs -> we only want freq amplitudes
  an_wndw = an_wndws[:,wndw_no] # col -> analysis window
  
  t_wndw_size = 43
  mean_centroids = []
  var_centroids = []

  mean_rolloffs = []
  var_rolloffs = []

  mean_fluxs = []
  var_fluxs = []

  mean_crossings = []
  var_crossings = []

  for i in range(0, 1294, t_wndw_size):
    mean_centroid, var_centroid = MVcentroid(an_wndws[:,i:i+t_wndw_size],freqs,t_wndw_size)
    mean_centroids = np.append(mean_centroids, mean_centroid)
    var_centroids = np.append(var_centroids, var_centroid)

    mean_rolloff, var_rolloff = MVrolloffs(an_wndws[:,i:i+t_wndw_size],t_wndw_size)
    mean_rolloffs = np.append(mean_rolloffs, mean_rolloff)
    var_rolloffs = np.append(var_rolloffs, var_rolloff)

    mean_flux, var_flux = MVflux(an_wndws[:,i:i+t_wndw_size], t_wndw_size)
    mean_fluxs = np.append(mean_fluxs, mean_flux)
    var_fluxs = np.append(var_fluxs, var_flux)

    mean_crossing, var_crossing = MVzero_crossing(stft_wndws[:,i:i+t_wndw_size],t_wndw_size)
    mean_crossings = np.append(mean_crossings, mean_crossing)
    var_crossings = np.append(var_crossings, var_crossing)
    

  print(mean_centroids)
  print(var_centroids)
  print(mean_rolloffs)
  print(var_rolloffs)
  print(mean_fluxs)
  print(var_fluxs)
  print(mean_crossings)
  print(var_crossings)
  # centroid = spectral_centroid(an_wndw, freqs)
  # rolloff = spectral_rolloff(an_wndw) # nåt lurt med denna, vafan betyder ens output 
  # flux = spectral_flux(an_wndw, an_wndws[:,wndw_no-1])
  # zero_crossings = time_zero_crossings(stft_wndws[wndw_no])





  wndw_no = 1
  an_wndws = np.abs(stft_wndws) # abs -> we only want freq amplitudes
  an_wndw = an_wndws[:,wndw_no] # col -> analysis window

  # Parameters for mfcc computation
  an_wndw_size = an_wndw.shape[0]
  [filter_bank, _] = audioFE.mfccInitFilterBanks(sample_rate, an_wndw_size)

  centroid = spectral_centroid(an_wndw, freqs)
  rolloff = spectral_rolloff(an_wndw) # nåt lurt med denna, vafan betyder ens output 
  flux = spectral_flux(an_wndw, an_wndws[:,wndw_no-1])
  zero_crossings = time_zero_crossings(wndw_no, samples, seg_size)
  mfcc = audioFE.stMFCC(an_wndw, filter_bank, 5)



