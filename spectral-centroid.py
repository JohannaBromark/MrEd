from pyAudioAnalysis import audioBasicIO
import matplotlib.pyplot as plt
import numpy as np
import numpy.fft as fft
import scipy.signal as signal

def read_files(isRock=True):
  # 30 sec klipp
  [Fs_rock, x_rock] = audioBasicIO.readAudioFile("genres/rock/rock.00093.wav")
  [Fs_pop, x_pop] = audioBasicIO.readAudioFile("genres/pop/pop.00000.wav")
  samples = x_rock if isRock else x_pop
  sample_rate = Fs_rock if isRock else Fs_pop
  return samples, sample_rate

def plot_fft(samples, sample_rate):
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
  plt.ylabel("Amplitude")
  plt.xlabel("Frequency [Hz]")

def plot_stft(samples, sample_rate, wnd_no, centroid):
  # STFT
  f, t, Zxx = signal.stft(samples, fs=sample_rate, nperseg=512*2, noverlap=0)
  plt.pcolormesh(t, f, np.abs(Zxx))
  plt.title('STFT Magnitude')
  plt.ylabel('Frequency [Hz]')
  plt.xlabel('Time [sec]')

def plot_window(window, f, centroid_idx=None):
  plt.title('Window FFT')
  plt.ylabel("Amplitude")
  plt.xlabel("Frequency [Hz]")
  plt.plot(f, wnd)
  if centroid_idx:
    plt.plot(f[centroid_idx], 0, 'o', label='centroid')

def spectral_centroid_idx(texture_window):
  """Returns index of spectral centroid frequency"""
  wnd = np.abs(texture_window)
  bin_sum = np.sum(wnd*[i for i in range(1, len(texture_window)+1)])
  mag_sum = np.sum(wnd)
  return int(round(bin_sum/mag_sum, 0)) - 1

def spectral_rolloff(texture_window):
  return 0.85*np.sum(np.abs(texture_window))

if __name__ == '__main__':
  samples, sample_rate = read_files()
  f, t, Zxx = signal.stft(samples, fs=sample_rate, nperseg=512*2, noverlap=0)
  Zxx = np.abs(Zxx)

  for i in range(5, 20):
    wnd = Zxx[:,i]
    s_c_index = spectral_centroid_idx(wnd)
    plot_window(wnd, f, centroid_idx=s_c_index) 
    plt.show()







