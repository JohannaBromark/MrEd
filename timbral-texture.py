from pyAudioAnalysis import audioBasicIO
import matplotlib.pyplot as plt
import numpy as np
import numpy.fft as fft
import scipy.signal as signal

def read_files(isRock=True):
  """Read audio file amplitudes with set sample frequency"""
  # 30 sec klipp
  [Fs_rock, x_rock] = audioBasicIO.readAudioFile("genres/rock/rock.00093.wav")
  [Fs_pop, x_pop] = audioBasicIO.readAudioFile("genres/pop/pop.00000.wav")
  samples = x_rock if isRock else x_pop
  sample_rate = Fs_rock if isRock else Fs_pop
  return samples, sample_rate

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
  plt.ylabel("Amplitude")
  plt.xlabel("Frequency [Hz]")

def plot_whole_stft(samples, sample_rate):
  """Plot STFT of an audio sample"""
  f, t, Zxx = signal.stft(samples, fs=sample_rate, nperseg=512*2, noverlap=0)
  plt.pcolormesh(t, f, np.abs(Zxx))
  plt.title('STFT Magnitude')
  plt.ylabel('Frequency [Hz]')
  plt.xlabel('Time [sec]')

def plot_an_window(analysis_window, f, include_centroid=False):
  """Plot an analysis window from STFT with optional spectral centroid"""
  plt.title('Window FFT')
  plt.ylabel("Amplitude")
  plt.xlabel("Frequency [Hz]")
  plt.plot(f, analysis_window)
  if include_centroid:
    plt.plot(f[spectral_centroid_idx(analysis_window)], 0, 'o', label='centroid')

def spectral_centroid_idx(analysis_window):
  """Return index of spectral centroid frequency of an analysis window 
  
  :param analysis_window: A matrix, frequency amplitudes for an analysis window
  computed from STFT
  """
  wnd = np.abs(analysis_window)
  bin_sum = np.sum(wnd*[i for i in range(1, len(wnd)+1)])
  mag_sum = np.sum(wnd)
  return int(round(bin_sum/mag_sum, 0)) - 1

def spectral_rolloff(analysis_window):
  """ Return the spectral rolloff in an analysis window

  :param analysis_window: A matrix, frequency amplitudes for an analysis window
  computed from STFT
  """
  return 0.85*np.sum(np.abs(analysis_window))

if __name__ == '__main__':
  samples, sample_rate = read_files()

  # Check if params are correct
  # Include overlap?
  # nperseg -> length of each segment (also number of frequencies per seg) should be *2
  # for some reason
  freqs, an_wndws_starts, an_wndws = signal.stft(samples, fs=sample_rate, nperseg=512*2, noverlap=0)
  # only interested in amplitudes
  an_wndws = np.abs(an_wndws)

  # plot FFT for whole audio file
  # plot_fft(samples, sample_rate)
  # plt.show()

  # plot STFT for while audio file
  # plot_whole_stft(samples, sample_rate)
  # plt.show()

  c_idx = spectral_centroid_idx(an_wndws[0])
  rolloff = spectral_rolloff(an_wndws[0])
  print(an_wndws[0, c_idx])
  print(rolloff)








