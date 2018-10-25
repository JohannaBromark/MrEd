from timbral_texture import read_file
import pywt # Pywavelets library
import matplotlib.pyplot as plt

# "signal is decomposed into a number of octave frequency bands using DWT" Does that mean that there is a filter
# applied so that only the correct frequencies of the signal is left??? Or what does it mean????
#

def full_wave_rectification(signal):
    pass


sample_rate, sample = read_file()

wavelet = pywt.Wavelet('db4')

# approx is from the low-pass filter and detail is from high-pass filter
approx_coeffs, detail_coeffs = pywt.dwt(sample, wavelet)

coeffs_list = pywt.wavedec(sample, wavelet)

plt.plot([i for i in range(len(approx_coeffs))], approx_coeffs)
plt.show()
plt.plot([i for i in range(len(detail_coeffs))], detail_coeffs)
plt.show()

print("hello world")
