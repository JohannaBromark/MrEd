from timbral_texture import read_file
import pywt # Pywavelets library
import matplotlib.pyplot as plt
import numpy as np

# "signal is decomposed into a number of octave frequency bands using DWT" Does that mean that there is a filter
# applied so that only the correct frequencies of the signal is left??? Or what does it mean????
#

def full_wave_rectification(signal):
    "Returns the time domain amplitude envelope"
    return np.abs(signal)

def low_pass_filtering(signal):
    "Applies a one-pole filter to smooth the signal (envelope of the signal)"
    y = []
    alpha = 0.99
    y.append((1-alpha)*signal[0])

    for idx, xn in enumerate(signal[1:]):
        y.append((1-alpha)*xn + (alpha*y[idx-1]))

    return np.array(y)

def down_sampling(signal):
    k = 16
    y = []
    i = 0
    while k*i - 1 < len(signal):
        y.append(signal[k*i])
        i += 1

    return np.array(y)

def mean_removal(signal):
    mean = np.mean(signal)
    return signal - mean

def enhanced_autocorrelation(signal):
    y = []
    N = len(signal)

    for k in range(N):
        # Now assuming that if n-k < 0 => x[n-k] = 1
        xk = np.ones(N)
        xk[k:] = signal[:(N-k)]
        res = signal * xk
        y.append(np.sum(res)/N)
    y = np.array(y)

    return np.array(y)

if __name__ == '__main__':

    sample_rate, sample = read_file()

    wavelet = pywt.Wavelet('db4')

    # approx is from the low-pass filter and detail is from high-pass filter
    approx_coeffs, detail_coeffs = pywt.dwt(sample[0:65537], wavelet)

    num_levels = 10

    #coeffs_list = pywt.wavedec(sample, wavelet, level=num_levels, mode="zero")  # mode zero-padding
    coeffs_list = pywt.wavedec(sample, wavelet, level=num_levels)  # mode symmetric

    #for i in range(num_levels):
    #    plt.plot([j for j in range(len(coeffs_list[i]))], coeffs_list[i])
    #    plt.show()

    envelope = full_wave_rectification(coeffs_list[5])
    print("envelope done")
    low_pass = low_pass_filtering(envelope)
    print("low pass done")
    down_sample = down_sampling(low_pass)
    print("down sample done")
    mean = mean_removal(down_sample)
    print("mean removal done")

    #plt.plot([j for j in range(len(mean))], mean)
    #plt.show()

    enhanced_auto = enhanced_autocorrelation(mean)
    print("calculation done")

    plt.plot([j for j in range(len(enhanced_auto))], enhanced_auto)
    plt.show()

    print("hello world")
