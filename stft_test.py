from scipy.signal import stft, hilbert
from pyAudioAnalysis import audioBasicIO
import matplotlib.pyplot as plt
from numpy import array, sign, zeros
from scipy.interpolate import interp1d
from matplotlib.pyplot import plot,show,grid
import numpy as np

# Approximating envelope computation from stack overflow https://stackoverflow.com/questions/34235530/python-how-to-get-high-and-low-envelope-of-a-signal
def comp_envelope(s):
    # Prepend the first value of (s) to the interpolating values. This forces the model to use the same starting point for both the upper and lower envelope models.

    u_x = [0, ]
    u_y = [s[0], ]

    l_x = [0, ]
    l_y = [s[0], ]

    # Detect peaks and troughs and mark their location in u_x,u_y,l_x,l_y respectively.

    for k in range(1, len(s) - 1):
        if (sign(s[k] - s[k - 1]) == 1) and (sign(s[k] - s[k + 1]) == 1):
            u_x.append(k)
            u_y.append(s[k])

        if (sign(s[k] - s[k - 1]) == -1) and ((sign(s[k] - s[k + 1])) == -1):
            l_x.append(k)
            l_y.append(s[k])

    # Append the last value of (s) to the interpolating values. This forces the model to use the same ending point for both the upper and lower envelope models.

    u_x.append(len(s) - 1)
    u_y.append(s[-1])

    l_x.append(len(s) - 1)
    l_y.append(s[-1])

    # Fit suitable models to the data. Here I am using cubic splines, similarly to the MATLAB example given in the question.

    u_p = interp1d(u_x, u_y, kind='cubic', bounds_error=False, fill_value=0.0)
    l_p = interp1d(l_x, l_y, kind='cubic', bounds_error=False, fill_value=0.0)

    # Evaluate each model over the domain of (s)

    q_u = np.zeros(len(s))
    q_l = np.zeros(len(s))
    for k in range(0, len(s)):
        q_u[k] = u_p(k)
        q_l[k] = l_p(k)

    # Plot everything
    plot(s)
    plot(q_u, 'r')
    plot(q_l, 'g')
    grid(True)
    show()

[Fs_rock, x_rock] = audioBasicIO.readAudioFile("genres/rock/rock.00093.wav")
#
#print(Fs_rock)
#
#plt.subplot(2, 1, 1); plt.plot(x_rock)
#plt.show()
#
#amp = 2 * np.sqrt(2)

#x_rock = x_rock/200

f, t, short_time_fourier = stft(x_rock, Fs_rock, nperseg=512) # Not sure how to specify the window size correctly

#plt.pcolormesh(t, f, np.abs(short_time_fourier))
#plt.title('STFT Magnitude')
#plt.ylabel('Frequency [Hz]')
#plt.xlabel('Time [sec]')
#plt.show()

# Plots the envelopes
#comp_envelope(short_time_fourier[:, 110].real)

# Plots the fft for a window? Does it look reasonable?
plt.plot(f, short_time_fourier[:, 1000].real) # Using only the real part, what does the imaginary do?

# Magnitude only the absolute value?? https://www.mathworks.com/help/matlab/math/fourier-transforms.html
plt.plot(f, np.abs(short_time_fourier[:, 1000].real))
plt.show()


# Fourier transform

#fft = np.fft.fft(x_rock)

#real = fft.real
#freq = np.fft.fftfreq(x_rock.shape[0])

#plt.plot(freq, real)
#plt.show()
