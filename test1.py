from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis import audioFeatureExtraction
import matplotlib.pyplot as plt
import numpy as np

[Fs_rock, x_rock] = audioBasicIO.readAudioFile("genres/rock/rock.00093.wav")
[Fs_pop, x_pop] = audioBasicIO.readAudioFile("genres/pop/pop.00000.wav")

# inputsignalen, samplignsfrequensen (Hz), short term window size (in samples), short term window step (in samples)
F, f_names = audioFeatureExtraction.stFeatureExtraction(x_rock, Fs_rock, 0.50*Fs_rock, 0.025*Fs_rock)

#[chromaGram, TimeAxis, FreqAxis] = audioFeatureExtraction.stChromagram(x_rock, Fs_rock, 0.50*Fs_rock, 0.025*Fs_rock, PLOT=True)
#nChroma, nFreqsPerChroma = audioFeatureExtraction.stChromaFeaturesInit(int(0.025*Fs_rock), Fs_rock)

#maxRows = np.max(chromaGram, 0)
#maxvalues = []

#for indx, maxR in enumerate(maxRows):
#    maxvalues.append(chromaGram[maxR, indx])

#print(maxRows)

# Plotting the signal
#plt.subplot(2, 1, 1); plt.plot(x_rock)
#plt.subplot(2, 1, 2); plt.plot(x_pop)
#plt.show()

#plt.subplot(2,1,1); plt.plot(F[0,:]); plt.xlabel('Frame no'); plt.ylabel(f_names[0]);
#plt.subplot(2,1,2); plt.plot(F[1,:]); plt.xlabel('Frame no'); plt.ylabel(f_names[1]); plt.show()

#plt.subplot(2, 2, 1); plt.plot(F[8, :])
#plt.subplot(2, 2, 2); plt.plot(F[9, :])
#plt.subplot(2, 2, 3); plt.plot(F[10, :])
#plt.subplot(2, 2, 4); plt.plot(F[11, :])
#plt.show()

#pitch_things = F[21:33, :]


#plt.subplot(2, 2, 1); plt.plot(F[21, :])
#plt.subplot(2, 2, 2); plt.plot(F[22, :])
#plt.subplot(2, 2, 3); plt.plot(F[23, :])
#plt.subplot(2, 2, 4); plt.plot(F[24, :])
#plt.show()

#plt.hist(F[32, :], "auto")
#plt.show()

# Also plotting if third parameter is True
#(bpm, ratio) = audioFeatureExtraction.beatExtraction(F, 0.5, True)
(bpm, ratio) = audioFeatureExtraction.beatExtraction(F, 0.05, True)

print(bpm)