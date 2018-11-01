from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis import audioFeatureExtraction as audioFE
import matplotlib.pyplot as plt
import numpy as np
import numpy.fft as fft
import scipy.signal as signal
from sklearn.preprocessing import normalize
from sklearn.mixture import GaussianMixture
import os
from utils import *

def spectral_centroid_idx(an_wndw):
  """Return index of spectral centroid frequency of an analysis window"""
  # DEPRECATED
  bin_sum = np.sum(an_wndw*[i for i in range(1, len(an_wndw)+1)])
  mag_sum = np.sum(an_wndw)
  return int(round(bin_sum/mag_sum, 0)) - 1

def spectral_centroid(an_wndw, freqs):
  """Return spectral centroid of an analysis window"""
  bin_sum = np.sum(an_wndw*[i for i in range(1, len(an_wndw)+1)])
  mag_sum = np.sum(an_wndw)
  return bin_sum/mag_sum

def spectral_rolloff(an_wndw, freqs):
  """Return the spectral rolloff freq of an analysis window"""
  limit = 0.85*np.sum(np.abs(an_wndw))
  total = 0
  for i in range(len(an_wndw)):
    total += an_wndw[i]
    if total > limit:
      return freqs[i-1]

def spectral_flux(an_wndw, prev_wndw):
  """Return the spectral flux of an analysis window
  
  :param prev_wndw: The analysis window one time step prior to an_wndw
  """
  # wrong previous normalisation
  # an_wndw *= 1./np.max(an_wndw, axis=0)
  # prev_wndw *= 1./np.max(prev_wndw, axis=0)
  an_wndw = an_wndw / np.square(np.var(an_wndw))
  prev_wndw = prev_wndw / np.square(np.var(prev_wndw))
  return np.sum(np.power(an_wndw-prev_wndw, 2))

def time_zero_crossings(wndw_no, samples, seg_size):
  """Return time domain zero crossings for an analysis window"""
  signed = np.where(samples[wndw_no*seg_size:(wndw_no+1)*seg_size] > 0, 1, 0)
  return (1/2) * np.sum([np.abs(signed[i]-signed[i-1]) for i in range(1, len(signed))])

def mfcc_coeffs(an_wndw, sample_rate):
  """Return the five first mfcc coefficients"""
  an_wndw_size = an_wndw.shape[0]
  [filter_bank, _] = audioFE.mfccInitFilterBanks(sample_rate, an_wndw_size)
  return audioFE.stMFCC(an_wndw, filter_bank, 5)

def rms_energy(wndw_no, samples, seg_size):
  """Return the RMS energy of an analysis window"""
  energy = [np.power(i, 2) for i in samples[wndw_no*seg_size:(wndw_no+1)*seg_size]]
  return np.square(np.sum(energy) * 1./seg_size)

#Mean and variance of the centroids
def MVcentroid(an_wndws,freqs,t_wndw_size):
  centroids = []
  for i in range(t_wndw_size):
    centroids = np.append(centroids, spectral_centroid(an_wndws[i],freqs))

  mean = np.sum(centroids)/t_wndw_size
  var = 0
  for k in range(t_wndw_size):
    var = var + ((centroids[k]-mean)**2)
  var/(t_wndw_size-1)
  return mean, var

#Mean and variance of the rolloffs
def MVrolloffs(an_wndws,t_wndw_size,freqs):
  rolloffs = []
  for i in range(t_wndw_size):
    rolloffs = np.append(rolloffs, spectral_rolloff(an_wndws[i],freqs))

  mean = np.sum(rolloffs)/t_wndw_size
  var = 0
  for k in range(t_wndw_size):
    var = var + ((rolloffs[k]-mean)**2)
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
    var = var + ((flux[k]-mean)**2)
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
    var = var + ((crossing[k]-mean)**2)
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

def MeVaRolloffs(an_wndws,t_wndw_size,nr_wndws,freqs):
  mean_rolloffs = []
  var_rolloffs = []
  for i in range(0, nr_wndws, t_wndw_size):
    mean_rolloff, var_rolloff = MVrolloffs(an_wndws[:,i:i+t_wndw_size],t_wndw_size,freqs)
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
      variance = variance + ((mfccs[a,k]-mean[k])**2)
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

def CreateFeatureVectors(song_nr,seg_size,samples,sample_rate,an_wndws,freqs,t_wndw_size,nr_wndws):
  mean_centroids, var_centroids = MeVaCentroid(an_wndws, freqs, t_wndw_size,nr_wndws)
  mean_rolloffs, var_rolloffs = MeVaRolloffs(an_wndws,t_wndw_size,nr_wndws,freqs)
  mean_fluxs, var_fluxs = MeVaFlux(an_wndws, t_wndw_size,nr_wndws)
  mean_crossings, var_crossings = MeVaZero_Crossings(samples,seg_size, t_wndw_size,nr_wndws)
  mean_mfccs, var_mfccs = MeVaMfcc(an_wndws,sample_rate,t_wndw_size,nr_wndws) #31 texture windows, 5 olika MFCSS i varje rad.
  mean_rms_energy = MeVaEnergy(samples,seg_size,t_wndw_size,nr_wndws)

  featureVector = np.zeros(20)
  featureMatrix = []
  for i in range(mean_centroids.size):
    featureVector[0] = song_nr
    featureVector[1] = mean_centroids[i]
    featureVector[2] = var_centroids[i]
    featureVector[3] = mean_rolloffs[i]
    featureVector[4] = var_rolloffs[i]
    featureVector[5] = mean_fluxs[i]
    featureVector[6] = var_fluxs[i]
    featureVector[7] = mean_crossings[i]
    featureVector[8] = var_crossings[i]
    featureVector[9] = mean_mfccs[i,0]
    featureVector[10] = mean_mfccs[i,1]
    featureVector[11] = mean_mfccs[i,2]
    featureVector[12] = mean_mfccs[i,3]
    featureVector[13] = mean_mfccs[i,4]
    featureVector[14] = var_mfccs[i,0]
    featureVector[15] = var_mfccs[i,1]
    featureVector[16] = var_mfccs[i,2]
    featureVector[17] = var_mfccs[i,3]
    featureVector[18] = var_mfccs[i,4]
    featureVector[19] = mean_rms_energy[i]
    featureMatrix = np.append(featureMatrix,featureVector)
  featureMatrix = featureMatrix.reshape(int(nr_wndws/43),20)
  # print(featureMatrix.shape)
  # print(featureMatrix)

  return featureMatrix
  
def createAll(all_samples,labels,flag = True):
  seg_size = 512
  t_wndw_size = 43
  
  featureMatrix = np.zeros((2,20))
  labelsMatrix = []
  for i in range(len(all_samples)):
    try:
      print(i)
      freqs, time_inits, stft_wndws = signal.stft(all_samples[i], fs=sample_rate, nperseg=seg_size, noverlap=0)
      an_wndws = np.abs(stft_wndws)
      nr_wndws = int(((all_samples[i].size/512)//43)*43)
      nr_t_wndws = int(nr_wndws/43)

      featureMatrix = np.concatenate((featureMatrix ,CreateFeatureVectors(i,seg_size,all_samples[i],sample_rate,an_wndws,freqs,t_wndw_size,nr_wndws)),axis =0)
      targets = np.zeros(nr_t_wndws)
      if flag:
        targets[0:nr_t_wndws] = labels[i][0]
      else:
        targets[0:nr_t_wndws] = labels[i]
      labelsMatrix = np.append(labelsMatrix,targets)
    except Exception as err:
      print('Någt gick snett till')
      print(err)
      print(i)
      print('******************')
      

    
  if flag:
    labelsMatrix = labelsMatrix.reshape(labelsMatrix.size,1)
  featureMatrix = featureMatrix[2:]
  return featureMatrix, labelsMatrix
def writetofile(all_samples,labels,filename1,filename2,flag=True):
  features, targets = createAll(all_samples,labels,flag)


  with open(filename1,'w') as file:
    for item in features:
      for element in item:
        file.write(str(element))
        file.write(' ')
      file.write('\n')

  with open(filename2, 'w') as file:
    for item in targets:
      if flag:
        file.write(str(int(item[0])))
      else:
        file.write(str(int(item)))
      file.write('\n')

if __name__ == '__main__':
  sample_rate, samples = read_file()
  all_samples, labels = read_directories()
  # print(all_samples)

  # sample_rateF,samplesF,targetsF = read_partition('features_targets/train_fault.txt')
  # sample_rateFT,samplesFT,targetsFT = read_partition('features_targets/test_fault.txt')
  # sample_rateFV,samplesFV,targetsFV = read_partition('features_targets/valid_fault.txt')


  # Check if params are correct
  # Include overlap? Praxis is to use default overlap setting
  # nperseg -> length of each segment (also number of frequencies per seg) should be *2 for some reason?
  # freqs, time_inits, stft_wndws = signal.stft(samples, fs=sample_rate, nperseg=seg_size, noverlap=0)
  # an_wndws = np.abs(stft_wndws) # abs -> we only want freq amplitudes
  # an_wndw = an_wndws[:,wndw_no] # col -> analysis window
 
  writetofile(all_samples,labels,'features_targets/ta_bort_mig.txt','features_targets/ta_bort_mig.txt') #all 1000 songs w/o partion
  # writetofile(samplesF,targetsF,'features_targets/featuresF.txt','features_targets/targetsF.txt',False) #Fault filtered partion train
  # writetofile(samplesFT,targetsFT,'features_targets/featuresFT.txt','features_targets/targetsFT.txt',False) #Fault filtered partion test
  # writetofile(samplesFV,targetsFV,'features_targets/featuresFV.txt','features_targets/targetsFV.txt',False) #Fault filtered partion vali



  

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

  # seg_size = 512
  # wndw_no = 1
  # freqs, time_inits, stft_wndws = signal.stft(samples, fs=sample_rate, nperseg=seg_size, noverlap=0)
  # an_wndws = np.abs(stft_wndws) # abs -> we only want freq amplitudes
  # an_wndw = an_wndws[:,wndw_no] # col -> analysis window

 
 

