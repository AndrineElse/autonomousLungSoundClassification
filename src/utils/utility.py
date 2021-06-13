#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 13:55:40 2020

@author: andrine
"""
import pandas as pd

import wave
import numpy as np
import scipy.io.wavfile as wf
import scipy.signal
import pywt
from scipy.signal import resample
import os
from sklearn import preprocessing
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.metrics import confusion_matrix

from sktime.utils.data_io import load_from_tsfile_to_dataframe,load_from_arff_to_dataframe


module_path = os.path.abspath(os.path.join('../..'))

from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
from matplotlib.colors import LinearSegmentedColormap
font = FontProperties(fname = module_path + '/src/visualization/CharterRegular.ttf', size = 10, weight = 1000)
font_small = FontProperties(fname = module_path + '/src/visualization/CharterRegular.ttf', size = 8, weight = 1000)


from sklearn.decomposition import PCA


def downsample(audio, sr, sr_new = 8000):
    secs = len(audio)/sr # Number of seconds in signal X
    new_sr = sr_new
    samps = round(secs*new_sr)     # Number of samples to downsample
    new_audio = resample(audio, samps)

    return new_audio


#Will resample all files to the target sample rate and produce a 32bit float array
def read_wav_file(str_filename, target_rate):
    wav = wave.open(str_filename, mode = 'r')
    (sample_rate, data) = extract2FloatArr(wav,str_filename)

    if (sample_rate != target_rate):
        ( _ , data) = resample_2(sample_rate, data, target_rate)

    wav.close()
    return (target_rate, data.astype(np.float32))

def resample_2(current_rate, data, target_rate):
    x_original = np.linspace(0,100,len(data))
    x_resampled = np.linspace(0,100, int(len(data) * (target_rate / current_rate)))
    resampled = np.interp(x_resampled, x_original, data)
    return (target_rate, resampled.astype(np.float32))

# -> (sample_rate, data)
def extract2FloatArr(lp_wave, str_filename):
    (bps, channels) = bitrate_channels(lp_wave)

    if bps in [1,2,4]:
        (rate, data) = wf.read(str_filename)
        divisor_dict = {1:255, 2:32768}
        if bps in [1,2]:
            divisor = divisor_dict[bps]
            data = np.divide(data, float(divisor)) #clamp to [0.0,1.0]
        return (rate, data)

    elif bps == 3:
        #24bpp wave
        return read24bitwave(lp_wave)

    else:
        raise Exception('Unrecognized wave format: {} bytes per sample'.format(bps))

#Note: This function truncates the 24 bit samples to 16 bits of precision
#Reads a wave object returned by the wave.read() method
#Returns the sample rate, as well as the audio in the form of a 32 bit float numpy array
#(sample_rate:float, audio_data: float[])
def read24bitwave(lp_wave):
    nFrames = lp_wave.getnframes()
    buf = lp_wave.readframes(nFrames)
    reshaped = np.frombuffer(buf, np.int8).reshape(nFrames,-1)
    short_output = np.empty((nFrames, 2), dtype = np.int8)
    short_output[:,:] = reshaped[:, -2:]
    short_output = short_output.view(np.int16)
    return (lp_wave.getframerate(), np.divide(short_output, 32768).reshape(-1))  #return numpy array to save memory via array slicing


def bitrate_channels(lp_wave):
    bps = (lp_wave.getsampwidth() / lp_wave.getnchannels()) #bytes per sample
    return (bps, lp_wave.getnchannels())


def denoise_audio(audio):
    coeff = pywt.wavedec(audio, 'db8')
    sigma = np.std(coeff[-1] )
    n= len( audio )
    uthresh = sigma * np.sqrt(2*np.log(n*np.log2(n)))
    coeff[1:] = ( pywt.threshold( i, value=uthresh, mode='soft' ) for i in coeff[1:] )
    denoised_audio =  pywt.waverec( coeff, 'db8' )
    return denoised_audio

def get_entropy(timeseries):
    timeseries_nz = timeseries[timeseries != 0]
    return - np.sum(((timeseries_nz**2)*np.log(timeseries_nz**2)))

def get_energy(timeseries):  
    N = len(timeseries)
    return np.sum(np.abs(timeseries) ** 2) / N


def get_t(y, sr):
    n = len(y)
    t = np.linspace(0, 1/ sr, n)
    return t

def convert_arff_to_ts(filepath, filename):
    X, y = load_from_arff_to_dataframe(filepath + '/' + filename)
    new_filename = filename[:-4] + 'ts'
    print(new_filename)
    dataset = filename.split('_')[0]
    print(dataset)
    
    labels = np.unique(y).astype(str)
    label_str = ''
    for label in labels:
        label_str = label_str + label + ' '
    print(label_str)
    w = open(filepath + '/' + new_filename, 'w+')
    
    w.write(f'@problemName {dataset} \n')
    w.write('@timeStamps false \n')
    w.write('@univariate true \n')
    w.write(f'@classLabel true {label_str} \n')
    w.write('@data \n')
    for (idx, row) in X.iterrows():
        new_row = (list(row)[0]).tolist()
        new_row = str(new_row)[1:-1].replace(' ', '') + ':' + y[idx] + '\n'
        w.write(new_row)
        
        


        
def plot_cm(y_true, y_pred, module_path = module_path, color_index = None, class_names = ['no-crackle', 'crackle' ], hex_color_str = None):
    cm = confusion_matrix(y_true, y_pred)
    colors = ['#F94144', '#F3722C', '#F8961E', '#F9844A', '#F9C74F', '#90BE6D', '#43AA8B', '#4D908E', '#577590', '#277DA1']
    #colors = ['#F9414466', '#90BE6D66', '#57759066','#F3722C66', '#F8961E66',
    #         '#F9844A66', '#F9C74F66', '#43AA8B66', '#4D908E66', '#277DA166']
    font = FontProperties(fname = module_path + '/src/visualization/CharterRegular.ttf', size = 10, weight = 1000)
    
    
    if hex_color_str:
        colors_2 = ['#FFFFFF', hex_color_str]
    elif color_index:
        colors_2 = ['#FFFFFF', colors[color_index]]
    else: 
        colors_2 = ['#FFFFFF', colors[0]]
    cmap_name = 'my colormap'
    font_small = FontProperties(fname =  module_path + '/src/visualization/CharterRegular.ttf', size = 6, weight = 1000)

    cm_map = LinearSegmentedColormap.from_list(cmap_name, colors_2)



    f, ax = plt.subplots(1,1) # 1 x 1 array , can also be any other size
    f.set_size_inches(3, 3)

    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    ax = sns.heatmap(cm, annot=True,
                fmt='.2%', cmap=cm_map, xticklabels=class_names,yticklabels=class_names )
    cbar = ax.collections[0].colorbar
    for label in ax.get_yticklabels() :
        label.set_fontproperties(font_small)
    for label in ax.get_xticklabels() :
        label.set_fontproperties(font_small)
    ax.set_ylabel('True Label', fontproperties = font)
    ax.set_xlabel('Predicted Label', fontproperties = font)
    ax.set_xticklabels(ax.get_xticklabels(), rotation = 0)

    for child in ax.get_children():
        if isinstance(child, matplotlib.text.Text):
            child.set_fontproperties(font)
    for l in cbar.ax.yaxis.get_ticklabels():
        l.set_fontproperties(font_small)
        
    return f,ax
