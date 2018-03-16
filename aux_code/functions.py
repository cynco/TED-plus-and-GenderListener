# These functions were created by Cynthia E. Correa, Ph.D.

import pandas as pd
import numpy as np
from scipy import stats
from sphfile import SPHFile
import matplotlib.pyplot as plt
import seaborn as sns
from math import log
import timeit as ti
import os


def dropCols(df, dropColsL):
    for col in dropColsL:
        try:
            df = df.drop(col, axis = 1)
        except ValueError:
            pass
    return df;

# beep to alert when finished
# import timeit as ti
# import os
def beep(n=3):
    beeps = lambda x: os.system("echo -n '\a';sleep 0.2;" * x) # Alert when code finishes runni
    return beeps(n)

def remove_duplicates(A):
   [A.pop(count) for count,elem in enumerate(A) if A.count(elem)!=1]
   return A

def axhlines(ys, **plot_kwargs):
    """
    Draw horizontal lines across plot
    :param ys: A scalar, list, or 1D array of vertical offsets
    :param plot_kwargs: Keyword arguments to be passed to plot
    :return: The plot object corresponding to the lines.
    """
    ys = np.array((ys, ) if np.isscalar(ys) else ys, copy=False)
    lims = plt.gca().get_xlim()
    y_points = np.repeat(ys[:, None], repeats=3, axis=1).flatten()
    x_points = np.repeat(np.array(lims + (np.nan, ))[None, :], repeats=len(ys), axis=0).flatten()
    plot = plt.plot(x_points, y_points, scalex = False, **plot_kwargs)
    return plot


def axvlines(xs, **plot_kwargs):
    """
    Draw vertical lines on plot
    :param xs: A scalar, list, or 1D array of horizontal offsets
    :param plot_kwargs: Keyword arguments to be passed to plot
    :return: The plot object corresponding to the lines.
    """
    xs = np.array((xs, ) if np.isscalar(xs) else xs, copy=False)
    lims = plt.gca().get_ylim()
    x_points = np.repeat(xs[:, None], repeats=3, axis=1).flatten()
    y_points = np.repeat(np.array(lims + (np.nan, ))[None, :], repeats=len(xs), axis=0).flatten()
    plot = plt.plot(x_points, y_points, scaley = False, **plot_kwargs)
    return plot

def removeEdgeCases(df, column):
    # remove bottom and top percentiles for a given field
    # Input is the original, unmodified df and column to be filtered
    # Output is titles of talks to be removed
    fieldBottom = df[column].quantile(0.03) # 5th percentile
    fieldTop = df[column].quantile(0.97) # 95th percentile
    omit_list = [title for ix, title in enumerate(df['title']) if df.iloc[ix][column]<fieldBottom or df.iloc[ix][column]>fieldTop]
    return omit_list

def plotEdgeCases(df, column):
    fieldBottom = df[column].quantile(0.05) # 5th percentile
    fieldTop = df[column].quantile(0.95) # 95th percentile
    ax = sns.distplot(df[column])
    axvlines([fieldBottom, fieldTop])
    # Limit axes ranges
    if min(df[column])>0.5*fieldBottom:
        ax.set(xlim=(0.5*fieldBottom, max(df[column])))
    if max(df[column])>1.5*fieldTop:
        ax.set(xlim=(min(df[column]), 1.5*fieldTop))
    return ax;


# In[19]:


#define convenience function for trend lines
def plotTrendLine(x,y,data, color='red', logx=False,logy=False):
    oldx = np.reshape(data[x].values,(-1,))
    oldy = np.reshape(data[y].values,(-1,))
    tempx = oldx
    tempy = oldy
    if logx:
        tempx = np.log10(tempx)
    if logy:
        tempy = np.log10(tempy)
    idx = np.isfinite(tempx) & np.isfinite(tempy)
    z = np.polyfit(tempx[idx],tempy[idx],1)
    tempy = z[0]*tempx+z[1]
    plt.plot(oldx,tempy,color=color)
    return z


# In[20]:

def normalize_and_plot(output, normalizer):
    # ex. output is df['Beautiful']
    # ex. normalizer is df['views']
    # standardize
    binned = output / normalizer
    binned = binned / np.mean(binned)
    sns.distplot(binned)
    plt.show()
    return binned




def normalize_and_bin(output, normalizer, binQuantiles=[.10,.20,.30, .40, .5,.6,.7,.8,.9]):
    # ex. output is df['Beautiful']
    # ex. normalizer is df['views']
    # standardize
    binned = output / normalizer
    binned = binned / np.mean(binned)
    bins = [round(binned.quantile(quant),4) for quant in binQuantiles] # 25th percentile
    #print(bins)
    binned = np.digitize(binned, bins, right=True)
    #sns.distplot(binned)
    #plt.show()
    return binned


def just_bin(output, normalizer, binQuantiles):
    # ex. output is df['Beautiful']
    # ex. normalizer is df['views']
    # ex. binQuantiles = [0.33, 0.66]
    # standardize
    binned = output
    #binned = binned / np.mean(binned)
    bins = [round(binned.quantile(quant),4) for quant in binQuantiles] # 25th percentile
    binned = np.digitize(binned, bins, right=True)
    return binned

# These functions are modified versions of pyAudioAnalysis functions

from aux_code.pyAudioAnalysis3 import audioBasicIO as aIO
from aux_code.pyAudioAnalysis3 import audioFeatureExtraction as aF
from scipy import signal
import pandas as pd
import numpy as np

def load_waveform(yt_id, folder = './Audio/'):
    '''load the .wav file corresponding to a given Youtube ID'''
    file_loc = folder + yt_id + '.wav'
    [Fs, x] = aIO.readAudioFile(file_loc)
    # Fs is the frame rate
    # x is a numpy array of the audio samples
    return Fs, x

def get_features(Fs,x,start,stop, window = 1.0, step = 1.0/2.0):

    #start_time = timeit.default_timer()
    F = aF.stFeatureExtraction(x[start*Fs:stop*Fs], Fs, window*Fs, step*Fs);
    #elapsed_time = timeit.default_timer() - start_time
    #print('Basic feature extraction took %d seconds' % elapsed_time)

    #Create a time vector appropriate for plotting the features F
    time_F = np.linspace(start, stop, F.shape[0])
    #print('len(time_F)',len(time_F)) # gives 34
    return F, time_F

# calc_audio_features requires load_waveform functions, mono, get_vec, get_features
# I added to calc_audio_features a path argument

def calc_audio_features(yt_id , path, window_size, step_size):
#analyse the audio file
    #yt_id is just the file name without .wav
    Fs, x = load_waveform(yt_id, path)
    x = get_mono(x)
    timevec = get_time_vec(Fs,x)
    start = 1
    stop = int(timevec[-1])
    video_length = stop
    #print(stop)

    #Analyse the file
    #window_size = 1.0
    #step_size = window_size/10.0  #step_size/2 works, step_size/3 doesn't.
    Features, FeatureTime = get_features(Fs,x,start, stop, window = window_size, step = step_size)
    return Features, FeatureTime

def load_waveform(yt_id, path = './Audio/'):
    '''load the .wav file corresponding to a given Youtube ID'''
    file_loc = path + yt_id + '.wav'
    [Fs, x] = aIO.readAudioFile(file_loc)
    # Fs is the frame rate, usually 16000
    # x is a numpy array of the audio samples, it's length is duration in seconds
    return Fs, x

def get_mono(x):
    '''average over the two audio channels to produce a mono signal'''
    # much taken from pyAudioAnalysis3 stereo2mono
    if isinstance(x, int):
        return -1
    if x.ndim==1:
        return x
    elif x.ndim==2:
        if x.shape[1]==1:
#             return x.flatten()
            x_mono = x.mean(axis=1)
            return x_mono
        else:
            if x.shape[1]==2:
                return ( (x[:,1] / 2) + (x[:,0] / 2) )
            else:
                return -1
    return x_mono

def get_time_vec(Fs, x):
    '''with a given sampling rate, produce a vector of times to use for plotting'''
    T = len(x) / Fs
    timevec = np.linspace(0, T, len(x))
    return timevec
