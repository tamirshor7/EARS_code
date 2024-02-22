from scipy.signal import butter, lfilter, filtfilt, stft
import librosa
import librosa.display as display
import matplotlib.pyplot as plt

import numpy as onp

import jax.numpy as jnp
from jax_spectral import spectral # https://github.com/cifkao/jax-spectral

########### filters ########### 
# https://scipy-cookbook.readthedocs.io/items/ButterworthBandpass.html

def butter_lowpass_helper(cutoff, fs, by, order):
    # helper func
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype=by, analog=False)
    return b, a

def butter_bandpass_helper(lowcut, highcut, fs, order, btype='bandpass'):
    # helper func
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype=btype)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5, by='lowpass'):
    b, a = butter_lowpass_helper(cutoff, fs, by=by, order=order)
    #y = lfilter(b, a, data)
    y = filtfilt(b, a, data)
    return y

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass_helper(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y

def butter_bandstop_filter(data, lowcut, highcut, fs, order=2):
    b, a = butter_bandpass_helper(lowcut, highcut, fs, order=order, btype='bandstop')
    y = filtfilt(b, a, data)
    return y

########### spectrogram ###########
def convert_to_spectrogram(x, n_fft=1024):
    """
    convert_audio_to_spectogram -- using librosa to simply plot a spectogram
    
    Arguments:
    filename -- filepath to the file that you want to see the waveplot for
    
    Returns -- None
    """
    
    # stft is short time fourier transform
    X = librosa.stft(x, n_fft)
    
    # convert the slices to amplitude
    Xdb = librosa.amplitude_to_db(abs(X))
    
    return Xdb

def plot_spectrogram(Xdb, sr):
    # plot
    plt.figure(figsize=(14, 5))
    display.specshow(Xdb, sr = sr, x_axis = 'time', y_axis = 'hz')
    plt.colorbar()
    labels, locations = plt.xticks()
    plt.xticks(labels[:-1], (labels/2)[:-1])
    #plt.savefig('images/spectro.png')

def scipy_stft(sample, n_fft=1024, return_db_and_phase=False):
    # compute the STFT of a sample.
    # note that scipy's implementation scale the result 
    # https://dsp.stackexchange.com/questions/71410/difference-between-librosa-stft-and-scipy-signal-stft

    # the lines below implement the same transform (almost)
    # https://gist.github.com/bmcfee/746e572232be36f3bd462749fb1796da
    #D = librosa.stft(audio_sample, center=False, window='hamming', dtype=np.complex256, n_fft=n_fft)
    X = stft(sample, padded=False, window='hamming', nperseg=n_fft, noverlap=n_fft - n_fft//4, boundary=None)[-1]

    if return_db_and_phase:
        # https://dsp.stackexchange.com/questions/45843/how-to-convert-db-back-to-manitude-and-then-to-rectangular-format
        Xdb = 20 * onp.log(onp.abs(X)) # db
        Xphi = onp.angle(X) # phase

        return Xdb, Xphi
    else:
        return X

def jax_scipy_stft(samples, n_fft=1024, return_db_and_phase=False):
    # implementation of STFT with JAX: https://github.com/cifkao/jax-spectral
    # usage: https://github.com/khiner/jaxdsp/tree/c50967bed24a4bc072364b40c92215b629c9fd14/jaxdsp
    #TODO: make sure when there's an official JAX implementation of scipy.signal.stft
    X = spectral.stft(samples, padded=False, window='hamming', nperseg=n_fft, noverlap=n_fft - n_fft//4, boundary=None)[-1]

    if return_db_and_phase:
        # https://dsp.stackexchange.com/questions/45843/how-to-convert-db-back-to-manitude-and-then-to-rectangular-format
        Xdb = 20 * jnp.log(jnp.abs(X)) # db
        Xphi = jnp.angle(X) # phase

        return Xdb, Xphi
    else:
        return X