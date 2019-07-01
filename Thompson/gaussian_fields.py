# -*- coding: utf-8 -*-
"""
Guassian Random fields

@author: Marek
"""


# Imports
#%matplotlib inline

import numpy as np
import scipy
import seaborn as sns
import scipy.fftpack
from IPython import display
from cmath import *


sns.set_style('darkgrid')
np.random.seed(0)
#

def fft(h):
    """Computes the discrete Fourier transform of the sequence h.
    Assumes that len(h) is a power of two.
    """
    N = len(h)
 
    # the Fourier transform of a single value is itself
    if N == 1: return h
 
    # recursively compute the FFT of the even and odd values
    He = fft(h[0:N:2])
    Ho = fft(h[1:N:2])
 
    # merge the half-FFTs
    i = complex(0,1)
    W = exp(2*pi*i/N)
    ws = [pow(W,k) for k in range(N)]
    H = [e + w*o for w, e, o in zip(ws, He+He, Ho+Ho)]
    return H

    
    
def psd(H, N):
    p = [Hn * Hn.conjugate() for Hn in H]
    freqs = range(N/2 + 1)
    p = [p[f].real for f in freqs]
    return freqs, p
    

    
def fftind(size):
    """ Returns a numpy array of shifted Fourier coordinates k_x k_y.
        
        Input:
        size (integer): The size of the coordinate array to create
        
        Output:
            k_ind, numpy array of shape (2, size, size) with:
                k_ind[0,:,:]:  k_x components
                k_ind[1,:,:]:  k_y components
    """
    k_ind = np.mgrid[:size, :size] - int( (size + 1)/2 )
    k_ind = scipy.fftpack.fftshift(k_ind)
    return( k_ind )
    
    
    
def gaussian_random_field(alpha = 3.0, size = 128, flag_normalize = True):
    """ 
    Returns a numpy array of sizexsize discrete Gaussian random field
        
    Input:
    alpha (double, default = 3.0): 
        The power of the power-law momentum distribution
    size (integer, default = 128):
        The size of the square output Gaussian Random Fields
    flag_normalize (boolean, default = True):
        Normalizes the Gaussian Field:
            - to have an average of 0.0
            - to have a standard deviation of 1.0
    Output:
    gfield (numpy array of shape (size, size)):
        The random gaussian random field

    Example:
    import matplotlib
    import matplotlib.pyplot as plt
    example = gaussian_random_field()
    plt.imshow(example)
    """
        
        # Defines momentum indices
    k_idx = fftind(size)

        # Defines the amplitude as a power law 1/|k|^(alpha/2)
    amplitude = np.power( k_idx[0]**2 + k_idx[1]**2 + 1e-10, -alpha/4.0 )
    amplitude[0,0] = 0
    
        # Draws a complex gaussian random noise with normal
        # (circular) distribution
    noise = np.random.normal(size = (size, size)) \
        + 1j * np.random.normal(size = (size, size))
    
        # To real space
    gfield = np.fft.ifft2(noise * amplitude).real
    
        # Sets the standard deviation to one
    if flag_normalize:
        gfield = gfield - np.mean(gfield)
        gfield = gfield/np.std(gfield)
        
    return gfield

    
