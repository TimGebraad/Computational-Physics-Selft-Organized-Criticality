# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 10:38:23 2016

@author: timge
"""
import numpy as np
import matplotlib.colors as mcolors
from scipy.optimize import curve_fit
from scipy.fftpack import fft

def FitToPowerlaw(D, bins):
    def Linear(x, a, b):
        return a*x+b
    
    #Determine points that are non-zero to avoid NaN and fit to linear relation
    ind = np.where(D!=0)
    popt, pcov = curve_fit(Linear, np.log10(bins[ind]), np.log10(D[ind]))
    return popt
    
def logFFT(Ft, dt, Nbins):
    #Obtain Fourier Transform and the corresponding frequency axis
    Ff = fft(Ft)
    f = np.linspace(0,1/(2*dt),len(Ft)/2)
    
    #Obtain a logarithmic scale for the frequency
    flog =np.logspace(np.log10(np.mean(f[0:2])), np.log10(np.mean(f[-2:])), Nbins)

    #Map the Fourier transform on the logarithmic scale by taking the average of corresponding intervals
    #This is an approximation!
    Fflog = np.zeros(Nbins)        
    for i in range(Nbins-1):
        indlow = np.argwhere((f<flog[i]))
        indhig = np.argwhere((f<flog[i+1]))
        Fflog[i] = np.mean(Ff[max(indlow):max(indhig)+1])
    Fflog[-1] = np.mean(Ff[max(indhig):])
    
    return flog, Fflog

def make_colormap(seq):
        seq = [(None,) * 3, 0.0] + list(seq) + [1.0, (None,) * 3]
        cdict = {'red': [], 'green': [], 'blue': []}
        for i, item in enumerate(seq):
            if isinstance(item, float):
                r1, g1, b1 = seq[i - 1]
                r2, g2, b2 = seq[i + 1]
                cdict['red'].append([item, r1, r2])
                cdict['green'].append([item, g1, g2])
                cdict['blue'].append([item, b1, b2])
        return mcolors.LinearSegmentedColormap('CustomMap', cdict)
