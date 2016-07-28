# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 13:44:00 2016

@author: timge
"""
import numpy as np
import copy
import matplotlib.pyplot as plt
from Miscellaneous import *

class Sandpile_2D(object):
    def __init__(self, N, Zc):
        self.N  = N
        self.Zc = Zc
        self.color = {'Non-Equilibrium': 'm', 'Flat': 'c'}
        
    def Initialize(self, method='Non-Equilibrium', ax=None):
        #Initialize a sandpile by two (extreme) methods
        #Non-Equilibrium: Create a random sandpile that exceeds Zcritical everywhere and let it relaxate to a meta-stable state
        #Flat: Start with an empty sandpile and start filling it by dropping sand grains at random positions and let the system relaxate to a met-stable state after each step
        
        self.InitMethod = method
        if method=='Non-Equilibrium':
            self.Z0 = np.random.randint(self.Zc+1, self.Zc*3, (self.N, self.N))
        
            #Boundary conditions
            self.Z0[[0,self.N-1],:] = 0
            self.Z0[:,[0,self.N-1]] = 0
            
            self.Z0, S = self.Stabilize(self.Z0)
            
        elif method=='Flat':
            t0 = self.Zc*self.N**2
            self.Z0 = np.zeros((self.N,self.N))   
            x = np.random.randint(1, self.N-1, (t0))
            y = np.random.randint(1, self.N-1, (t0))
        
            for i in range(t0):
                self.Z0[x[i], y[i]] +=1
                self.Z0, S = self.Stabilize(self.Z0)
                
        else:
            print('Method unknown')
            
        if ax!=None:
            c = mcolors.ColorConverter().to_rgb
            cm = make_colormap([c('red'), c('yellow'), 0.50, c('yellow'), c('green')])
            X, Y = np.meshgrid(np.arange(0, self.N, 1), np.arange(0, self.N, 1))
            ax.scatter(X, Y, c=self.Z0, cmap=cm, s=25, lw=0, marker='s')
            ax.set_xlim([0, self.N])
            ax.set_ylim([0, self.N])
            ax.set_title(self.InitMethod)
        
    def Stabilize(self, Z, bTrack=False):
        S = []
        
        while np.any(Z > np.ones(Z.shape)*self.Zc):
            #Determine which sand grain piles exceed the critical value
            n = np.argwhere(Z>self.Zc)
            
            #Drop sand grains to neighbouring positions
            Z[n[:,0],   n[:,1]  ] -= 4
            Z[n[:,0]-1, n[:,1]  ] += 1
            Z[n[:,0]+1, n[:,1]  ] += 1
            Z[n[:,0],   n[:,1]-1] += 1
            Z[n[:,0],   n[:,1]+1] += 1
            
            #Boundary conditions
            Z[[0,self.N-1],:] = 0
            Z[:,[0,self.N-1]] = 0
               
            #Keep track of the avalanche information if wanted
            if bTrack:
                S.append(n)
            
        return Z, S
        
    def AnalyzeFractals(self, ax, Nsamples=5):
        c = mcolors.ColorConverter().to_rgb
        cm = make_colormap([c('red'), c('yellow'), 0.50, c('yellow'), c('green')])
        
        Z = copy.copy(self.Z0)
        for i in range(Nsamples):
            #Increase one sandpile that is critical by one sand grain
            R = np.argwhere(Z==self.Zc)
            ind = np.random.randint(0, len(R), 1)
            Z[R[ind,0], R[ind,1]]  +=1
            
            #Stabalize and keep track of the avalanche
            Zi, S = self.Stabilize(Z, bTrack=True)
            
            t = []
            x = []
            y = []
            for ti, rii in enumerate(S):
                for ri in rii:
                    t.append(ti)
                    x.append(ri[0])
                    y.append(ri[1])
                    
            #Plot the avalanche with increasing time by color
            ax.scatter(x, y, c=np.array(t), cmap=cm, s=100, lw=0, alpha=0.15, marker='s')
            
        ax.set_xlim([0, self.N])
        ax.set_ylim([0, self.N])
     
             
        
    def AnalyzeNoise(self, ax, T=1000):
        #Measure the energy dissipation(==number of grains falling) by increasing a random sandpile one by one and keep track of the energy dissipation
        Z = copy.copy(self.Z0)
        x = np.random.randint(1, self.N-1, (T))
        y = np.random.randint(1, self.N-1, (T))
        Ft = []
        for i in range(T):
            Z[x[i], y[i]] +=1
            Z, s = self.Stabilize(Z, bTrack=True)
            
            for si in s:
                Ft.append(len(si))
                
        #Plot the energy dissipation in time
        ax[0].plot(Ft, c=self.color[self.InitMethod])
        ax[0].set_xlabel('Time [a.u.]')
        ax[0].set_ylabel('Energy dissipation [a.u.]')
        
        #Take the Fourier transform of the signal (on a logarthmic frequency axis) and fit it to a power law
        f, Ff = logFFT(Ft, dt=1, Nbins=40)
        ax[1].loglog(f, abs(Ff), c=self.color[self.InitMethod], marker='.', ls='none')
        ax[1].set_xlabel('Frequency [1/a.u.]')
        ax[1].set_ylabel('Amplitude [a.u.]')
        popt = FitToPowerlaw(abs(Ff), f)
        print('Noise parameters:', popt)
        ax[1].loglog(f, f**popt[0]*10**popt[1], c=self.color[self.InitMethod])  
        
        
        
    def AnalyzeLengthAndTime(self, ax, Nsamples=1):
        #Analyze the size and time scale at which avalanches occur by taking Nsamples, 
        #increasing the critical sandpiles one by one and keep track of the size and duration of an avalanche
        S = []
        T = []
        
        for j in range(Nsamples):
            if Nsamples!=0:
                self.Initialize(self.InitMethod)
                
            n = np.argwhere(self.Z0==self.Zc)
        
            for i, r in enumerate(n):
                Z = copy.copy(self.Z0)
                Z[r[0], r[1]] +=1
                Z, s = self.Stabilize(Z, bTrack=True)
                
                S.append(0)
                for si in s:
                    S[-1] += len(si)
                    
                T.append(len(s))
                
        nBins=20
        
        #Plot the probability of the size of an avalanche and fit it to a power law
        ax[0].set_xlabel('Size')
        ax[0].set_ylabel('Probability')
        D, bins = np.histogram(np.array(S), bins=np.logspace(0, np.log10(max(S)), nBins), density=True) 
        popt = FitToPowerlaw(D, bins)
        print('Fitting parameters size:', popt)
        ax[0].loglog(bins[0:-1], D, c=self.color[self.InitMethod], ls='None', marker='.')
        ax[0].loglog(bins[0:-1], bins[0:-1]**popt[0]*10**popt[1], c=self.color[self.InitMethod])
        
        #Plot the probability of the duration of an avalanche and fit it to a power law
        ax[1].set_xlabel('Time')
        ax[1].set_ylabel('Probability')
        D, bins = np.histogram(np.array(T), bins=np.logspace(0, np.log10(max(T)), nBins), density=True)
        popt = FitToPowerlaw(D, bins)
        print('Fitting parameters time', popt)
        ax[1].loglog(bins[0:-1], D, c=self.color[self.InitMethod], ls='None', marker='.')
        ax[1].loglog(bins[0:-1], bins[0:-1]**popt[0]*10**popt[1], c=self.color[self.InitMethod])

        return S, T
              
if __name__ == '__main__':
    figInit  = plt.figure()
    figFrac  = plt.figure()
    figNoise = plt.figure()
    axNoiseTime = figNoise.add_subplot(1,2,1)
    axNoiseFreq = figNoise.add_subplot(1,2,2)
    figScale = plt.figure() 
    axLength = figScale.add_subplot(1, 2, 1)
    axTime   = figScale.add_subplot(1, 2, 2)
    
    N = 250
    
    for i, method in enumerate(['Non-Equilibrium', 'Flat']):
        print(method)
        #Initialize sandpile
        Sandpile = Sandpile_2D(N=N, Zc=3)
        axInit = figInit.add_subplot(1,2,i+1)
        Sandpile.Initialize(method=method, ax=axInit)
        
        #Analyze fractal patterns
        axFrac = figFrac.add_subplot(1,2,i+1)
        axFrac.set_title(method)
        Sandpile.AnalyzeFractals(axFrac, Nsamples=4)
                
        #Analyze noise frequency behavior
        Sandpile.AnalyzeNoise([axNoiseTime, axNoiseFreq], T=1000)
        
        #Analyze length and time scale behavior
        Sandpile.AnalyzeLengthAndTime([axLength, axTime], Nsamples=1)
        

        
    plt.show()