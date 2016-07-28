# -*- coding: utf-8 -*-
"""
Created on Wed Jul 20 15:22:19 2016

@author: timge
"""

import numpy as np
import copy
import matplotlib.pyplot as plt
from Miscellaneous import *


class TrafficJams(object):
    def __init__(self,N, vmax=5):
        self.N = N
        self.vmax = vmax
        self.color = {'Jam': 'r', 'Flow': 'g', 'Random': 'b'}
                
    def Initialize(self, rho=0.1, method='Flow'):        
        self.Initialization = method
        
        #Create 3 different initializations by setting initial position and velocity:
            #Jam:    All cars are behind each other with velocity 0 (they stand for example before a traffic light)
            #Flow:   The cars are equally spaced on the available road with the maximum velocity
            #Random: The cars are randomly positioned and have a random velocity between 0-vmax
        
        #Initial condition positions:
        self.x0 = np.zeros((self.N), dtype=int)
        Xbase = np.zeros(int(1/rho))
        Xbase[0] = 1
        self.x0 = np.tile(Xbase, self.N/len(Xbase))   
        if method=='Random':
            np.random.shuffle(self.x0)   
        elif method=='Jam':
            self.x0 = np.sort(self.x0)
            self.x0 = np.roll(self.x0, int(self.N/2))
        
        #Initial condition velocities
        self.v0 = np.zeros((self.N), dtype=int)
        if method=='Flow':
            self.v0[np.argwhere(self.x0)] = np.ones(np.sum(self.x0))*self.vmax
        elif method=='Random':
            self.v0[np.argwhere(self.x0)] = np.random.randint(0, self.vmax, (np.sum(self.x0), 1))
        
    def Simulate(self, T):
        self.T = T
        x = np.zeros((self.N, self.T), dtype=int)
        v = np.zeros((self.N, self.T), dtype=int)
        Q = np.zeros((self.N, self.T), dtype=int)
        
        x[:,0] = self.x0
        v[:,0] = self.v0
        
        for t in range(1,self.T):
            X  = np.argwhere(x[:,t-1]==1)[:,0]
            dX = (np.roll(X, -1)-X)%self.N
            V  = v[X, t-1] 
            
            #Rule 1: Accelerate if possible
            V += np.logical_and(V<np.ones(len(V))*self.vmax,V<(dX+1))
            #Rule 2: Slow down if needed
            ind = np.argwhere(dX<=V)[:,0]
            V[ind] = dX[ind]-1
            #Rule 3: Slow down with chance p
            V -= np.logical_and(np.random.rand(len(V))<0.5, 0<V)
            #Rule 4: Take next time step
            x[(X+V)%self.N, t] = 1
            v[(X+V)%self.N, t] = V
            
            #Obtain the flow at each position
            for i, xi in enumerate(X):
                Q[xi:(xi+V[i])%self.N,t] +=1

        self.X = x
        self.V = v
        self.Q = Q
        
    def PlotTraffic(self, ax):
        X = np.argwhere(self.X==1)
        c = mcolors.ColorConverter().to_rgb
        cm = make_colormap([c('red'), c('yellow'), 0.50, c('yellow'), c('green')])

        ax.scatter(X[:,0],X[:,1], c=self.V[X[:,0], X[:,1]], cmap=cm, s=3, lw=0)
        ax.set_xlim([0, self.N])
        ax.set_ylim([0, self.T])
        ax.invert_yaxis()
        
    def DensityVsFlow(self, ax, T0, dT=50):
        #Obtain the correlation between the density and flow of cars by taking samples of dT long along the entire road
        N = int((self.T-T0)/dT)
        rho = []
        q   = []
        for i in range(N):
            rho =  np.concatenate((rho, np.mean(self.X[self.vmax+1:self.N-self.vmax, T0+i*dT:T0+(i+1)*dT], axis=1)))
            q    = np.concatenate((q  , np.mean(self.Q[self.vmax+1:self.N-self.vmax, T0+i*dT:T0+(i+1)*dT], axis=1)))

        ax.scatter(rho,q, lw=0, c=self.color[self.Initialization], alpha=0.5)
        return 
        
    def AnalyzeNoise(self, ax, T0=1000, dT=25):  
        #Anlyse the noise behavior of the system by looking at the flow of cars at a certain point (middle of the road)
        #as a function of time (averaged of a time sample of dT). Take the Fourier transform to see the noise in frequency space
    
        S = np.sum(self.Q[int(self.N/2), T0:].reshape((dT, (self.T-T0)/dT)), axis=0)
        ax[0].plot(S)
        
        f, Sf = logFFT(S, 1, 10)
        ax[1].loglog(f, abs(Sf), self.color[self.Initialization], marker='.', ls='none')
        popt = FitToPowerlaw(abs(Sf), f)
        ax[1].loglog(f, f**popt[0]*10**popt[1])  
        print('Noise power', popt[0])
        
    def GetJams(self, T0):
        #Determine where traffic jams start and stop. A car in a jam is defined by a velocity of 0 or 1. 
        #A jam is assumed to start if there was no car in a jam in a block of 2 timesteps before and 4 positions ahead of the car
        #A jam is assumed to stop if there is no car in a jam in a block of dt after and dx behind of the car
        #The values for dx and dt are arbitrary and obtained by trial and error, where dt is made dependent on the number of cars ahead in time of the starting point of the jam
    
        #Determin at which position and time cars are in a jam
        X = np.argwhere(self.X==1)
        J = np.zeros(self.X.shape)
        V = copy.copy(self.V)
        V -= 1
        V[X[:,0],X[:,1]] +=1
        J[X[:,0],X[:,1]] += np.logical_and(np.zeros(len(X))<=V[X[:,0], X[:,1]], V[X[:,0], X[:,1]]<np.ones(len(X)))
        Jams = []  
        
        dx = 2     
        
        #Determine the jams including the starting and stopping points
        i = 0
        for x, t in np.argwhere(J):
            i +=1
            
            #Avoid the boundary in time
            if t<=T0 or t>=self.T-15:
                continue
            
            #Determine if a jam starts
            if np.sum(J[x:x+4, t-2:t+1])==1:
                for dt in range(2,15):                
                    if np.sum(J[x:x+4, t-dt:t+1]) >1:
                        break
                
                x0, t0 = x, t 
                tNone = 0                
                xmin, xmax = [x0, x0]                
                while tNone<dt and t+1<self.T:
                    t +=1
                    tNone +=1
                    X = np.argwhere(J[:,t].take(range(xmin-dx, xmax+int(dx/2)), mode='wrap')) + (xmin-dx)
                    if len(X)!=0:
                        tNone=0
                        xmin, xmax = [min(X), max(X)]
                Jams.append([[x0, xmin],[t0, t-tNone]])
        
        self.J = J
        self.Jams = np.array(Jams)
        
    def PlotJams(self, ax):
        J = np.argwhere(self.J)
        #Plot cars in a jam
        ax.scatter(J[:,0], J[:,1], c='k', s=3, lw=0)
        
        #Plot jams themselves
        for jam in self.Jams:
            ax.scatter(jam[0,0], jam[1,0], c='r', s=10, alpha=0.7)
            ax.plot(jam[0], jam[1])   
            
        ax.set_xlim([0, self.N])
        ax.set_ylim([0, self.T])
        ax.invert_yaxis()
        
    def AnalyzeLengthAndTime(self, ax):
        #Obtain the lengths and duration of traffic jams
        dx = abs(self.Jams[:,0,0]-self.Jams[:,0,1])
        dt = abs(self.Jams[:,1,1]-self.Jams[:,1,0])
        
        nBins = 20
                    
        #Plot and fit the lengths and duration of traffic jams
        D, bins = np.histogram(np.array(dx), bins=np.logspace(0, np.log10(max(dx)), nBins), density=True)
        popt = FitToPowerlaw(D[:-4], bins)
        print('Length power', popt[0])
        ax[0].loglog(bins[0:-1], D, c=self.color[self.Initialization], ls='None', marker='.')
        ax[0].loglog(bins[0:-1], bins[0:-1]**popt[0]*10**popt[1], c=self.color[self.Initialization])

        
        D, bins = np.histogram(np.array(dt), bins=np.logspace(0, np.log10(max(dt)), nBins), density=True)
        popt = FitToPowerlaw(D[:-4], bins)
        print('Time power', popt[0])
        ax[1].loglog(bins[0:-1], D, c=self.color[self.Initialization], ls='None', marker='.')
        ax[1].loglog(bins[0:-1], bins[0:-1]**popt[0]*10**popt[1], c=self.color[self.Initialization])
        
        
        
if __name__ == '__main__':
    figTraf  = plt.figure()
    figDens  = plt.figure()
    figNoise = plt.figure()
    axNoiseTime = figNoise.add_subplot(1,2,1)
    axNoiseTime.set_xlabel('Time')
    axNoiseTime.set_ylabel('Flow')
    axNoiseFreq = figNoise.add_subplot(1,2,2)
    axNoiseFreq.set_xlabel('Frequency')
    axNoiseFreq.set_ylabel('???')
    figJams  = plt.figure()
    figScale = plt.figure()
    axLength = figScale.add_subplot(1,2,1)
    axLength.set_xlabel('Length')
    axLength.set_ylabel('Probability')
    axTime   = figScale.add_subplot(1,2,2)
    axTime.set_xlabel('Time')
    axTime.set_ylabel('Probability')
    
    N  = 1000
    T0 = 2*N
    for i, method in enumerate(['Jam', 'Flow', 'Random']):
        print(method)
        #Start and simulate traffic
        TrafficJam = TrafficJams(N=N)
        TrafficJam.Initialize(rho=0.1, method=method)
        TrafficJam.Simulate(T=200000)
        
        #Plot the traffic
        axTraf = figTraf.add_subplot(1,3,i+1)
        TrafficJam.PlotTraffic(axTraf)
        axTraf.set_title(method)
        axTraf.set_xlabel('Position (x)')
        axTraf.set_ylabel('Time (t)')      
        
        #Analyze the Density vs Flow
        axDens = figDens.add_subplot(1,3,i+1)
        TrafficJam.DensityVsFlow(ax=axDens, T0=T0, dT=100)
        axDens.set_title(method)
        axDens.set_xlabel('Occupancy [cars/site]')
        axDens.set_ylabel('Flow [cars/timestep]')        
        
        #Analyze the frequency behavior
        TrafficJam.AnalyzeNoise([axNoiseTime, axNoiseFreq], T0=T0)
        
        #Obtain and plot the traffic jams
        axJams = figJams.add_subplot(1,3,i+1)
        TrafficJam.GetJams(T0=T0)
        TrafficJam.PlotJams(axJams)
        axJams.set_title(method)
        axJams.set_xlabel('Position (x)')
        axJams.set_ylabel('Time (t)')  
        
        #Analyze the Time and Length scale
        TrafficJam.AnalyzeLengthAndTime([axLength, axTime])
        
    plt.show()
        
        
        