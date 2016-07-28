# -*- coding: utf-8 -*-
"""
Created on Fri Jul 22 15:50:03 2016

@author: timge
"""


import numpy as np
import copy
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from Miscellaneous import *

class BakSteppenModel(object):
    def __init__(self,N):
        self.N = N
        self.color = {'AllZero': 'r', 'AllOne': 'g', 'Random': 'b'}
        
    def Initialize(self, method='Random'):
        #Initialize the species at t=0 accoring to three possibilities
            #AllZero: Every species has fitness zero
            #AllOne:  Every species has fitness one
            #Random:  Every species is assigned a random fitness
    
        self.method = method
        if method=='AllZero':
            self.X0 = np.zeros(self.N)
        elif method=='AllOne':
            self.X0 = np.ones(self.N)
        elif method=='Random':
            self.X0 = np.random.rand(self.N)
        else:
            print('Method is not known')
        
    def Simulate(self, T):  
        #Setup matrices to store the data
        self.T = T
        X = np.zeros((self.T,self.N))
        Age = np.zeros((self.T, self.N))
        Lifetime = []
        Agefrac = np.zeros((self.T,self.N))
        X[0] = self.X0
        
        #Run the model for an amount of time and keep track of the fitness, age and lifetimes
        for t in range(T-1):
            X[t+1] = X[t]
            Xmin = np.argmin(X[t,:])

            X[t+1,[Xmin, (Xmin+1)%self.N, (Xmin-1)%self.N]] = np.random.rand(3)
            Age[t+1,:] = Age[t,:] + 1
            Age[t+1,[Xmin, (Xmin+1)%self.N, (Xmin-1)%self.N]] = 0
            
            Agefrac[t+1] = Age[t+1]
            if Age[t, Xmin]>0:
                Agefrac[t-Age[t,Xmin]            :t+1  , Xmin]            /= Age[t,Xmin]           
            if Age[t, (Xmin+1)%self.N]>0:
                Agefrac[t-Age[t,(Xmin+1)%self.N] :t+1  , (Xmin+1)%self.N] /= Age[t,(Xmin+1)%self.N]   
            if Age[t, (Xmin-1)%self.N]>0:
                Agefrac[t-Age[t,(Xmin-1)%self.N] :t+1  , (Xmin-1)%self.N] /= Age[t,(Xmin-1)%self.N]
            Lifetime = np.concatenate((Lifetime, Age[t,[Xmin, (Xmin+1)%self.N, (Xmin-1)%self.N]]))
                
        #Calculate the age in a fraction of the total age
        for i in range(self.N):
            if Age[self.T-1,i] != 0:
                Agefrac[self.T-Age[self.T-1,i]:,i] /= Age[self.T-1,i]
                        
        #Store the relevant data in the class        
        self.X = X
        self.Age = Age
        self.Agefrac = Agefrac
        self.Lifetime = Lifetime
        
    def PlotAges(self, ax, AgeMethod='Relative Age'):
        #Plots the evolution of the species relative or absolute age as a funtion of time
        ax.set_xlabel('Species')
        ax.set_ylabel(AgeMethod)    
        
        xx, tt  =np.meshgrid(np.arange(0, self.N, 1), np.arange(0, self.T, 1))
        
        c = mcolors.ColorConverter().to_rgb
        cm = make_colormap([c('red'), c('yellow'), 0.50, c('yellow'), c('green')])
        
        data = self.Agefrac if AgeMethod=='Relative Age' else self.Age
        ax.scatter(xx, tt,c=data, lw=0, cmap=cm, marker='s')
        ax.set_xlim([0, self.N])
        ax.set_ylim([0, self.T])
        ax.set_title(self.method)
        ax.invert_yaxis()
                            
    def AnalyzeFitness(self,T0,dt,ax):
        #Plot the distribution of the fitness by taking samples every dt from T0 on
        A = []
        for i in range(int((self.T-T0)/dt)):
            A =np.concatenate((A,self.X[T0+i*dt]))
        
        y,binEdges=np.histogram(A,50, density=True)
        bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
        ax.plot(bincenters,y,'-',c = self.color[self.method])
 
    def AnalyzeRange(self,T0,ax):
        #Anayze the scale behavior of the system by looking at the length and duration of branches and the lifetimes of a species
    
        #Determine the branches by setting up a matrix where species originate
        Pos = np.zeros((self.T,self.N))
        R  = np.argwhere(self.Age==0)
        Pos[R[:,0], R[:,1]] = 1 
        R  = np.argwhere(Pos)
        
        c = mcolors.ColorConverter().to_rgb
        cm = make_colormap([c('black'),  c('white')])
        fig = plt.figure()
        axBranch = fig.add_subplot(1,1,1)
        xx, tt  =np.meshgrid(np.arange(0, self.N, 1), np.arange(0, self.T, 1))
        axBranch.scatter(xx, tt,c=Pos, lw=0, cmap=cm, marker='s', s=20) 

        #A branch is supposed to start if there are three originating species at a time t that did not originate in the previoues timestep        
        StartBranch = []        
        for t,x in R:
            if np.sum(Pos[t].take(range(x,x+3), mode='wrap'))==3 and np.sum(Pos[t-1].take(range(x,x+3), mode='wrap')) ==0 and t>T0 and t+1<self.T:
                StartBranch.append([t,x])     
        
     
        #Determine the length and duration of a branch by looking at where it stops. 
        #A branch is supposed to continue of a species originates in the next timestep that has already originated before in the branch
        BranchLen = []
        BranchTime =[]
        for t0, xmin in StartBranch:
            t = t0
            xmax = xmin + 2
            while np.sum(Pos[t+1].take(range(xmin,xmax), mode='wrap'))!=0 and t+2<self.T:
                X = np.argwhere(Pos[t+1].take(range(xmin-2,xmax+2), mode='wrap'))+(xmin-2)
                xmin = min(xmin, np.min(X))
                xmax = max(xmax, np.max(X))
                t +=1
                
            BranchLen.append(xmax-xmin+1)
            BranchTime.append(t-t0)
            
            axBranch.plot([xmin, xmax],[t0,t0], 'r')
            axBranch.plot([xmin,xmin],[t0,t], 'b')
            

        axBranch.invert_yaxis()
        axBranch.set_xlabel('Species')
        axBranch.set_ylabel('Time')
        

        nBins = 20
               
        #Determine and plot the length of the branches and fit them to a power law distribution
        ax[0].set_xlabel('Branch length')
        ax[0].set_ylabel('Probability')    
        D, bins = np.histogram(np.array(BranchLen), bins=np.logspace(0, np.log10(max(BranchLen )), nBins), density=True)
        popt = FitToPowerlaw(D, bins)
        print('Fitting parameters size:', popt[0])
        ax[0].loglog(bins[0:-1], D, ls='None', marker='.',c = self.color[self.method])
        ax[0].loglog(bins[0:-1], bins[0:-1]**popt[0]*10**popt[1],c = self.color[self.method])
    
        #Determine and plot the duration of the branches and fit them to a power law distribution
        ax[1].set_xlabel('Branch Time')
        ax[1].set_ylabel('Probability')
        D, bins = np.histogram(np.array(BranchTime), bins=np.logspace(0, np.log10(max(BranchTime)), nBins), density=True)
        popt = FitToPowerlaw(D, bins)
        print('Fitting parameters time', popt[0])
        ax[1].loglog(bins[0:-1], D, ls='None', marker='.',c = self.color[self.method])
        ax[1].loglog(bins[0:-1], bins[0:-1]**popt[0]*10**popt[1],c = self.color[self.method])

        #Determine and plot the lifetimes of the species and fit them to a power law distribution
        ax[2].set_xlabel('Lifetime')
        ax[2].set_ylabel('Probability')
        D, bins = np.histogram(np.array(self.Lifetime), bins=np.logspace(0, np.log10(max(self.Lifetime)), nBins), density=True)     
        popt = FitToPowerlaw(D, bins)
        print('Fitting parameters time', popt[0])
        ax[2].loglog(bins[0:-1], D, ls='None', marker='.',c = self.color[self.method])
        ax[2].loglog(bins[0:-1], bins[0:-1]**popt[0]*10**popt[1],c = self.color[self.method])



    def AnalyzeNoise(self,T0,ax):
        #Determine the frequency power spectrum by looking at three different properties in time and Fourier transform them to the frequency domain
        nBins = 20
        
        #Behavior of the mean of the fraction of the lifetimes in time corrected for the condition at t=0 and t=T
        Noisefrac = np.mean(self.Agefrac[T0/2:-T0/2],axis=1)
        ax[0].plot(Noisefrac)        
        f, Sf = logFFT(Noisefrac, 1, nBins)
        ax[1].loglog(f, abs(Sf), marker='.', ls='none',c = self.color[self.method])
        ax[1].set_xlabel('Frequency')
        ax[1].set_ylabel('Relative Age')
        popt = FitToPowerlaw(abs(Sf), f)
        ax[1].loglog(f, f**popt[0]*10**popt[1],c = self.color[self.method])  
        print('Relative Age', popt[0])      
        
        #Behavior of the mean of the age in time corrected for the condition at t=0 and t=T
        NoiseAge = np.mean(self.Age[T0/2:-T0/2],axis=1)
        ax[2].plot(NoiseAge)        
        ff, Sff = logFFT(NoiseAge, 1, nBins)
        ax[3].loglog(ff, abs(Sff), marker='.', ls='none',c = self.color[self.method])
        ax[3].set_xlabel('Frequency')
        ax[3].set_ylabel('Age')
        popt = FitToPowerlaw(abs(Sff), ff)
        ax[3].loglog(ff, ff**popt[0]*10**popt[1],c = self.color[self.method])  
        print('Age', popt[0])
 
        #Behavior of the mean of the fitness in time corrected for the conditation at t=0 only
        NoiseMean = np.mean(self.X[T0:],axis=1)
        ax[4].plot(NoiseMean)        
        fff, Sfff = logFFT(NoiseMean, 1, nBins)
        ax[5].loglog(fff, abs(Sfff), marker='.', ls='none',c = self.color[self.method])
        ax[5].set_xlabel('Frequency')
        ax[5].set_ylabel('Fitness')
        popt = FitToPowerlaw(abs(Sfff), fff)
        ax[5].loglog(fff, fff**popt[0]*10**popt[1],c = self.color[self.method])  
        print('Fitness', popt[0])       

    
if __name__ == '__main__':
    figNoise = plt.figure()
    figRange = plt.figure()
    figProbability = plt.figure()
    figPlotAges = plt.figure()
    
    axNoiseTime = figNoise.add_subplot(3,2,1)

    axNoiseFreq = figNoise.add_subplot(3,2,2)

    
    axNoiseTime1 = figNoise.add_subplot(3,2,3)

    axNoiseFreq1 = figNoise.add_subplot(3,2,4)

    
    axNoiseTime2 = figNoise.add_subplot(3,2,5)

    axNoiseFreq2 = figNoise.add_subplot(3,2,6)

    
    axLength = figRange.add_subplot(1,3,1)
    axLength.set_xlabel('Length')
    axLength.set_ylabel('Probability')
    axTime   = figRange.add_subplot(1,3,2)
    axTime.set_xlabel('Time')
    axTime.set_ylabel('Probability') 
    axLifetime   = figRange.add_subplot(1,3,3)
    axLifetime.set_xlabel('Lifetime')
    axLifetime.set_ylabel('Probability') 
    
    axProbability = figProbability.add_subplot(1,1,1)
    axProbability.set_xlabel('Fitness')
    axProbability.set_ylabel('Probability')
    
    T0 =50000
    for i, method in enumerate(['AllZero', 'AllOne', 'Random']):  
        print(method)
        
        #start and simulate population
        Baksteppen = BakSteppenModel(N=100)
        Baksteppen.Initialize(method)
        Baksteppen.Simulate(T=200000)
        
        #Plot ages
        axPlotAges = figPlotAges.add_subplot(1,1,i+1)
        Baksteppen.PlotAges(axPlotAges)
    
        #Analyze the frequency behavior
        Baksteppen.AnalyzeNoise(T0=T0,ax=[axNoiseTime,axNoiseFreq,axNoiseTime1,axNoiseFreq1,axNoiseTime2, axNoiseFreq2])

       
        #Analyze the Time and Length scale and Lifetime
        Baksteppen.AnalyzeRange(T0=T0 ,ax=[axLength, axTime,axLifetime])    
    
        #Fitness
        Baksteppen.Probability(T0=T0,dt=125,ax = axProbability)
    

    plt.show()
    