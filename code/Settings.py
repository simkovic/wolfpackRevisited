##    The MIT License (MIT)
##
##    Copyright (c) <2013> <Matus Simkovic>
##
##    Permission is hereby granted, free of charge, to any person obtaining a copy
##    of this software and associated documentation files (the "Software"), to deal
##    in the Software without restriction, including without limitation the rights
##    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
##    copies of the Software, and to permit persons to whom the Software is
##    furnished to do so, subject to the following conditions:
##
##    The above copyright notice and this permission notice shall be included in
##    all copies or substantial portions of the Software.
##
##    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
##    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
##    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
##    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
##    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
##    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
##    THE SOFTWARE.

from psychopy import visual,monitors
from psychopy.misc import pix2deg,deg2pix
from Constants import *
import pickle
from os import getcwd
from os import path as pth

class Settings():
    def __init__(self,monitor,os,trialDur,refreshRate,agentSize,phiRange,
        pDirChange,initDistCC,bckgCLR,agentCLR,mouseoverCLR,selectedCLR,aSpeed,
        guiPos,winPos,fullscr):
        '''
            monitor - psychopy.monitors.Monitor instance
            os - operating system, 1-windows, 0-linux
            trialDur - trial duration in seconds
            refreshRate - monitor refresh rate in Hz
            agentSize - size (diameter) of the agent in degrees of visual angle
            phiRange - the size of the window for the choice of the new
                motion direction in degrees, e.g. for phiRange=120 the new
                movement direction will be selected randomly from a uniform
                window between 60 degrees to the left and 60 degrees to the
                right of the old movement direction
            pDirChange - tuple with three values for chasee, chaser
                and distractor respectively
                each value gives the rate of direction changes
                for the respective agent given as the average
                number of direction changes per second
            initDistCC - tuple with two values that give the minimum
                and maximum initial distance at the start of each trial
                between the chaser and chasee
                the actual distance is chosen uniformly randomly as a
                value between the minimum and maximum
            bckgCLR - background color used for presentation
                tuple with three RGB value scale between [-1,1]
            agentCLR - gray-scaled value between [-1,1]
                gives the color of the agent
            mouseoverCLR - gray-scaled value between [-1,1]
                gives the color of the agent when mouse hovers over it
            selectedCLR - gray-scaled value between [-1,1]
                gives the color of the agent when the agent was selected
            aSpeed - agent speed in degrees per second
            guiPos - tuple with the gui position in pixels
            fullscr - if True presents experiment as fullscreeen
            winPos - position of the window for stimulus presentation
        '''
        self.refreshRate=float(refreshRate)
        self.monitor=monitor
        self.os=os
        self.fullscr=fullscr
        self.setTrialDur(trialDur)
        self.agentSize=agentSize
        self.phiRange=phiRange
        self.setpDirChange(pDirChange)
        self.initDistCC=initDistCC
        self.bckgCLR=bckgCLR
        self.agentCLR=agentCLR
        self.mouseoverCLR=mouseoverCLR
        self.selectedCLR=selectedCLR
        self.setAspeed(aSpeed)
        self.guiPos=guiPos
        self.winPos=winPos
        self.delim=pth.sep
        path = getcwd()
        path = path.rstrip('code')
        self.inputPath=path+"trajData"+self.delim
        self.outputPath=path+"behData"+self.delim
        self.stimPath=path+"stimuli"+self.delim
        self.agentRadius=self.agentSize/2.0
        self.fullscr=fullscr
##########################################################
# The following routines should be used to set the attributes

    def setTrialDur(self,td):
        self.trialDur=td
        self.nrframes=self.trialDur*self.refreshRate+1
    def setpDirChange(self,pDirChange):
        self.pDirChange=[pDirChange[CHASEE]/self.refreshRate,
             pDirChange[CHASER]/self.refreshRate,
             pDirChange[DISTRACTOR]/self.refreshRate]
    def setAspeed(self,aSpeed):  self.aSpeed=aSpeed/self.refreshRate
##########################################################  
    def initDisplay(self,sz=None):
        if sz==None: sz=(800,800)
        elif type(sz)==int: sz=(sz,sz)
        wind=visual.Window(monitor=self.monitor,fullscr=self.fullscr,
            size=sz,units='deg',color=self.bckgCLR,pos=self.winPos,
            winType='pyglet',screen=1)
        return wind
##########################################################
# Unit conversion routines (normed, pixels, degrees of visual angle)
    def norm2pix(self,xy):
        return (np.array(xy)) * np.array(self.monitor.getSizePix())/2.0
    def norm2deg(self,xy):
        xy=self.norm2pix(xy)
        return pix2deg(xy,self.monitor)
    def pix2deg(self,pix):
        return pix2deg(pix,self.monitor)
    def deg2pix(self,deg):
        return deg2pix(deg,self.monitor)
##########################################################
# load/save functionality
    def save(self,filepath):
        f=open(filepath,'wb')
        try: pickle.dump(self,f);f.close()
        except: f.close(); raise
    @staticmethod
    def load(filepath):
        f=open(filepath,'rb')
        try: out=pickle.load(f);f.close()
        except: f.close(); raise
        return out
##########################################################
# monitors, please define your own monitor
t60=monitors.Monitor('tobii', width=34, distance=50); t60.setSizePix((1280,1024))
##########################################################
# these are the settings used to conduct the experiment
tobiilab={'monitor' :   t60,
        'refreshRate':  75,                 
        'os':           WINDOWS,              
        'phiRange':     [90,90],          
        'agentSize':    1.9,                  
        'initDistCC':   [4.0 ,4.0],       
        'pDirChange':   [3.0,3.0,3.0],         
        'bckgCLR':      [-1,-1,-1],
        'agentCLR':     1,                  
        'mouseoverCLR': 0.5,                
        'selectedCLR':  -0.5,               
        'trialDur':     17,                 
        'aSpeed':       7.8,               
        'guiPos':       (-800,400),          
        'winPos':       (0,0),
        'fullscr':      True}
Q=Settings(**tobiilab)
