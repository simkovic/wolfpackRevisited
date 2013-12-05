
# -*- coding: utf-8 -*-
from psychopy import visual, core, event,gui,sound,parallel
from psychopy.misc import pix2deg, deg2pix
import time, sys,os
import numpy as np

from Settings import Q
from Constants import *
try: from eyetracking.Tobii import TobiiController
except: print 'Tobii import failed'

 
class Gao10e3Experiment():
    def __init__(self,of=None):
        # ask infos
        myDlg = gui.Dlg(title="Experiment zur Bewegungswahrnehmung",pos=Q.guiPos)   
        myDlg.addText('VP Infos')   
        myDlg.addField('Subject ID:',0)
        myDlg.addField('Block:',0)
        myDlg.addField('Alter:', 21)
        myDlg.addField('Geschlecht (m/w):',choices=(u'weiblich',u'maennlich'))
        myDlg.addField(u'Händigkeit:',choices=('rechts','links'))
        myDlg.addField(u'Dominantes Auge:',choices=('rechts','links'))
        myDlg.addField(u'Sehschärfe: ',choices=('korrigiert','normal'))
        myDlg.addField(u'Wochenstunden vor dem Komputerbildschirm:', choices=('0','0-2','2-5','5-10','10-20','20-40','40+'))
        myDlg.addField(u'Wochenstunden Komputerspielen:', choices=('0','0-2','2-5','5-9','10-20','20+'))
        myDlg.addField('Starte bei Trial:', 0)
        myDlg.addField(u'Stimulus:',choices=('dart','eyes'))
        myDlg.show()#show dialog and wait for OK or Cancel
        vpInfo = myDlg.data
        if myDlg.OK:#then the user pressed OK
            subinf = open(Q.outputPath+'vpinfo.res','a')
            subinf.write('%d\t%d\t%d\t%s\t%s\t%s\t%s\t%s\t%s\t%d\t%s\n'% tuple(vpInfo))
            subinf.close()               
        else: 
            print 'Experiment cancelled'
            return
        self.id=vpInfo[0]
        self.block=vpInfo[1]
        self.initTrial=vpInfo[-2]
        self.isDart= vpInfo[-1] == 'dart'
        # save settings, which we will use
        Q.save(Q.inputPath+'vp%03d'%self.id+Q.delim+'SettingsExp.pkl')
        if of==None: self.output = open(Q.outputPath+'vp%03d.res'%self.id,'a')
        else: self.output = open(Q.outputPath+of,'a')
        #init stuff
        self.wind=Q.initDisplay()
        # init text
        fs=1 # font size
        self.text1=visual.TextStim(self.wind,text='Error',wrapWidth=30,pos=[0,2])
        self.text2=visual.TextStim(self.wind,text='Error',wrapWidth=30,pos=[0,0])
        self.text3=visual.TextStim(self.wind, text='Error',wrapWidth=30,pos=[0,-10])
        self.text1.setHeight(fs)
        self.text2.setHeight(fs)
        self.text3.setHeight(fs)
        self.f=0
        self.permut=np.load(Q.inputPath+'vp%03d'%self.id+Q.delim
            +'ordervp%03db%d.npy'%(self.id,self.block))
        if len(self.permut.shape)>1 and self.permut.shape[1]>1:
            self.data=self.permut[:,1:]
            self.permut=self.permut[:,0]
        self.nrtrials=self.permut.size
    def flip(self):
        if self.block!=2: self.area.draw()
        mpos=self.mouse.getPos()
        if np.linalg.norm(mpos)>11.75:
            phi=np.arctan2(mpos[Y],mpos[X])
            self.mouse._setPos([np.cos(phi)*11.75,np.sin(phi)*11.75])
        mpos=self.mouse.getPos()
        self.chasee[self.f,:]=mpos
        #print self.t, self.trialType.shape,self.quadMaps.shape 
        qmap=self.quadMaps[int(self.trialType[self.t])]
        self.oris=np.arctan2(mpos[Y]-self.pos[:,Y],mpos[X]-self.pos[:,X])
        for i in range(4):
            if qmap[i]>0: self.oris[3*i:3*(i+1)]-=np.pi/2.0
        epos = np.concatenate([self.pos,self.pos])
        for i in range(self.cond):
            epos[i,X]+= np.cos(self.oris[i]+0.345)*0.71; epos[i,Y]+= np.sin(self.oris[i]+0.345)*0.71
            epos[i+self.cond,X]+= np.cos(self.oris[i]-0.345)*0.71; epos[i+self.cond,Y]+= np.sin(self.oris[i]-0.345)*0.71
        self.epos=epos
        self.eyes.setXYs(self.epos)
        self.elem.setOris(self.oris/np.pi*180)
        if self.repmom: 
            rm= np.array([np.cos(self.oris),np.sin(self.oris)]).T*self.repmom
            self.elem.setXYs(self.pos+rm)
        dist= np.sqrt(np.power(self.pos-np.matrix(mpos),2).sum(1))
        sel=np.array(dist).squeeze() <Q.agentSize/2.0+self.chradius
        clr=np.ones((self.cond,3))
        if np.any(sel): 
            clr[sel,1]=-1;clr[sel,2]=-1
            if not self.tone.onn and self.block<2 and self.id>350: 
                self.tone.play();self.tone.onn=True
        elif self.tone.onn: self.tone.stop(); self.tone.onn=False
        if self.f==Q.nrframes-1 and self.tone.onn: self.tone.stop(); self.tone.onn=False
        if self.block!=2: self.elem.setColors(clr)
        self.elem.draw()
        if self.repmom==0 and not self.isDart: self.eyes.draw()
        self.mouse.draw()
        self.wind.flip()
            
    def getJudgment(self):
        qmap=self.quadMaps[int(self.trialType[self.t])]
        if self.block!=2:
            self.mouse.clickReset()
            self.mouse.setPointer(self.pnt2)
            mkey=self.mouse.getPressed()
            mpos=self.mouse.getPos()
            self.pnt1.setPos(mpos)
            # find three nearest agents 
            ags=np.argsort((np.power(np.array(mpos,ndmin=2)-self.pos,2)).sum(1))
            ags=ags[:3]
            # ensure that the agent does not overlap
            valid1=[]
            for ag in ags:
                accept=True
                for a in range(self.pos.shape[0]):
                    if ag!=a and np.linalg.norm(self.pos[ag,:]-self.pos[a,:])<Q.agentSize: accept=False
                if accept: valid1.append(ag)
            # choose wolfpack and perpendicular equally often
            valid2=[]
            for ag in ags:
                if (self.wolfcount> self.t/2.0 and qmap[ag/3] or
                    self.wolfcount<= self.t/2.0 and qmap[ag/3]==0):
                    valid2.append(ag)
            valid = list(set(valid1) & set(valid2))
            if not valid: # select at random 
                #print 'search failed'
                ag=ags[np.random.randint(3)]
            elif valid:
                ag=valid[np.random.randint(len(valid))]
            elif valid1:
                ag=valid1[np.random.randint(len(valid1))]
            elif valid2:
                ag=valid2[np.random.randint(len(valid2))]
            else: print 'does never happen'
            if qmap[ag/3]==0: self.wolfcount+=1
            #print self.oris[ag],self.oris.shape
            self.pos[ag,:]=np.inf # make it disappear
            self.epos[self.cond+ag,:]=np.inf; self.epos[ag,:]=np.inf
            self.eyes.setXYs(self.epos)
            if self.repmom: 
                rm= np.array([np.cos(self.oris),np.sin(self.oris)]).T*self.repmom
                self.elem.setXYs(self.pos+rm)
            else: self.elem.setXYs(self.pos)
            while sum(mkey)==0: # wait for subject response
                self.area.draw()
                self.elem.draw()
                if self.repmom==0 and not self.isDart:  self.eyes.draw()
                self.pnt1.draw()
                self.mouse.draw()
                self.wind.flip()
                mkey=self.mouse.getPressed()
            mpos=self.mouse.getPos()
            self.output.write('\t%d\t%.3f\t%d\t%.3f\t%.3f\t%.3f\t%d'%(self.trialType[self.t],
                self.repmom,ag, self.oris[ag],mpos[0],mpos[1],int(qmap[ag/3]==0)))
        else: 
            self.output.write('\t%d\t%.3f\t%d\t%d\t%.3f\t%.3f\t-1'%(self.trialType[self.t],
                self.repmom,self.asel[0],self.asel[1],self.oris[self.asel[0]],self.oris[self.asel[1]]))
        np.save(Q.inputPath+'vp%03d/chsVp%03db%dtrial%03d.npy' % (self.id,self.id,self.block,self.t),self.chasee)
        return False
    def runTrial(self,replay=False):
        self.repmom=self.manipType[self.t]
        print self.t
        if self.block==2:
            q=np.random.permutation(4)
            q0= np.array(self.quadMaps[int(self.trialType[self.t])]).nonzero()[0]
            q1=(np.array(self.quadMaps[int(self.trialType[self.t])])==0).nonzero()[0]
            a1=q0[np.random.randint(2)]*3+np.random.randint(3)
            a2=q1[np.random.randint(2)]*3+np.random.randint(3)
            self.asel=(a1,a2)
            if replay: 
                self.asel=self.aas[self.t,1:]
                a1=self.asel[0];a2=self.asel[1]
                self.trialType[self.t]=self.aas[self.t,0]
            for a in range(self.cond):
                if a!=a1 and a!=a2:
                    self.traj[:,a,:]=np.nan
            mpos=np.squeeze(self.traj[0,a1,:2]+self.traj[0,a2,:2])/2.0
        lim=1000
        if replay:
            self.mouse=MouseFromData(self.id,self.block,self.t,self.pnt2)
        else:
            self.mouse=visual.CustomMouse(self.wind, leftLimit=-lim,rightLimit=lim,
                topLimit=lim,bottomLimit=-lim,pointer=self.pnt2)
        self.chasee=np.zeros((Q.nrframes,2))*np.nan
        self.bringMouseToPosition()
        self.mouse.setPointer(self.pnt1)
        
        self.nrframes=self.traj.shape[0]
        self.cond=self.traj.shape[1]
        if self.isDart: mask=DART
        else: mask = CIRCLE
        self.elem=visual.ElementArrayStim(self.wind,fieldShape='sqr',
            nElements=self.cond, sizes=Q.agentSize,interpolate=True,
            elementMask=mask,elementTex=None,colors=Q.agentCLR)
        self.mouse.clickReset()
        event.clearEvents() #agents[a].setLineColor(agentCLR)
        self.f=0
        while self.f<self.nrframes:
            self.pos=self.traj[self.f,:,[X,Y]].transpose()
            self.phi=self.traj[self.f,:,PHI].squeeze()
            self.elem.setXYs(self.pos)
            self.flip()
            # check for termination signal
            for key in event.getKeys():
                if key in ['escape']:
                    self.wind.close()
                    core.quit()
                    sys.exit()
            self.f+=1
        self.output.write('%d\t%d\t%d\t%s'%(self.id,self.block,self.t,int(self.permut[self.t])))
        self.getJudgment()
        self.wind.flip()
        core.wait(1.0)
        self.output.write('\n')
        self.output.flush()
        
    def bringMouseToPosition(self):
        if self.block!=2:
            mpos=np.matrix(np.random.rand(2)*23.5-11.75)
            pos0=np.copy(self.traj[0,:,:2])
            while (np.any(np.sqrt(np.power(mpos-pos0,2).sum(1))<Q.initDistCC) or 
                np.linalg.norm(mpos)>11.75):
                mpos=np.matrix(np.random.rand(2)*23.5 -11.75)
            # ask subject to bring the mouse on the position
            self.mouse.clickReset()
            mpos=np.array(mpos).squeeze()
        else: mpos=np.squeeze(self.traj[0,self.asel[0],:2]+self.traj[0,self.asel[1],:2])/2.0
        self.pnt1.setPos(mpos)
        mkey=self.mouse.getPressed()
        mp=np.array([np.inf,np.inf]); pN=0
        while np.linalg.norm(mp-mpos)>self.pnt1.radius:
            while np.sum(mkey)==pN:
                if self.block!=2: self.area.draw()
                self.pnt1.draw()
                self.mouse.draw()
                self.wind.flip()
                mkey=self.mouse.getClicks()
            pN+=1
            mp=self.mouse.getPos()
    def run(self):
        self.quadMaps=[[1,1,0,0],[0,0,1,1],[0,1,0,1],[1,0,1,0],[1,0,0,1],[0,1,1,0]]
        self.tone=sound.SoundPygame(value='A',secs=5)
        self.tone.onn=False
        if not self.isDart: manip=[0.05,0.1,0.15,0.2,0.25,0.3]
        elif self.id<371: manip=[-0.2,-0.4]
        else: manip=[0.2,0.4]
        if self.block==1:
            NT= 2; temp=[]
            for n in range(NT): temp.append(np.ones(self.nrtrials/NT)*manip[n])
            self.manipType=np.concatenate(temp)
            self.manipType=self.manipType[np.random.permutation(self.nrtrials)]
        elif self.block==2:
            self.manipType=np.zeros(self.nrtrials)
            N=24;inc=N/len(manip)
            temp=np.zeros(N)
            for n in range(len(manip)):
                temp[inc*n:inc*(n+1)]=manip[n]
            temp=temp[np.random.permutation(N)]
            self.manipType[(self.nrtrials-N):]=temp
        else: self.manipType=np.zeros(self.nrtrials)
        self.chradius=0.6
        self.wolfcount=0
        self.area=visual.Circle(self.wind,radius=11.75,fillColor=None,lineColor='gray',edges=128,interpolate=False)
        fname='vp300trial000.npy'
        traj= np.load(Q.inputPath+'vp300'+Q.delim+fname)
        self.cond=traj.shape[1]
        self.eyes= visual.ElementArrayStim(self.wind,fieldShape='sqr',
            nElements=self.cond*2, sizes=0.38,interpolate=True,
            elementMask='circle',elementTex=None,colors='red')
        NT= 6; temp=[]
        for n in range(NT): temp.append(np.ones(self.nrtrials/NT)*n)
        self.trialType=np.concatenate(temp)
        self.trialType=self.trialType[np.random.permutation(self.nrtrials)]
        self.pnt1=visual.Circle(self.wind,radius=self.chradius,fillColor='green',lineColor=None,interpolate=False)
        self.pnt2=visual.ShapeStim(self.wind,interpolate=False, 
            vertices=[(-0.5,0),(-0,0),(0,0.5),(0,0),(0.5,0),(0,-0),(0,-0.5),(-0,0), (-0.5,0)],
            closeShape=False,lineColor='gray')
        # loop trials
        for trial in range(self.initTrial,self.nrtrials):   
            self.t=trial
            fname='vp300trial%03d.npy' % self.permut[trial]
            self.traj= np.load(Q.inputPath+'vp300'+Q.delim+fname)
            self.runTrial()
        if self.block==2: self.runBlock3()  
        self.output.close()
        self.text1.setText(u'Der Block ist nun zu Ende.')
        self.text1.draw()
        self.wind.flip()
        core.wait(10)
        self.wind.close()

    def runBlock3(self):
        self.block=3
        self.text1.setText(u'Der Versuch ist fast zu Ende.')
        self.text1.draw()
        self.wind.flip()
        core.wait(2)
        shp=['Kreis','Pfeil']
        self.text1.setText(u'Gleich sehen Sie einen %s verschwinden'%shp[int(self.isDart)])
        self.text3.setText(u'Bitte zeigen Sie seine letzte Position an')
        self.text1.draw(); self.text3.draw()
        self.wind.flip()
        core.wait(3)
        self.mouse.setPointer(self.pnt2)
        self.pos=np.ones((self.cond,2))*50
        self.oris=np.zeros(self.pos.shape[0]) 
        for k in range(13):
            if k>4: self.text3.setText(u'Bitte klicken Sie die Position wo der %s steht an'%shp[int(self.isDart)])
            if self.isDart: self.oris[0]=(2*np.random.rand()-1)*180
            else: self.oris[0]=(2*np.pi*np.random.rand()-np.pi)
            self.pos[0,Y]=0
            self.pos[0,X]=3*(k%5)-6
            
            self.elem.setXYs(self.pos)
            self.elem.setOris(self.oris)
            epos = np.concatenate([self.pos,self.pos])
            epos[0,X]+= np.cos(self.oris[0]+0.345)*0.71
            epos[0,Y]+= np.sin(self.oris[0]+0.345)*0.71
            epos[0+self.cond,X]+= np.cos(self.oris[0]-0.345)*0.71
            epos[0+self.cond,Y]+= np.sin(self.oris[0]-0.345)*0.71
            self.text3.draw()
            self.elem.draw()
            if k<10 and not self.isDart: 
                self.eyes.setXYs(epos)
                self.eyes.draw()
            self.wind.flip()
            core.wait(2+np.random.rand())
            self.mouse.clickReset()
            mkey=self.mouse.getPressed()
            while sum(mkey)==0: # wait for subject response
                if k>=5: 
                    self.elem.draw()
                    if k<10 and not self.isDart: self.eyes.draw()
                self.mouse.draw()
                self.text3.draw()
                self.wind.flip()
                mkey=self.mouse.getPressed()
            mpos=self.mouse.getPos()
            self.output.write('%d\t%d\t%d\t%.3f\t%.3f\t%.3f\t-1\t-1\t-1\t-1\t-1\n'%(self.id,self.block,k,self.oris[0],float(mpos[0]),float(mpos[1])))
    def getf(self):
        return self.f
        
class TobiiExperiment(Gao10e3Experiment):
    def __init__(self):
        Gao10e3Experiment.__init__(self)
        self.eyeTracker = TobiiController(self.wind,self.getf,sid=self.id,block=self.block)
        self.eyeTracker.sendMessage('Monitor Distance\t%f'% self.wind.monitor.getDistance())
        self.eyeTracker.doMain()
    def run(self):
        Gao10e3Experiment.run(self)
        self.eyeTracker.closeConnection()
    def runTrial(self,*args):
        self.eyeTracker.preTrial(False)#self.t,False,self.getWind(),autoDrift=True)
        self.eyeTracker.sendMessage('Trial\t%d'%self.t)
        Gao10e3Experiment.runTrial(self,*args)
        self.eyeTracker.postTrial()
    def getJudgment(self,*args):
        self.eyeTracker.sendMessage('Detection')
        resp=Gao10e3Experiment.getJudgment(self,*args)
        return resp 
    def omission(self):
        self.eyeTracker.sendMessage('Omission')
        Gao10e3Experiment.omission(self)


class MouseFromData():
    def __init__(self,vp,b,tr,pointer):
        self.vp=vp
        self.b=b
        self.tr=tr
        self.data=np.load(Q.inputPath+'vp%03d/chsVp%03db%dtrial%03d.npy' % (vp,vp,b,tr))
        self.f=0
        self.pointer=pointer

    def getPressed(self): return (1,1,1)
    def getClicks(self): return [np.inf,np.inf,np.inf]
    def clickReset(self): pass
    def draw(self):
        self.pointer.setPos(self.data[self.f,:2])
        self.pointer.draw()
        self.f+=1
    def getPos(self):
        return self.data[min(self.f,self.data.shape[0]-1),:2]
    def setPointer(self,pointer):
        self.pointer=pointer
    def _setPos(self,pos):
        print 'Warning setting data position',self.getPos(),pos
    
class DataReplay(Gao10e3Experiment):
    def __init__(self):
        Gao10e3Experiment.__init__(self,of='datareplay.res')
        dat=np.loadtxt(Q.outputPath+'vp%03d.res'%self.id)
        self.aas= dat[dat[:,1]==2,:]
        self.aas=self.aas[:,[4,6,7]]
    def flip(self):
        Gao10e3Experiment.flip(self)
        self.wind.getMovieFrame()
        self.wind.saveMovieFrames(Q.outputPath+'vid%03d.png'%(self.f))
        
    def getJudgment(self): pass
    def runBlock3(self): pass
    def bringMouseToPosition(self): pass
    def runTrial(self,*args):
        Gao10e3Experiment.runTrial(self,*args,replay=True)
        
        

if __name__ == '__main__':
    #E=TobiiExperiment()
    #E=Gao10e3Experiment()
    E=DataReplay()
    E.run()


