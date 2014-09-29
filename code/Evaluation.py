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

import numpy as np
import pylab as plt
import matplotlib as mpl
import pystan
import scipy.stats as stats
from scipy.stats import nanmean as mean
from scipy.stats import nanmedian as median
from scipy.stats import nanstd as std
from scipy.stats import scoreatpercentile as sap
from matustools.matusplotlib import *
import os
#some constants and settings
BASE=os.getcwd().rstrip('code')+os.path.sep
TRAJPATH=BASE+'trajData'+os.path.sep
BEHDPATH=BASE+'behData'+os.path.sep
FIGPATH=BASE+os.path.sep.join(['paper','fig',''])

X=0;Y=1;M=0;P=1;W=2
man=[-0.15,-0.05,0.05,0.1,0.15,0.2,0.25,0.3,-0.2,-0.4,0.4]
quadMaps=[[1,1,0,0],[0,0,1,1],[0,1,0,1],[1,0,1,0],[1,0,0,1],[0,1,1,0]]
T=42 # number of trials
TDUR=17.0 # trial duration in seconds
CHR=0.3 # radius of the green circle in deg
AR=0.95 # radius of distractor circles in deg
MAXD=4 # maximum accepted displacement magnitude in judgment task, inclusion criterion for stats
MONHZ=75 # monitor frame rate in hz
#FIGCOL=[3.27,4.86,6.83] # width of figure for a 1,1.5 and 2 column layout of plosone article
CoM=(1-3**0.5/2)*AR # shift of the origin due to center of mass (dart) 

def drawCircularAgent(pos,scale=1,eyes=True,ori=0,bcgclr=True,rc=[0.7]*3):
    ax=plt.gca();pos=np.array(pos)
    if bcgclr: ax.set_axis_bgcolor(rc)
    epos1=np.array([np.cos(0.345+ori), np.sin(0.345+ori)])*0.71*scale
    epos2=np.array([np.cos(-0.345+ori), np.sin(-0.345+ori)])*0.71*scale
    c=plt.Circle(pos,AR*scale,fc='white',fill=True,
            ec=rc,zorder=-3,clip_on=False)
    e1=plt.Circle(epos1+pos,0.19*scale,fc='red',fill=True,
            ec='red',zorder=-2,alpha=0.1,clip_on=False)
    e2=plt.Circle(epos2+pos,0.19*scale,fc='red',fill=True,
            ec='red',zorder=-2,alpha=0.1,clip_on=False)
    ax.add_patch(c)
    if eyes:
        ax.add_patch(e1)
        ax.add_patch(e2)

def drawDartAgent(pos,scale=1,ori=0,bcgclr=True,eyes=True,rc=[0.7]*3,fill=True):
    ax=plt.gca();pos=np.array(pos)
    t1=[np.cos(ori),np.sin(ori)]
    t2=[np.cos(ori+2*np.pi/3),np.sin(ori+2*np.pi/3)]
    t3=[np.cos(ori-2*np.pi/3),np.sin(ori-2*np.pi/3)]
    xy=np.array([t1,t2,[0,0],t3])*AR*scale+np.array(pos,ndmin=2)
    if bcgclr: ax.set_axis_bgcolor(rc)
    c=plt.Polygon(xy,fc='white',fill=fill,
            ec=rc,zorder=-3,clip_on=False)
    ax.add_patch(c)

def loadDataB12(vpn,correct=False):
    ''' vpn - subjects for which data will be loaded
        correct - if true undoes the manipulation from block 2
        the format of the output matrix is described in E1B1.ipynb
    '''
    D=np.zeros((len(vpn),2,T,11))*np.nan
    for i in range(len(vpn)):
        vp=vpn[i]
        B= np.loadtxt(BEHDPATH+'vp%03d.res'%vp)
        for b in range(2):
            c=B[B[:,1]==b,:]
            if c.shape[0]==0:
                print 'missing: subject %d, block %d'%(vpn[i],b)
                continue
            C=np.zeros((T,c.shape[1]))*np.nan
            C[:c.shape[0],:]=c
            # gao10 evaluation: compute mean proportion of time spent in wolfpack quadrants
            for t in range(c.shape[0]):
                ms=np.load(TRAJPATH+'vp%03d/'%vp+
                    'chsVp%03db%dtrial%03d.npy'%(vp,C[t,1],C[t,2]))
                qmap=quadMaps[int(C[t,4])]
                D[i,b,t,0]=np.logical_and(ms[:,X]>0, ms[:,Y]>0).mean()*qmap[0]
                D[i,b,t,0]+=np.logical_and(ms[:,X]<=0, ms[:,Y]>0).mean()*qmap[1]
                D[i,b,t,0]+=np.logical_and(ms[:,X]>0, ms[:,Y]<=0).mean()*qmap[2]
                D[i,b,t,0]+=np.logical_and(ms[:,X]<=0, ms[:,Y]<=0).mean()*qmap[3]
                D[i,b,t,0]=1-D[i,b,t,0]
                assert (qmap[int(C[t,6])/3]==0)==(C[t,-1]==1)
                D[i,b,t,1]=np.sqrt(np.power(np.diff(ms,axis=0),2).sum(1)).sum(0)/TDUR
            D[i,b,:,2]=C[:,-1]        
            # memory displacement
            posA=np.zeros((T,2))*np.nan # distractor position
            phiA=np.zeros(T)*np.nan # distractor direction
            posC=np.zeros((T,2))*np.nan # green circle position
            for t in range(c.shape[0]):
                D[i,b,t,3]=0
                fname='vp300trial%03d.npy' % C[t,3]
                traj=np.load(TRAJPATH+'vp300/'+fname)
                posA[t,:]=traj[-1,C[t,6],:2]
                phiA[t]=(traj[-1,C[t,6],2]-180)/180.0*np.pi
                
                ms=np.load(TRAJPATH+'vp%03d/'%vp+
                    'chsVp%03db%dtrial%03d.npy'%(vp,C[t,1],C[t,2]))
                posC[t,:]=ms[-1,:]
                for a in range(12):
                    D[i,b,t,3]+=(np.sqrt(np.power(traj[:,a,:2]-ms,
                                2).sum(1))<AR+CHR).mean()
            phiRM=C[:,7]# nose orientation
            if correct and np.any(C[:,5]!=0): # undo the manipulation
                corr= np.array([np.cos(phiRM)*C[:,5],np.sin(phiRM)*C[:,5]])
                posA+= corr.T
            phiC=np.arctan2(posC[:,Y]-posA[:,Y],posC[:,X]-posA[:,X]) # mouse ori
            d=C[:,[8,9]]-posA
            phiD= np.arctan2(d[:,Y],d[:,X]) # judgment displacement ori
            dist=np.sqrt(np.power(d,2).sum(1)) # judgment displacement magnitude
            sel=np.sqrt(np.power(posC-C[:,[8,9]],2).sum(1))>CHR
            sel=np.logical_and(dist<MAXD,sel)               
            D[i,b,:,4]=sel
            D[i,b,:,5]=dist
            D[i,b,:,6]=phiD
            D[i,b,:,7]=phiA 
            D[i,b,:,8]=phiC 
            D[i,b,:,9]=phiRM
            D[i,b,:,10]=C[:,5]
            # for latter estimation we remove zeros and ones
            owt=D[:,:,:,0]
            owt[owt>0.999]=0.999
            owt[owt<0.001]=0.001
            # note, now we have changed the values in D
    return D


def loadDataB345(vpn,exp=1,robust=False,correct=False):
    ''' vpn - subjects for which data will be loaded
        exp - 1==eyes, 2==darts
        robust - if true, the displacement for each trial is computed
            as frame-wise median, if false mean is used
        correct - if true undoes the manipulation of nominal displacement
        return three matrices B3,B4, B5
        B3 - nominal displacement 0, B4 - nonzero nominal displacement
        columns in B3 and B4 give 0,1- displacement from the mid-section 
        parallel (horizontal) and orthogonal (vertical) to the line,
        2- manipulated nominal displacement
        B5 - columns give position recalled by subject in each trial + orientation
    '''
    N=len(vpn);
    B3=np.zeros((N,3,T))*np.nan#  
    B5=np.zeros((N,13,3))*np.nan# collect data from block 4
    if robust: fs=np.median
    else: fs=np.mean
    for k in range(N):
        vp=vpn[k]
        D= np.loadtxt(BEHDPATH+'vp%03d.res'%vp)
        if np.any(D[:,1]==3):B5[k,:,:]=D[D[:,1]==3,3:6]
        if not np.any(D[:,1]==2): continue
        D=D[D[:,1]==2,:]
        traj=np.zeros((T,MONHZ*TDUR+1,3,2))
        for t in range(T):
            ms=np.load(TRAJPATH+'vp%03d/'%vp+
                'chsVp%03db%dtrial%03d.npy'%(vp,D[t,1],D[t,2]))
            traj[t,:,M,:]=ms
            fname='vp300trial%03d.npy' % D[t,3]
            tt=np.load(TRAJPATH+'vp300/'+fname)
            traj[t,:,P,:]=tt[:,D[t,6],:2]
            traj[t,:,W,:]=tt[:,D[t,7],:2]
        if correct: # shift 
            oriP=np.arctan(traj[:,:,M,Y]-traj[:,:,P,Y],traj[:,:,M,X]-traj[:,:,P,X])
            oriW=np.arctan(traj[:,:,M,Y]-traj[:,:,W,Y],traj[:,:,M,X]-traj[:,:,W,X])
            man=np.array(D[:,5],ndmin=2).T
            traj[:,:,P,X]+= man*np.cos(oriP+[+np.pi/2,-np.pi/2][exp-1])
            traj[:,:,P,Y]+= man*np.sin(oriP+[+np.pi/2,-np.pi/2][exp-1])
            traj[:,:,W,X]+= man*np.cos(oriW)
            traj[:,:,W,Y]+= man*np.sin(oriW)
        traj=traj[:,int(2*MONHZ):,:,:]# discard first 2 seconds   
        mid=traj[:,:,[P,W],:].mean(2)# real mid-point

        dist= np.sqrt(np.power(traj[:,:,M,X]-mid[:,:,X],2)+
                      np.power(traj[:,:,M,Y]-mid[:,:,Y],2))
        phiP=np.arctan2(traj[:,:,P,Y]-traj[:,:,W,Y],traj[:,:,P,X]-traj[:,:,W,X])
        phiM=np.arctan2(traj[:,:,M,Y]-mid[:,:,Y],traj[:,:,M,X]-mid[:,:,X])

        B3[k,X,:]=fs(np.cos(phiM-phiP)*dist*2,axis=1)
        B3[k,Y,:]=fs(np.sin(phiM-phiP)*dist*2,axis=1)
        B3[k,2,:]=D[:,5]
    hasEyes=B3[0,2,:]==0
    B4=B3[:,:,~hasEyes]
    B3=B3[:,:,hasEyes]
    return B3,B4,B5

def plotB1(D,exp=1):
    if exp==1: drawAgent=drawCircularAgent
    else: drawAgent=drawDartAgent
    clrs=getColors(D.shape[0])
    
    titles=['Raw Displacement','Representational Momentum',
           'Lazy Hand Gravity','Orientation']
    spid=[1,3,6,4]
    for i in range(D.shape[0]):
        k=0
        for phiK in [0,D[i,0,:,7],D[i,0,:,8],D[i,0,:,9]]:
            subplot(2,2,k+1)#,3,spid[k])
            plt.grid(axis='y')
            ax=plt.gca()                         
            phi=D[i,0,:,6]-phiK
            x=np.cos(phi)*D[i,0,:,5]
            y=np.sin(phi)*D[i,0,:,5]
            # plot wolves
            sel=np.logical_and(D[i,0,:,4],D[i,0,:,2]==1)
            plt.plot(x[sel],y[sel],'o',mfc=clrs[i],alpha=0.5,ms=3, mec='k',mew=0.15)
            #plot perpendicular
            sel=np.logical_and(D[i,0,:,4],D[i,0,:,2]==0)
            plt.plot(x[sel],y[sel],'o',mfc=clrs[i],alpha=0.5,ms=3, mec='k',mew=0.15)
            # display medians
            #if k==3:print np.median(x[sel])
            plt.plot(np.median(x[D[i,0,:,4]==1]),
                     np.median(y[D[i,0,:,4]==1]),
                     'x',mec=clrs[i],mew=1.5,ms=6,alpha=1,zorder=3)
            #plt.xlabel('x axis');plt.ylabel('y axis')
            plt.xlim([-2,2]);plt.ylim([-2,2])
            ax.set_xticks(range(-2,3))
            ax.set_yticks(range(-2,3))
            plt.title(titles[k])
            plt.plot([-10,10],[0,0],color='#262626',lw=0.5)
            plt.plot([0,0],[-10,10],color='#262626',lw=0.5)
            if k<2: ax.set_xticklabels([])
            if k%2==1: ax.set_yticklabels([]) 
            ax.set_aspect('equal')
            if k==3: drawAgent((2.5,0),scale=0.4,bcgclr=False)
            elif k==1:
                drawAgent((2.5,0),scale=0.4,bcgclr=False,eyes=False)
                drawAgent((2.55,0),scale=0.4,bcgclr=False,eyes=False)
                drawAgent((2.6,0),scale=0.4,bcgclr=False,eyes=False)
                drawAgent((2.65,0),scale=0.4,bcgclr=False,eyes=False)
            elif k==2:
                ax.add_patch(plt.Circle((2.3,0),radius=0.6*CHR,fc='g',alpha=0.2,clip_on=False))
            k+=1
            if i==0: plt.text(plt.xlim()[0]+0.1*(plt.xlim()[1]-plt.xlim()[0]),
                plt.ylim()[1]-0.1*(plt.ylim()[1]-plt.ylim()[0]), 
                str(unichr(65+k-1)),horizontalalignment='center',verticalalignment='center',
                fontdict={'weight':'bold'},fontsize=12)
    plt.subplots_adjust(left=0.07,bottom=0.05,top=0.95,hspace=0.12)
    
def plotB2reg(prefix=''):
    w=loadStanFit(prefix+'revE2B2LHregCa.fit')
    px=np.array(np.linspace(-0.5,0.5,101),ndmin=2)
    a1=np.array(w['ma'][:,4],ndmin=2).T+1
    a0=np.array(w['ma'][:,3],ndmin=2).T
    printCI(w,'ma')
    y=np.concatenate([sap(a0+a1*px,97.5,axis=0),sap(a0+a1*px[:,::-1],2.5,axis=0)])
    x=np.squeeze(np.concatenate([px,px[:,::-1]],axis=1))
    man=np.array([-0.4,-0.2,0,0.2,0.4])
    plt.plot(px[0,:],np.median(a0)+np.median(a1)*px[0,:],'red')
    #plt.plot([-1,1],[0.5,0.5],'grey')
    ax=plt.gca()
    ax.set_aspect(1)
    ax.add_patch(plt.Polygon(np.array([x,y]).T,alpha=0.2,fill=True,fc='red',ec='w'))
    y=np.concatenate([sap(a0+a1*px,75,axis=0),sap(a0+a1*px[:,::-1],25,axis=0)])
    ax.add_patch(plt.Polygon(np.array([x,y]).T,alpha=0.2,fill=True,fc='red',ec='w'))
    mus=[]
    for m in range(len(man)):
        mus.append(loadStanFit(prefix+'revE2B2LHC%d.fit'%m)['ma4']+man[m])
    mus=np.array(mus).T
    errorbar(mus,x=man)
    ax.set_xticks(man)
    plt.xlim([-0.5,0.5])
    plt.ylim([-0.6,0.8])
    plt.xlabel('Nominal Displacement')
    plt.ylabel('Perceived Displacemet')
    
def plotB5(B5s,vpns,clrs=None,exps=[1],suffix=''):
    tt=0
    for gg in range(len(B5s)):
        B5=B5s[gg]; exp=exps[gg];vpn=vpns[gg]
        
        titles=['Recalled Position','Selected Position','Selected Position']
        posx=np.array([-6,-3,0,3,6,-6,-3,0],ndmin=2)
        seln=~np.isnan(B5[:,0,0])
        a=np.zeros(B5.shape[1]); a[:5]=1
        b=np.zeros(B5.shape[1]);
        if exp==1: b[5:10]=1
        else: b[5:]=1
        c=np.zeros(B5.shape[1]); c[10:]=1
        sels=[a==1,b==1];kk=1
        if exp==1: sels.append(c==1)
        if clrs is None or len(clrs)!=B5.shape[0]: clrs=getColors(B5.shape[0])
        #if gg==1: clrs=getColors(28)[:18]
        #elif gg==2: clrs=getColors(28)[18:]
        res=[]
        for sel in sels:
            objs=[]
            v=[]
            subplot(len(B5s),3,gg*3+kk);plt.grid(axis='y');kk+=1
            if exp==1: drawCircularAgent(pos=(0,0),eyes=kk<4)
            else: drawDartAgent(pos=(0,0))
            plt.plot([-1,1],[0,0],color='#262626',label='_nolegend_',lw=0.5);
            plt.plot([0,0],[-1,1],color='#262626',lw=0.5,label='_nolegend_')
            dx=B5[:,sel,1]-posx[0,:sel.sum()]
            dy=B5[:,sel,2]
            mag=np.sqrt(np.power(dx,2)+np.power(dy,2))
            if exp==1: phi=np.arctan2(dy,dx)-B5[:,sel,0]
            else:  phi=np.arctan2(dy,dx)-np.pi*B5[:,sel,0]/180
            xn=np.cos(phi)*mag
            yn=np.sin(phi)*mag
            for i in range(B5.shape[0]):
                plt.plot(xn[i,:],yn[i,:],'o',mfc=clrs[i],
                         label='_nolegend_',mew=0.15,alpha=0.5,ms=3,mec='k')
                valid=np.sqrt(np.power(xn[i,:],2)+np.power(yn[i,:],2))<1.2
                v.extend(valid.tolist())
                if seln[i]: label=str(vpn[i])
                else:label='_nolegend_'
                plt.plot(np.median(xn[i,valid]),np.median(yn[i,valid]),'x',
                              label=label,mec=clrs[i],mew=1.5,ms=6,zorder=3)
                xn[i,~valid]=np.nan
            plt.gca().set_aspect('equal')
            plt.xlim([-1,1]);plt.ylim([-1,1])
            #if kk>2: plt.gca().set_yticklabels([])
            if gg==0: plt.title(titles[kk-2],fontsize=12)
            #print np.array(v).sum(),float(seln.sum()), len(v)/float(xn.shape[0])
            res.append([xn])
            xn=xn.flatten()[np.array(v)]
            
            #if gg==1: xxn=np.copy(xn)
            #elif gg==2: xn=np.concatenate([xxn,xn])
            #xn=median(xn,1)
            m=np.mean(xn)
            plt.text(plt.xlim()[0]+0.1*(plt.xlim()[1]-plt.xlim()[0]),
                     plt.ylim()[1]-0.1*(plt.ylim()[1]-plt.ylim()[0]), 
                    str(unichr(65+tt)),horizontalalignment='center',verticalalignment='center',
                    fontdict={'weight':'bold'},fontsize=12);tt+=1
            plt.plot([m,m],[-1,1],'--g',color='gray',label='_nolegend_',zorder=-2)
            sse=std(xn,bias=True)/xn.size**0.5
            res[-1].extend([m,sse,xn.size])
            er= sse* stats.t.ppf(0.975,xn.size)
            er=[m-2*er,m+2*er]#[sap(xn,25),sap(xn,75)]
            plt.gca().add_patch(plt.Rectangle([er[0],-5],er[1]-er[0],10,
                                    color='k',zorder=-2,alpha=0.1))
            print " %.3f CI [%.3f, %.3f]"%(m,er[0],er[1])
            if False:#kk==2:
                plt.legend(bbox_to_anchor=(0,0,1,1),loc=2,mode="expand",ncol=seln.sum()/2+1,
                    bbox_transform=plt.gcf().transFigure,frameon=False)
            if exp==2:
                x0=(1-3**0.5/2)*AR
                for i in range(2):
                    plt.plot([x0,x0],[-1,1],':',color='gray')
    return res

def plotB2Wreg():
    w=loadStanFit('revE2B2WHreg.fit')
    a1=np.array(w['ma1'],ndmin=2).T
    a0=np.array(w['ma0'],ndmin=2).T
    px=np.array(np.linspace(-0.5,0.5,101),ndmin=2)
    printCI(w,'ma0');printCI(w,'ma1')
    y=np.concatenate([sap(a0+a1*px,97.5,axis=0),sap(a0+a1*px[:,::-1],2.5,axis=0)])
    x=np.squeeze(np.concatenate([px,px[:,::-1]],axis=1))
    plt.plot(px[0,:],np.median(a0)+np.median(a1)*px[0,:],'red')
    #plt.plot([-1,1],[0.5,0.5],'grey')
    ax=plt.gca()
    ax.add_patch(plt.Polygon(np.array([x,y]).T,alpha=0.2,fill=True,fc='red',ec='red'))
    y=np.concatenate([sap(a0+a1*px,75,axis=0),sap(a0+a1*px[:,::-1],25,axis=0)])
    ax.add_patch(plt.Polygon(np.array([x,y]).T,alpha=0.2,fill=True,fc='red',ec='red'))
    mus=[];man=np.array([-0.4,-0.2,0,0.2,0.4])
    for m in range(len(man)):
        mus.append(loadStanFit('revE2B2WBB%d.fit'%m)['mmu'])
    mus=np.array(mus).T
    errorbar(mus,x=man)
    ax.set_xticks(man)
    plt.xlim([-0.5,0.5])
    plt.ylim([0.4,0.56])
    plt.xlabel('Nominal Displacement')
    plt.ylabel('Prop. of Time in Wolfpack Quadrants')
    
def plotB3(B3,clrs=None,exp=1):
    if clrs is None: clrs=getColors(B3.shape[0])
    if exp==1: drawAgent=drawCircularAgent
    else: drawAgent=drawDartAgent
    for i in range(B3.shape[0]):
        drawAgent((-2.3,0),bcgclr=False,scale=0.4)
        drawAgent((2,0),bcgclr=False,scale=0.4,ori=[-np.pi/2,np.pi/2][exp-1])
        plt.plot([-1.5,1.5],[0,0],color='#262626',lw=0.5);
        plt.plot([0,0],[-1.5,1.5],color='#262626',lw=0.5)
        plt.plot(B3[i,X,:],B3[i,Y,:],'o',mfc=clrs[i],mec='k',ms=3,mew=0.15,alpha=0.5)
        plt.plot(np.median(B3[i,X,:]),np.median(B3[i,Y,:]),'x',mec=clrs[i],mew=1.5,ms=6,zorder=3)
        plt.xlim([-1.5,1.5])
        plt.ylim([-1.5,1.5])
        ax=plt.gca()
        ax.set_clip_on(False)
        ax.set_aspect('equal')
        #plt.title('Displacement in Degrees')
        
def plotB3reg():
    w=loadStanFit('revE2B3BHreg.fit')
    printCI(w,'mmu')
    printCI(w,'mr')
    for b in range(2):
        subplot(1,2,b+1)
        plt.title('')
        px=np.array(np.linspace(-0.5,0.5,101),ndmin=2)
        a0=np.array(w['mmu'][:,b],ndmin=2).T
        a1=np.array(w['mr'][:,b],ndmin=2).T
        y=np.concatenate([sap(a0+a1*px,97.5,axis=0),sap(a0+a1*px[:,::-1],2.5,axis=0)])
        x=np.squeeze(np.concatenate([px,px[:,::-1]],axis=1))
        plt.plot(px[0,:],np.median(a0)+np.median(a1)*px[0,:],'red')
        #plt.plot([-1,1],[0.5,0.5],'grey')
        ax=plt.gca()
        ax.set_aspect(1)
        ax.add_patch(plt.Polygon(np.array([x,y]).T,alpha=0.2,fill=True,fc='red',ec='w'))
        y=np.concatenate([sap(a0+a1*px,75,axis=0),sap(a0+a1*px[:,::-1],25,axis=0)])
        ax.add_patch(plt.Polygon(np.array([x,y]).T,alpha=0.2,fill=True,fc='red',ec='w'))
        man=np.array([-0.4,-0.2,0,0.2,0.4])
        mus=[]
        for m in range(len(man)):
            mus.append(loadStanFit('revE2B3BH%d.fit'%m)['mmu'][:,b])
        mus=np.array(mus).T
        errorbar(mus,x=man)
        ax.set_xticks(man)
        plt.xlim([-0.5,0.5])
        plt.ylim([-0.4,0.8])
        #plt.xlabel('Manipulated Displacement')
        if b==0:
            plt.ylabel('Perceived Displacemet')
            plt.gca().set_yticklabels([])
        subplot_annotate()
    plt.text(-1.1,-0.6,'Nominal Displacement',fontsize=8);

def plotB3fit(fit,suffix='',pars=['mu','mmu']):
    plt.figure(figsize=(6,3))
    D=fit.extract()[pars[0]]
    errorbar(D[:,:,X],x=0.9)
    plt.xlabel('Subject')
    errorbar(D[:,:,Y],x=1.1,clr=(1,0.5,0.5))
    plt.ylabel('Displacement in Degrees')
    if pars[1] is '': return
    d=fit.extract()[pars[1]]
    print 'X: %.3f, CI %.3f, %.3f'%(d[:,0].mean(), sap(d[:,0],2.5),sap(d[:,0],97.5))
    print  'Y: %.3f, CI %.3f, %.3f'%(d[:,1].mean(), sap(d[:,1],2.5),sap(d[:,1],97.5))

def plotB4(B4,clrs=None,exp=1):
    if clrs is None: clrs=getColors(B4.shape[0])
    if exp==1: drawAgent=drawCircularAgent
    else: drawAgent=drawDartAgent
    plt.figure(figsize=(12,6))
    for i in range(B4.shape[0]):
        man=np.unique(B4[i,2,:])
        res=[]
        for m in man.tolist():
            res.append([B4[i,X,B4[i,2,:]==m],B4[i,Y,B4[i,2,:]==m]])
        res=np.array(res)
        for d in range(2):
            ax=plt.subplot(2,2,(i>4)+1+d*2);
            perturb=np.random.rand(B4.shape[2])*0.01-0.005
            ppl.scatter(ax,B4[i,2,:]+perturb,B4[i,d,:],color=clrs[i])
            ppl.plot(ax,man,res[:,d,:].mean(1),color=clrs[i])
            plt.xlim([man[0]-0.05,man[-1]+0.05])
            plt.ylim([-2,2])
            if d: plt.ylabel('y axis displacement')
            else: plt.ylabel('x axis displacement')

def plotVectors():
    plt.grid(axis='y')
    ax=plt.gca()
    ax.set_aspect('equal')
    ax.set_axis_off()
    plt.plot([-10,10],[0,0],color='gray',zorder=-3)
    plt.plot([0,0],[-10,10],color='gray',zorder=-3)
    plt.xlim([-1.6,2]);plt.ylim([-1.6,2])
    #drawCircularAgent((-0.2,0),eyes=False)
    ax.add_patch(plt.Circle((-0.2,0),radius=AR,ec=[0.75]*3,zorder=-3,fill=False))
    ax.add_patch(plt.Circle((-0.4,0),radius=AR,ec=[0.8]*3,zorder=-3,fill=False))
    ax.add_patch(plt.Circle((-0.6,0),radius=AR,ec=[0.85]*3,zorder=-3,fill=False))
    drawCircularAgent((0,0),ori=np.arctan2(1,1.5)-np.pi/2)
    ax.add_patch(plt.Circle((1.5,1),radius=CHR,fc='g',alpha=0.2))
    plt.arrow(0.3,0.2,0.36,0,head_width=0.08,lw=2,color='b',alpha=0.2,length_includes_head=True)
    plt.arrow(0.66,0.2,0,-0.36,head_width=0.08,lw=2,color='k',alpha=0.2,length_includes_head=True)
    plt.arrow(0.66,-0.16,0.2,-0.3,head_width=0.08,lw=2,color='r',alpha=0.2,length_includes_head=True)
    plt.arrow(0,0,0.3,0.2,head_width=0.08,lw=2,color='g',length_includes_head=True)
    plt.arrow(0,0,0.2,-0.3,head_width=0.08,lw=2,color='r',length_includes_head=True)
    plt.arrow(0,0,0.36,0,head_width=0.08,lw=2,color='b',length_includes_head=True)
    plt.arrow(0,0,0,-0.36,head_width=0.08,lw=2,color='k',length_includes_head=True)

    plt.plot([0.86],[-0.46],'+k',ms=16,mew=2)
    
def plotManipulation():
    plt.grid(axis='y')
    man=[[-0.15,-0.05,0.05,0.1,0.15,0.2],[0.05, 0.1, 0.15, 0.2, 0.25, 0.3]]
    man=[[-0.4,-0.2],[0.2,0.4]]
    drawDartAgent((0,0),ori=-np.pi)
    plt.xlim([-1,1]);plt.ylim([-1,1])
    plt.gca().set_aspect('equal')
    plt.plot([-1,1],[0,0],color='gray',zorder=-3)
    plt.plot([0,0],[-1,1],color='gray',zorder=-3)
    plt.plot(man[0],[0]*2,'ok')
    plt.plot(man[1],[0]*2,'xk',ms=8,mew=2)
    #plt.gca().set_yticks([])
    
def plotRotation():
    plt.grid(axis='y')
    ax=plt.gca()
    ax.set_aspect('equal')
    plt.plot([0,AR],[0,0],color='b',zorder=-3)
    plt.plot([0,0],[0,AR],color='r',zorder=-3)
    plt.xlim([-1,1]);plt.ylim([-1,1])
    drawDartAgent((0,0),ori=np.pi/2,bcgclr=False,rc='r',fill=False)
    drawDartAgent((0,0),ori=0,bcgclr=False,rc='b',fill=False)
    w= mpl.patches.Wedge((0,0),CoM,0,90,ec=None,fc='gray',fill=True,alpha=0.2)
    ax.add_patch(w)
    plt.plot([0],[0],'yo',ms=8)
    plt.plot([0],[CoM],'r+',ms=8,mew=2)
    plt.plot([CoM],[0],'b+',ms=8,mew=2)
    d=0.4
    plt.plot([0],[d],'rx',ms=8,mew=2)
    plt.plot([d],[0],'bx',ms=8,mew=2)
    w= mpl.patches.Wedge((0,0),d,0,90,ec=None,fc='gray',fill=True,alpha=0.2)
    ax.add_patch(w)
    
def plotGao(D):
    plt.grid(axis='y')
    n=7
    ci=0.26/17.0*stats.t.ppf(0.975,n-1)/(n**0.5)
    print ci
    r=plt.Rectangle((-4,0.4688-ci),width=50,height=2*ci,fc='red',alpha=0.2,ec='white')
    ax=plt.gca()
    ax.add_patch(r)
    plt.plot([-12,14],[0.4688,0.4688],'r')
    x=range(1,D.shape[0]+1)
    plt.plot(x,D[:,0,:,0].mean(1),'o',
        alpha=0.5, mec='k',mfc='#66C2A5',mew=0.15)
    plt.plot([-12,14],[0.5,0.5],color='#262626',lw=0.5)
    plt.xlim([0,14])
    ax.set_xticks(x)
    plt.xlabel('Subject')
    plt.ylabel('Prop. of Time in Wolfpack Quadrants')
    
def plotComp():
    #E=[]
    D=[]
    for i in [1,2]:
        if i==2:
            find= len(D)
            D.append(np.load('force.npy'))
        if i==2: w=loadStanFit('revE2B2LHregCa.fit')
        else: w=loadStanFit('revE1B1LH.fit')
        D.append(w['ma'][:,3])
        if i==2: 
            w=loadStanFit('revE%dB2WHreg.fit'%i)
            D.append((w['ma0']-0.5)/w['ma1'])
            printCI(D[-1])
            w=loadStanFit('revE%dB3BHreg.fit'%i)
        else:
            w=loadStanFit('revE1B3BH.fit')
        D.append(w['mmu'][:,0])
        if i==2:vpn=range(351,381); vpn.remove(369); vpn.remove(370)
        else: vpn=range(301,314)
        B3,B4,B5=loadDataB345(vpn,correct=False,exp=i)
        b4=plotB5([B5],[vpn],exps=[i])
        plt.close()
        D.append(b4[0])
        D.append(b4[1])
        #E.append(D)
    figure(size=2,aspect=0.66)
    clr=(0.2, 0.5, 0.6)

    for i in range(len(D)):
        if i in [0,1,5,6,7]:
            dat=[D[i].mean(),sap(D[i],2.5),sap(D[i],97.5),sap(D[i],25),sap(D[i],75) ]
        elif i==find:dat=D[i]
        else:
            err= D[i][2]* stats.t.ppf(0.975,D[i][3])
            err2= D[i][2]* stats.t.ppf(0.75,D[i][3])
            dat=[D[i][1],D[i][1]-err, D[i][1]+err,D[i][1]-err2, D[i][1]+err2]
        inc=[1,2][int(i>3)]
        plt.plot([dat[1],dat[2]],[i+inc,i+inc],color=clr)
        plt.plot([dat[3],dat[4]],[i+inc,i+inc],
            color=clr,lw=3,solid_capstyle='round')
        plt.plot([dat[0]],[i+inc],mfc=clr,mec=clr,ms=8,marker='+',mew=2)
    ax=plt.gca()
    plt.ylim([0,len(D)+2])
    plt.xlabel('Perceived Displacement')
    plt.xlim([-0.05,0.4])
    #subplot_annotate(loc=[0.95,0.9])
    plt.grid()
    plt.ylabel('Bugs'+' '*15+'Darts',fontsize=14)
    ax.set_yticks(range(1,len(D)+2))
    ax.set_yticklabels(['Location Recall','Distance Bisection',
        'Static Recall','Static Localization','','Mouse Position','Location Recall',
        'Leave-Me-Alone','Distance Bisection','Static Recall',
                    'Static Localization'])

def plotForce():
    figure(size=3,aspect=0.5)
    subplot(1,2,1)
    from EvalTraj import plotFF
    plotFF(vp=351,t=28,f=900,cm=0.6,foffset=8)
    subplot_annotate()
    
    subplot(1,2,2)
    for i in [1,2,3,4]:
        R=np.squeeze(np.load('Rdpse%d.npy'%i))
        R=stats.nanmedian(R,axis=2)[:,1:,:]
        dps=np.linspace(-1,1,201)[1:]
        plt.plot(dps,R[:,:,2].mean(0));
    plt.legend([0,0.1,0.2,0.3],loc=3) 
    i=2
    R=np.squeeze(np.load('Rdpse%d.npy'%i))
    R=stats.nanmedian(R,axis=2)[:,1:,:]
    mn=np.argmin(R,axis=1)
    y=np.random.randn(mn.shape[0])*0.00002+0.0438
    plt.plot(np.sort(dps[mn[:,2]]),y,'+',mew=1,ms=6,mec=[ 0.39  ,  0.76,  0.64])
    plt.xlabel('Displacement')
    plt.ylabel('Average Net Force')
    hh=dps[mn[:,2]]
    err=np.std(hh)/np.sqrt(hh.shape[0])*stats.t.ppf(0.975,hh.shape[0])
    err2=np.std(hh)/np.sqrt(hh.shape[0])*stats.t.ppf(0.75,hh.shape[0])
    m=np.mean(hh)
    print m, m-err,m+err
    np.save('force',[m, m-err,m+err,m-err2,m+err2])
    plt.xlim([-0.5,0.5])
    plt.ylim([0.0435,0.046])
    plt.grid(b=True,axis='x')
    subplot_annotate()

def saveFigures():
    vpna=range(301,314)
    B3a,B4,B5a=loadDataB345(vpna)
    vpnb=range(351,381); vpnb.remove(369); vpnb.remove(370)
    B3b,B4,B5b=loadDataB345(vpnb)

    figure(size=3)
    plotB5([B5a,B5b[:18,:,:],B5b[18:,:,:]],[vpna,vpnb[:18],vpnb[18:]],exps=[1,2,2])
    plt.savefig(FIGPATH+'b5')

    figure(size=1,aspect=1.4)
    subplot(2,1,1);plt.grid(axis='y')
    plotB3(B3a)
    plt.text(plt.xlim()[0]+0.1*(plt.xlim()[1]-plt.xlim()[0]),
             plt.ylim()[1]-0.1*(plt.ylim()[1]-plt.ylim()[0]), 
            str(unichr(65+0)),horizontalalignment='center',
            verticalalignment='center',fontdict={'weight':'bold'},fontsize=12)
    subplot(2,1,2);plt.grid(axis='y')
    plotB3(B3b,exp=2)
    plt.text(plt.xlim()[0]+0.1*(plt.xlim()[1]-plt.xlim()[0]),
             plt.ylim()[1]-0.1*(plt.ylim()[1]-plt.ylim()[0]), 
            str(unichr(65+1)),horizontalalignment='center',
            verticalalignment='center',fontdict={'weight':'bold'},fontsize=12)
    plt.subplots_adjust(top=0.95,bottom=0.05)
    plt.savefig(FIGPATH+'b3')

    D=loadDataB12(vpna)
    figure(size=2,aspect=0.9,tight_layout=False)
    plotB1(D)
    plt.savefig(FIGPATH+'b1')

    figure(size=2,aspect=0.7)
    plotB3reg()
    plt.savefig(FIGPATH+'b3reg')

    figure(size=1, aspect=1.15)
    plotB2reg()
    plt.savefig(FIGPATH+'b2reg')

    figure(size=1, aspect=0.8)
    plotB2Wreg()
    plt.savefig(FIGPATH+'lmaReg')

    figure(size=1)
    plotManipulation()
    plt.savefig(FIGPATH+'man')

    figure(size=1)
    plotRotation()
    plt.savefig(FIGPATH+'rot')

    figure(size=1,aspect=0.8,tight_layout=True)
    plotGao(D)
    plt.savefig(FIGPATH+'gao')

    figure(size=1)
    plotVectors()
    plt.savefig(FIGPATH+'vectors')

    plotForce()
    plt.savefig(FIGPATH+'force')

    plotComp()
    plt.savefig(FIGPATH+'compar')
    
def plotExp():    
    figure(size=2,aspect=0.35)
    for i in range(3):
        subplot(1,3,1+i)
        plt.gca().set_axis_off()
        plt.imshow(plt.imread(FIGPATH+'imb%d.png'%(i+1)))
        plt.text(plt.xlim()[0]+0.1*(plt.xlim()[1]-plt.xlim()[0]),
            plt.ylim()[1]-0.1*(plt.ylim()[1]-plt.ylim()[0]), 
            str(unichr(65+i)),horizontalalignment='center',color='w',
            verticalalignment='center',fontdict={'weight':'bold'},fontsize=12)
    plt.subplots_adjust(bottom=0,top=1,wspace=-0.5,left=0,right=1)
    plt.savefig(FIGPATH+'exp',bbox_inches='tight')


if __name__ == '__main__':
    mpl.rcParams['savefig.format'] = 'jpg'
    mpl.rcParams['savefig.dpi']=900
    saveFigures()
    #plotComp()
    #plt.savefig(FIGPATH+'compar')

