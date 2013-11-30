import numpy as np
import pylab as plt
import pystan
import prettyplotlib as ppl
import scipy.stats as stats
from scipy.stats import nanmean as mean
from scipy.stats import nanmedian as median
from scipy.stats import nanstd as std
from scipy.stats import scoreatpercentile as sap
from matusplotlib.plottingroutines import *
#some constants and settings
TRAJPATH='/home/matus/Desktop/research/wolfpackRevisited/trajData/'
BEHDPATH='/home/matus/Desktop/research/wolfpackRevisited/behData/'
FIGPATH='/home/matus/Desktop/research/wolfpackRevisited/paper/fig/'

X=0;Y=1;M=0;P=1;W=2
man=[-0.15,-0.05,0.05,0.1,0.15,0.2,0.25,0.3,-0.2,-0.4,0.4]
quadMaps=[[1,1,0,0],[0,0,1,1],[0,1,0,1],[1,0,1,0],[1,0,0,1],[0,1,1,0]]
T=42 # number of trials
TDUR=17.0 # trial duration in seconds
CHR=0.3 # radius of the green circle in deg
AR=0.95 # radius of distractor circles in deg
MAXD=4 # maximum accepted displacement magnitude in judgment task
MONHZ=75 # monitor frame rate in hz

def drawCircularAgent(pos,scale=1,eyes=True,ori=0,bcgclr=True,rc=0.9):
    ax=plt.gca();pos=np.array(pos)
    if bcgclr: ax.set_axis_bgcolor([0.9]*3)
    epos1=np.array([np.cos(0.345+ori), np.sin(0.345+ori)])*0.71*scale
    epos2=np.array([np.cos(-0.345+ori), np.sin(-0.345+ori)])*0.71*scale
    c=plt.Circle(pos,AR*scale,fc='white',fill=True,
            ec=[rc]*3,zorder=-3,clip_on=False)
    e1=plt.Circle(epos1+pos,0.19*scale,fc='red',fill=True,
            ec='red',zorder=-2,alpha=0.1,clip_on=False)
    e2=plt.Circle(epos2+pos,0.19*scale,fc='red',fill=True,
            ec='red',zorder=-2,alpha=0.1,clip_on=False)
    ax.add_patch(c)
    if eyes:
        ax.add_patch(e1)
        ax.add_patch(e2)

def drawDartAgent(pos,scale=1,ori=0,bcgclr=True,eyes=True,rc=0.9):
    ax=plt.gca();pos=np.array(pos)
    t1=[np.cos(ori),np.sin(ori)]
    t2=[np.cos(ori+2*np.pi/3),np.sin(ori+2*np.pi/3)]
    t3=[np.cos(ori-2*np.pi/3),np.sin(ori-2*np.pi/3)]
    xy=np.array([t1,t2,[0,0],t3])*AR*scale+np.array(pos,ndmin=2)
    if bcgclr: ax.set_axis_bgcolor([rc]*3)
    c=plt.Polygon(xy,fc='white',fill=True,
            ec=[rc]*3,zorder=-3,clip_on=False)
    ax.add_patch(c)

def loadDataB12(vpn,correct=False):
    ''' correct undoes the manipulation from block 2'''
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


def loadDataB345(vpn,robust=False,correct=False):
    N=len(vpn);
    B3=np.zeros((N,3,T))*np.nan#  0,1- displacement from the mid-section 
    #parallel and orthogonal to the line, 2- manipulated physical displacement
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
            traj[:,:,P,X]+= man*np.cos(oriP-np.pi/2)
            traj[:,:,P,Y]+= man*np.sin(oriP-np.pi/2)
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
#vpn=range(351,369)
#B3,B4,B5=loadDataB345(vpn,correct=True)

def plotB1(D,exp=1):
    if exp==1: drawAgent=drawCircularAgent
    else: drawAgent=drawDartAgent
    clrs=getColors(D.shape[0])
    plt.figure(figsize=(12,10))
    titles=['Raw Displacement','Representational Momentum',
           'Lazy Hand Gravity','Orientation Displacement']
    spid=[1,3,6,4]
    for i in range(D.shape[0]):
        k=0
        for phiK in [0,D[i,0,:,7],D[i,0,:,8],D[i,0,:,9]]:
            ax=plt.subplot(2,2,k+1)#,3,spid[k])
                         
            phi=D[i,0,:,6]-phiK
            x=np.cos(phi)*D[i,0,:,5]
            y=np.sin(phi)*D[i,0,:,5]
            #plot perpendicular
            sel=np.logical_and(D[i,0,:,4],D[i,0,:,2]==0)
            ppl.scatter(ax,x[sel],y[sel],color=clrs[i])
            # plot wolves
            sel=np.logical_and(D[i,0,:,4],D[i,0,:,2]==1)
            ppl.scatter(ax,x[sel],y[sel],color=clrs[i])
            # display medians
            plt.plot(np.median(x[D[i,0,:,4]==1]),
                     np.median(y[D[i,0,:,4]==1]),
                     'x',mec=clrs[i],mew=2,ms=9)
            plt.xlabel('x axis');plt.ylabel('y axis')
            plt.xlim([-2,2]);plt.ylim([-2,2])
            plt.title(titles[k])
            plt.plot([-10,10],[0,0],color='gray')
            plt.plot([0,0],[-10,10],color='gray')
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
    plt.savefig(FIGPATH+'e%db1.png'%exp,dpi=400)

def plotB5(B5,vpn,clrs=None,exp=1,suffix=''):
    titles=['Recalled Position','Selected Position','Selected Position']
    posx=np.array([-6,-3,0,3,6,-6,-3,0],ndmin=2)
    seln=~np.isnan(B5[:,0,0])
    a=np.zeros(B5.shape[1]); a[:5]=1
    b=np.zeros(B5.shape[1]);
    if exp==1: b[5:10]=1
    else: b[5:]=1
    c=np.zeros(B5.shape[1]); c[10:]=1
    sels=[a==1,b==1];kk=1
    plt.figure(figsize=(len(sels)*4,5))
    if exp==1: sels.append(c==1)
    if clrs is None: clrs=getColors(B5.shape[0])
    res=[]
    for sel in sels:
        objs=[]
        v=[]
        plt.subplot(1,len(sels),kk);kk+=1
        if exp==1: drawCircularAgent(pos=(0,0),eyes=kk<4)
        else: drawDartAgent(pos=(0,0))
        plt.plot([-1,1],[0,0],'gray',label='_nolegend_');
        plt.plot([0,0],[-1,1],'gray',label='_nolegend_')
        dx=B5[:,sel,1]-posx[0,:sel.sum()]
        dy=B5[:,sel,2]
        mag=np.sqrt(np.power(dx,2)+np.power(dy,2))
        if exp==1: phi=np.arctan2(dy,dx)-B5[:,sel,0]
        else:  phi=np.arctan2(dy,dx)-np.pi*B5[:,sel,0]/180
        xn=np.cos(phi)*mag
        yn=np.sin(phi)*mag
        for i in range(B5.shape[0]):
            ppl.scatter(plt.gca(),xn[i,:],yn[i,:],color=clrs[i],label='_nolegend_')
            valid=np.sqrt(np.power(xn[i,:],2)+np.power(yn[i,:],2))<1.2
            v.extend(valid.tolist())
            if seln[i]: label=str(vpn[i])
            else:label='_nolegend_'
            plt.plot(np.median(xn[i,valid]),np.median(yn[i,valid]),'x',
                          label=label,mec=clrs[i],mew=2,ms=8)
            xn[i,~valid]=np.nan
        plt.gca().set_aspect('equal')
        plt.xlim([-1,1]);plt.ylim([-1,1])
        #if kk>2: plt.gca().set_yticklabels([])
        plt.title(titles[kk-2])
        xn=xn.flatten()[np.array(v)]
        res.append(xn)
        #xn=median(xn,1)
        m=np.median(xn)
        plt.plot([m,m],[-1,1],'--g',color='gray',label='_nolegend_',zorder=-2)
        sse=std(xn,bias=True)/xn.size**0.5
        er= sse* stats.t.ppf(0.95,xn.size)
        er=[sap(xn,25),sap(xn,75)]
        plt.gca().add_patch(plt.Rectangle([er[0],-5],er[1]-er[0],10,
                                color='k',zorder=-2,alpha=0.1))
        print titles[kk-2]+" %.3f CI [%.3f, %.3f]"%(m,er[0],er[1])
        if kk==2:
            plt.legend(bbox_to_anchor=(0,0,1,1),loc=2,mode="expand",ncol=seln.sum()/2+1,
                bbox_transform=plt.gcf().transFigure,frameon=False)
        if exp==2:
            x0=(1-3**0.5/2)*AR
            for i in range(2):
                plt.plot([x0,x0],[-1,1],':',color='gray')
    plt.savefig(FIGPATH+'e%db5%s.png'%(exp,suffix),dpi=400,bbox_inches='tight', pad_inches=0.1)


def plotB3(B3,clrs=None,exp=1):
    #plt.figure(figsize=(6,12))
    if clrs is None: clrs=getColors(B3.shape[0])
    if exp==1: drawAgent=drawCircularAgent
    else: drawAgent=drawDartAgent
    for i in range(B3.shape[0]):
        drawAgent((-2.2,0),bcgclr=False,scale=0.4)
        drawAgent((2,0),bcgclr=False,scale=0.4,ori=-np.pi/2)
        plt.plot([-1.5,1.5],[0,0],'gray'); plt.plot([0,0],[-1.5,1.5],'gray')
        ppl.scatter(plt.gca(),B3[i,X,:],B3[i,Y,:],color=clrs[i])
        plt.plot(np.median(B3[i,X,:]),np.median(B3[i,Y,:]),'x',mec=clrs[i],mew=2,ms=8)
        plt.xlim([-1.5,1.5])
        plt.ylim([-1.5,1.5])
        ax=plt.gca()
        ax.set_clip_on(False)
        ax.set_aspect('equal')
        plt.title('Displacement in Degrees')
    plt.savefig(FIGPATH+'e1b3.png',dpi=400, pad_inches=0.1)
def plotB3fit(fit,suffix='',pars=['mu','mmu']):
    plt.figure(figsize=(6,3))
    D=fit.extract()[pars[0]]
    errorbar(D[:,:,X],x=0.9)
    plt.xlabel('Subject')
    errorbar(D[:,:,Y],x=1.1,clr=(1,0.5,0.5),xaxis=False)
    plt.ylabel('Displacement in Degrees')
    plt.savefig(FIGPATH+'e1b3%s.png'%suffix,dpi=400,bbox_inches='tight', pad_inches=0)

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
    plt.savefig(FIGPATH+'e1b4.png',dpi=400,bbox_inches='tight', pad_inches=0.1)

def plotVectors():
    ax=plt.gca()
    ax.set_aspect('equal')
    ax.set_axis_off()
    plt.plot([-10,10],[0,0],color='gray',zorder=-3)
    plt.plot([0,0],[-10,10],color='gray',zorder=-3)
    plt.xlim([-2,2]);plt.ylim([-2,2])
    #drawCircularAgent((-0.2,0),eyes=False)
    ax.add_patch(plt.Circle((-0.2,0),radius=AR,ec=[0.94]*3,zorder=-3))
    ax.add_patch(plt.Circle((-0.4,0),radius=AR,ec=[0.96]*3,zorder=-3))
    ax.add_patch(plt.Circle((-0.6,0),radius=AR,ec=[0.98]*3,zorder=-3))
    drawCircularAgent((0,0),ori=np.arctan2(1,1.5)-np.pi/2)
    ax.add_patch(plt.Circle((1.5,1),radius=CHR,fc='g',alpha=0.2))
    plt.arrow(0.3,0.2,0.36,0,head_width=0.08,lw=1,color=[0.8]*3,length_includes_head=True)
    plt.arrow(0.66,0.2,0,-0.36,head_width=0.08,lw=1,color=[0.8]*3,length_includes_head=True)
    plt.arrow(0.66,-0.16,0.2,-0.3,head_width=0.08,lw=1,color=[0.8]*3,length_includes_head=True)
    plt.arrow(0,0,0.3,0.2,head_width=0.08,lw=2,color='k',length_includes_head=True)
    plt.arrow(0,0,0.2,-0.3,head_width=0.08,lw=2,color='k',length_includes_head=True)
    plt.arrow(0,0,0.36,0,head_width=0.08,lw=2,color='k',length_includes_head=True)
    plt.arrow(0,0,0,-0.36,head_width=0.08,lw=2,color='k',length_includes_head=True)

    plt.plot([0.86],[-0.46],'+k',ms=12)
    plt.savefig(FIGPATH+'vectoraddition.png',dpi=400,bbox_inches='tight', pad_inches=0.1)
