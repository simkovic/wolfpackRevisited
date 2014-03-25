import numpy as np
import pylab as plt
from scipy import stats
import os,pickle,sys

BASE=os.getcwd().rstrip('code')+os.path.sep
TRAJPATH=BASE+'trajData'+os.path.sep
BEHDPATH=BASE+'behData'+os.path.sep

FILTER=stats.norm.pdf(np.linspace(-3,3,7),0,1)
FILTER/=FILTER.sum()
VTH=15
MD=0.3
MONHZ=75
quadMaps=np.array([[1,1,0,0],[0,0,1,1],[0,1,0,1],[1,0,1,0],[1,0,0,1],[0,1,1,0]])

def computeState(isFix,md,nfm=np.inf):
    fixations=[]
    if isFix.sum()==0: return np.int32(isFix),[]
    fixon = np.bitwise_and(isFix,
        np.bitwise_not(np.roll(isFix,1))).nonzero()[0].tolist()
    fixoff=np.bitwise_and(np.roll(isFix,1),
        np.bitwise_not(isFix)).nonzero()[0].tolist()
    if len(fixon)==0 and len(fixoff)==0: fixon=[0]; fixoff=[isFix.size-1]
    if fixon[-1]>fixoff[-1]:fixoff.append(isFix.shape[0]-1)
    if fixon[0]>fixoff[0]:fixon.insert(0,0)
    if len(fixon)!=len(fixoff): print 'invalid fixonoff';raise TypeError
    for f in range(len(fixon)):
        fs=fixon[f];fe=(fixoff[f]+1);dur=fe-fs
        if  dur<md[0] or dur>md[1]:
            isFix[fs:fe]=False
        else: fixations.append([fs,fe-1])
    return isFix,fixations

def computeForce(loc,points=[],maxmag=np.inf,circlemass=0.):
    r=np.array(loc,ndmin=2)-points
    phi=np.arctan2(r[:,1],r[:,0])
    res=np.array([np.cos(phi),np.sin(phi)])/np.array(np.square(r).sum(axis=1),ndmin=2)
    res=res.sum(1)
    phi=np.arctan2(-loc[1],-loc[0])
    res+=circlemass*np.array([np.cos(phi),np.sin(phi)])/np.square(12-np.linalg.norm(loc))
    if np.linalg.norm(res)<maxmag:return res
    else: 
        phi=np.arctan2(res[1],res[0])
        return np.array([np.cos(phi),np.sin(phi)])*maxmag
def findNearestLocalMin(points,start=[0,0],stepsize=0.05,
                        maxiter=200,circlemass=0.):
    ''' start point, points xys and stepsize in degrees '''
    dif=1
    oldf=computeForce(start,points,circlemass=circlemass)
    inif=np.copy(oldf)
    k=0
    while dif>0:
        phi=np.arctan2(oldf[1],oldf[0])
        start+=np.array([np.cos(phi),np.sin(phi)])*stepsize
        newf=computeForce(start,points,circlemass=circlemass)
        dif=np.linalg.norm(oldf)-np.linalg.norm(newf)
        #print oldf,np.linalg.norm(oldf),newf, np.linalg.norm(newf)
        oldf=newf
        k+=1
        if k>maxiter or np.linalg.norm(start)>11.75: break
    #print 'search finished after %d iterations'%k
    return start,oldf,inif

def plotForceField(mouse,points,delta=0.,cm=0.):
    N=51
    sz=12
    rng=np.linspace(-sz,sz,N)
    R=np.zeros((N,N,2))
    for x in range(rng.size):
        for y in range(rng.size):
            offset=np.array([np.cos(points[:,2]),np.sin(points[:,2])]).T
            res=computeForce([rng[x],rng[y]],points=points[:,:2]+delta*offset,
                             maxmag=0.1,circlemass=cm)
            R[x,y,:]=res#np.linalg.norm(res)
    plt.cla()
    c=plt.Circle((0,0),radius=11.75,fill=False)
    plt.gca().add_patch(c)
    plt.plot(points[:,0],points[:,1],'ks',ms=8)
    plt.plot(mouse[0],mouse[1],'go',ms=8)
    plt.xlim([-12,12])
    plt.ylim([-12,12])
    plt.gca().set_aspect(1)
    #plt.pcolormesh(rng,rng,R.T,vmax=0.1)
    #R=np.square(R)
    plt.quiver(rng,rng,R[:,:,0].T,R[:,:,1].T,np.linalg.norm(R,axis=2).T,scale=3)
    #plt.pcolormesh(rng,rng,np.linalg.norm(R,axis=2).T)
    loc,minforce,ms=findNearestLocalMin(points[:,:2],start=np.copy(mouse[:2]),
                                        circlemass=cm)
    plt.plot(loc[0],loc[1],'bd',ms=8)
    plt.grid(b=False)
    ax=plt.gca()
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.set_xticklabels([]);ax.set_yticklabels([])

def plotFF(vp=351,b=0,t=0,f=675,foffset=0,cm=0.):
    ''' foffset - frame offset correction between point and mouse coords '''
    order=np.load(TRAJPATH+'vp%d/ordervp%db%d.npy'%(vp,vp,b))
    points=np.load(TRAJPATH+'vp%d/vp%dtrial%03d.npy'%(300,300,order[t]))
    mouse=np.load(TRAJPATH+'vp%d/chsVp%db%dtrial%03d.npy'%(vp,vp,b,t))
    print np.linalg.norm(mouse[f,:])
    plotForceField(mouse[f,:],points[f+foffset,:,:],cm=cm)

#h=3;delta=0;k=2
#plotForceField(mouse=D[h][k,-1,:],points=D[h][k,:-1,:],delta=delta)

#fig=plt.figure(figsize=(14,9))
#for f in range(1275):
#    plotForceField(mouse=m[f,:],points=traj[f,:,:],delta=0.)
#    plt.savefig(BEHDPATH+'fig%04d.png'%f,dpi=100)


#fms=range(25)#[0,5,10,15,20,25]
#cms=[0.1]#[0,0.1,0.2,0.3,0.4,0.5]


def measureFit(vpn,fms,cms,dps,eventbased=False,vpcms=None,fshift=0):
    R=np.zeros((len(vpn),len(fms),len(cms),len(dps),51*42,6))*np.nan;b=0
    offset=0
    for h in range(len(vpn)):
        vp=vpn[h]
        print vp
        B= np.loadtxt(BEHDPATH+'vp%03d.res'%vp)
        for fm in range(len(fms)):
            for cm in range(len(cms)):
                for dp in range(len(dps)):
                    g=0
                    for t in range(42):
                        order=np.load(TRAJPATH+'vp%d/ordervp%db%d.npy'%(vp,vp,b))
                        points=np.load(TRAJPATH+'vp%d/vp%dtrial%03d.npy'%(300,300,order[t]))
                        mouse=np.load(TRAJPATH+'vp%d/chsVp%db%dtrial%03d.npy'%(vp,vp,b,t))
                        if eventbased:  
                            vel=np.linalg.norm(np.diff(mouse[:,:2],axis=0),axis=1)*MONHZ
                            vel=np.convolve(vel,FILTER)
                            sel,events=computeState(vel<VTH,md=[MD*MONHZ,np.inf])
                            fs=np.array(events)[1:,0]+fshift
                        else: fs=range(50,mouse.shape[0],25)
                        nowolf=np.int32(quadMaps[int(B[t,4]),np.arange(12)/3])
                        for f in fs:
                            if f-fms[fm]<0 or f-fms[fm]>1274: continue
                            if dp!=0:
                                temp=np.array(mouse[f+fms[fm],:],ndmin=2)-points[f+fms[fm],:,:2]
                                phis=np.arctan2(temp[:,1],temp[:,0])-np.pi*nowolf
                                offset=np.array([np.cos(phis),np.sin(phis)]).T
                            if vpcms is None: c=cms[cm]
                            else: c=vpcms[h]
                            loc,minforce,msforce=findNearestLocalMin(points[f+fms[fm],:,:2]+dps[dp]*offset,
                                    start=np.copy(mouse[f,:2]),circlemass=c)
                            R[h,fm,cm,dp,g,:]=[np.linalg.norm(loc-mouse[f,:2]),
                                            np.linalg.norm(minforce),
                                          np.linalg.norm(msforce),t,f,np.linalg.norm(mouse[f,:])]
                            g+=1
    return R
if __name__=='__main__':
    sel=int(sys.argv[1])
    print sel
    vpn=range(351,381); vpn.remove(369); vpn.remove(370);vpn.remove(379)
    fmss=np.linspace(-30,-10,41)
    cmss=np.linspace(0,1.4,41)
    dpss=np.linspace(-1,1,201)
    fns=['Rfmse','Rcmse','Rfmsn','Rcmsn','Rdpse','Rdpsn','Rdpse']
    if sel==0: R=measureFit(vpn,fms=fmss,cms=[0.8],dps=[0],eventbased=True)
    elif sel==1: R=measureFit(vpn,fms=[-15],cms=cmss,dps=[0],eventbased=True)
    elif sel==2: R=measureFit(vpn,fms=fmss,cms=[0.8],dps=[0],eventbased=False)
    elif sel==3: R=measureFit(vpn,fms=[-15],cms=cmss,dps=[0],eventbased=False)
    elif sel==4: R=measureFit(vpn,fms=[-15],cms=[0.8],dps=dpss,eventbased=True)
    elif sel==5: R=measureFit(vpn,fms=[-15],cms=[0.8],dps=dpss,eventbased=False)
    elif sel>=6:
        R=np.squeeze(np.load('Rcmse.npy'))
        cms=np.load('cms.npy')
        cm=[]
        for vp in range(R.shape[0]):
            cm.append(cms[4:][np.argmin(stats.nanmedian(R[vp,4:,:,0],axis=1))])
        cm=np.array(cm)
        R=measureFit(vpn,fms=[-15],cms=[0.8],dps=dpss,eventbased=True,
                     vpcms=cm,fshift=[0,8,15,23][sel-6])
    np.save(fns[6]+str(sel-5),R)
    if sel<6:
        np.save('dps',dpss)
        np.save('fms',fmss)
        np.save('cms',cmss)
