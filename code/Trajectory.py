## The MIT License (MIT)
##
## Copyright (c) <2015> <Matus Simkovic>
##
## Permission is hereby granted, free of charge, to any person obtaining a copy
## of this software and associated documentation files (the "Software"), to deal
## in the Software without restriction, including without limitation the rights
## to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
## copies of the Software, and to permit persons to whom the Software is
## furnished to do so, subject to the following conditions:
##
## The above copyright notice and this permission notice shall be included in
## all copies or substantial portions of the Software.
##
## THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
## IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
## FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
## AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
## LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
## OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
## THE SOFTWARE.

from psychopy.event import xydist
import numpy as np
import random, os, pickle
# NOTE: most of the motion settings are setup in Settings.py, not here
#   in particular check that the monitor frame rate parameter is set up
#   correctly in Settings.py, the actual monitor frame rate doesnt matter
#   for the trajectory generation
from Settings import Q 
from Constants import *
from Maze import *

######################################
# trajectory generation routines
        
class RandomAgent():
    '''generates the trajectory for a random agent '''
    def __init__(self,nrframes,dispSize,pos,pdc,sd,moveRange):
        ''' nrframes - number of frames (consecutive positions) to generate
            dispSize - size of the square movement area
            pos - initial position
            pdc - probability of a direction change
            sd - agent speed
            moveRange - size of the range from which new motion
                direction is selected after a direction change
        '''
        self.offset=pos
        self.ds=dispSize
        self.nrframes=nrframes
        self.traj=np.zeros((nrframes,3))
        self.reset()
        # some vars used for loging stats for the Diagnosis class
        self.pdc=pdc
        self.sd=sd
        self.nrcrashes=np.zeros((nrframes))
        self.ndc=np.zeros((self.nrframes))
        self.moveRange=moveRange/2.0
    
    def reset(self):
        """ reset the agent to initial position
            discards any previously generated trajectory
        """
        self.ndc=np.zeros((self.nrframes))
        self.nrcrashes=np.zeros((self.nrframes))
        self.f=0
        self.i=0
        self.traj[self.f,:]=np.array((random.random()*self.ds[X]-self.ds[X]/2.0+self.offset[X],
            random.random()*self.ds[Y]-self.ds[Y]/2.0+self.offset[Y],random.random()*360))
    def backtrack(self):
        ''' should the algorithm backtrack to previous position?
            returns boolean
        '''
        self.f-=51#31
        return self.f<0 or self.i>100000#10000
    def getPosition(self,dec=0):
        ''' return current/latest position'''
        return self.traj[self.f+dec,[X,Y]]
    def getTrajectory(self):
        return self.traj
    def move(self):
        ''' generate next position'''
        self.f+=1
        self.i+=1
        f=self.f
        self.nrcrashes[f]=0
        rnd = random.random()<self.pdc
        self.ndc[f]=float(rnd)
        if rnd: # change direction chasee
            self.traj[f,PHI]=(self.traj[f-1,PHI]
                    +random.uniform(-self.moveRange,self.moveRange))%360
        else:
            self.traj[f,PHI]= self.traj[f-1,PHI]
        adjust =np.array((np.cos(self.traj[f,PHI]/180.0*np.pi) 
                *self.sd,np.sin(self.traj[f,PHI]/180.0*np.pi)*self.sd))
        self.traj[f,[X,Y]]=self.traj[f-1,[X,Y]]+adjust
        return (f+1)==self.nrframes
    def crashed(self,newD):
        ''' adjust direction and position after a contact with the boundary
            newD - new direction
        '''
        self.nrcrashes[self.f]=1
        self.traj[self.f,PHI]=newD[1]
        self.traj[self.f,[X,Y]]=newD[0]

class HeatSeekingChaser(RandomAgent):
    '''generates the trajectory for a heat-seeking chaser '''
    def __init__(self,*args,**kwargs):
        ''' the last argument (isGreedy) determines how boundary
            colisions are handle
            if True chaser makes random direction change after collision
            otherwise it moves towards chasee
        '''
        isGreedy=args[-1]        
        RandomAgent.__init__(self,*args[:-1],**kwargs)
        self.isGreedy=isGreedy
    def move(self,targetPos,crash=False):
        ''' generate next position
            targetPos - chasee's position
            crash - chaser made contact with boundary
        '''
        if not crash:
            self.f+=1
            self.i+=1
        f=self.f
        rnd = random.random()<self.pdc
        self.ndc[f]=int(rnd)
        if rnd or crash:
            self.traj[f,PHI]=np.arctan2(targetPos[Y],
                            targetPos[X])/np.pi * 180
            self.traj[f,PHI]=(360+self.traj[f,PHI]
                +random.uniform(-self.moveRange,self.moveRange))%360
        else:
            self.traj[f,PHI]= self.traj[f-1,PHI]   
        adjust =np.array((np.cos(self.traj[f,PHI]/180.0*np.pi) 
                *self.sd,np.sin(self.traj[f,PHI]/180.0*np.pi)*self.sd))
        self.traj[f,[X,Y]]=self.traj[f-1,[X,Y]]+adjust
    def crashed(self,newD=None,targetPos=(0,0)):
        ''' adjust direction and position after a contact with the boundary
            targetPos - chasee's position
            newD - new direction
        '''
        #print 'crashed', self.f
        if not self.isGreedy:
            RandomAgent.crashed(self,newD)
        else:
            #print 'before', self.getPosition(), targetPos
            self.move(targetPos,crash=True)
            self.nrcrashes[self.f]=1
            #print 'after', self.getPosition()

            
def generateTrial(nragents,maze,rejectionDistance=0.0,STATISTICS=False):
    ''' generates the agent trajectories for a single trial
        the generation may take considerably longer when rejectionDistance>0
        nragents - number of agents to generate
        maze - Maze class instance (see Maze.py)
        rejectionDistance - chaser-chasee minimum distance in degrees
        STATISTICS - if True will log stats for the Diagnosis class
        
        returns ndarray of size (nrframes x nragents x 3)
            first dim - number of frames is derived based on trial duration
                and frame rate (both in Settings.py)
            second dim - the first agents is chasee,
                         the second agents is chaser,
                         the rest are distractors
            third dim - X position in degrees, Y position in degrees,
                direction in radians  
        if STATISTICS is True returns statistics
        
    '''
    if STATISTICS: nrbacktracks=0
    # init chaser chasee
    chasee=RandomAgent(Q.nrframes,maze.dispSize,maze.pos,
            Q.pDirChange[CHASEE],Q.aSpeed,Q.phiRange[0])
    chaser=HeatSeekingChaser(Q.nrframes,maze.dispSize,maze.pos,
            Q.pDirChange[CHASER],Q.aSpeed,Q.phiRange[CHASER],True)
    while (xydist(chaser.getPosition(),chasee.getPosition())<Q.initDistCC[MIN]
        and xydist(chaser.getPosition(),chasee.getPosition())>Q.initDistCC[MAX]):
        # resample until valid distance between chaser and chasee is obtained
        chasee.reset(); chaser.reset()
    agents=[chasee,chaser]
    # init distractors
    for d in range(nragents-2):
        distractor=RandomAgent(Q.nrframes,maze.dispSize,maze.pos,
            Q.pDirChange[DISTRACTOR],Q.aSpeed,Q.phiRange[CHASEE])
        agents.append(distractor)
    # check for wall collisions
    for a in range(nragents):
        d,edge=maze.shortestDistanceFromWall(agents[a].getPosition())
        while d<=Q.agentRadius:
            agents[a].reset()
            d,edge=maze.shortestDistanceFromWall(agents[a].getPosition())
    # generate the movement of chasee and chaser
    finished=False
    while not finished:
        # check the distance
        (dx,dy)=chasee.getPosition() - chaser.getPosition()
        if np.sqrt(dx**2+dy**2)<rejectionDistance:
            if STATISTICS: nrbacktracks+=1
            deadend=chaser.backtrack()
            chasee.backtrack()
            if deadend: # reset the algorithm 
                print 'dead end', chasee.f
                if STATISTICS: return None, None, None, None,None
                else: return None
            (dx,dy)=chasee.getPosition() - chaser.getPosition()
        # move chaser and avoid walls
        chaser.move((dx,dy))
        d,edge=maze.shortestDistanceFromWall(chaser.getPosition())
        if d<=Q.agentRadius:
            newD=maze.bounceOff(chaser.getPosition(),
                chaser.getPosition(-1),edge,Q.agentRadius)
            chaser.crashed(newD=newD,targetPos=(dx,dy))
        # move chasee and avoid walls
        finished=chasee.move()
        d,edge=maze.shortestDistanceFromWall(chasee.getPosition())
        if d<=Q.agentRadius:
            newD=maze.bounceOff(chasee.getPosition(),
                chasee.getPosition(-1),edge,Q.agentRadius)
            chasee.crashed(newD)
        #if chaser.f>401:
        #    raise NameError('stop')
    # generate distractor movement
    finished=False
    while not finished and nragents>2:
        for a in range(2,nragents):
            finished=agents[a].move()
            d,edge=maze.shortestDistanceFromWall(agents[a].getPosition())
            if d<=Q.agentRadius:
                newD=maze.bounceOff(agents[a].getPosition(),
                    agents[a].getPosition(-1),edge,Q.agentRadius)
                agents[a].crashed(newD)
    trajectories=np.zeros((Q.nrframes,nragents,3))
    for a in range(nragents):
        tt=agents[a].getTrajectory()
        trajectories[:,a,X]=tt[:,X]
        trajectories[:,a,Y]=tt[:,Y]
        trajectories[:,a,PHI]=tt[:,PHI]
    if STATISTICS:
        #statistics=np.zeros((nrframes,nragents,3))
        statistics=[trajectories,np.zeros((Q.nrframes,3)),
                    np.zeros((3)),nrbacktracks,[chasee.ndc.sum(),chaser.ndc.sum(),agents[2].ndc.sum()]]
        for a in range(3):
            statistics[1][:,a]=agents[a].getTrajectory()[:,PHI]
            statistics[2][a]=agents[a].nrcrashes.sum()
        return statistics
    else: return trajectories

def generateExperiment(vpn):
    ''' 
        vpn - tuple of ints, each value gives the subject id
    '''
    offs=5.875; sz=(2*offs+Q.agentSize,2*offs+Q.agentSize)
    quadrants=[EmptyMaze((1,1),dispSize=sz,pos=(offs,offs),lw2cwRatio=0),
        EmptyMaze((1,1),dispSize=sz,pos=(-offs,offs),lw2cwRatio=0),
        EmptyMaze((1,1),dispSize=sz,pos=(offs,-offs),lw2cwRatio=0),
        EmptyMaze((1,1),dispSize=sz,pos=(-offs,-offs),lw2cwRatio=0)]
    nrtrials=42; 
    os.chdir('..');os.chdir('input/')
    for vp in vpn:
        vpname='vp%03d' % vp;os.mkdir(vpname);os.chdir(vpname)
        for trial in range(nrtrials):
            if vp>300 and vp<400 and vp!=350: continue
            trajectories=[]
            for k in range(len(quadrants)):
                traj=generateTrial(5,maze=quadrants[k], rejectionDistance=0.0)
                trajectories.append(traj[:,2:,:])
            fn='%strial%03d'% (vpname,trial); 
            np.save(fn,np.concatenate(trajectories,axis=1))
        np.save('order%sb%d'% (vpname,0),np.random.permutation(nrtrials))
        np.save('order%sb%d'% (vpname,1),np.random.permutation(nrtrials))
        np.save('order%sb%d'% (vpname,2),np.random.permutation(nrtrials))

        Q.save('SettingsTraj.pkl')
        os.chdir('..')

##################################################       
    
if __name__ == '__main__':
    generateExperiment(range(300,400))   
