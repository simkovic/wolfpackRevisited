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
# some constants .
WINDOWS=1
LINUX=0
MIN=0
MAX=1
X=0;Y=1;PHI=2;T=2
CHASEE=0; CHASER=1; DISTRACTOR=2

def pointInTriangle(t1,t2,t3,pt):
    def sign(p1,p2,p3):
        return (p1[X]-p3[X])*(p2[Y]-p3[Y])-(p2[X]-p3[X])*(p1[Y]-p3[Y])
    b1=sign(pt,t1,t2)<0
    b2=sign(pt,t2,t3)<0
    b3=sign(pt,t3,t1)<0
    return b1==b2 and b2==b3

def drawCircle(M,pos,radius,value=1):
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if np.sqrt((i-pos[X])**2+(j-pos[Y])**2)<radius:
                M[i,j]=value
    return M   
##########################################################
# create masks that define various agent shapes 
N=128 # each mask is a NxN rectangle
# circle
CIRCLE=np.ones((N,N))*-1
for i in range(N):
    for j in range(N):
        if np.sqrt((i-N/2+0.5)**2+(j-N/2+0.5)**2)<N/2:
            CIRCLE[i,j]= 1
            
RING=np.ones((N,N))*-1
for i in range(N):
    for j in range(N):
        if np.sqrt((i-N/2+0.5)**2+(j-N/2+0.5)**2)>2*N/5 and np.sqrt((i-N/2+0.5)**2+(j-N/2+0.5)**2)<N/2:
            RING[i,j]= 1
            
DART=np.ones((N,N))*-1
a=np.cos(np.pi/3.0)*N/2.0
b=np.sin(np.pi/3.0)*N/2.0
c=N/2-0.5
t1=(0,c);t2=(c+a,c-b); t3=(c+a,c+b)
for i in range(N):
    for j in range(N):
        DART[i,j]+=2*(pointInTriangle(t1,t2,(c,c),(i,j)) or pointInTriangle(t1,t3,(c,c),(i,j)))
        
del a,b,c,t1,t2,t3
DART=np.rot90(DART,-1)


