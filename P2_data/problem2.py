import pandas as pd
import math as m 
import matplotlib
import numpy as np


df=pd.read_csv('P2_train.csv',sep=',',header=None)
trdata=df.values
df=pd.read_csv('P2_test.csv',sep=',',header=None)
tsdata=df.values

trlabel=trdata[:,2]
trdata=np.delete(trdata,2,1)
#print(trdata)
tslabel=tsdata[:,2]
tsdata=np.delete(tsdata,2,1)
#print(tsdata)

nr=len(trdata)
nc=len(trdata[0])

"""--------------------calculate mean and varience-------------------------"""
x0=[]
c0=0
x1=[]
c1=0
mu0=0
mu1=0
for i in range(0,nr):
	if(trlabel[i]==1):
		x1.append(trdata[i])
		mu1=mu1+trdata[i]
		c1=c1+1
	else:
		x0.append(trdata[i])
		mu0=mu0+trdata[i]
		c0=c0+1

x0=np.matrix(x0)
x1=np.matrix(x1)
mu0=mu0/c0;
mu1=mu1/c1;
mu0=np.matrix(mu0)
mu1=np.matrix(mu1)

cov0=np.matmul(np.transpose(x0-mu0),(x0-mu0))
cov0=cov0/c0
#print(cov0) 
cov1=np.matmul(np.transpose(x1-mu1),(x1-mu1))
cov1=cov1/c1
#print(cov1)

"""-----------------------change covarience----------------------"""

"""cov1=0.1*np.eye(2)
#cov1[0][0]=5;cov1[0][1]=4;cov1[1][0]=1.5;cov1[1][1]=2;
#cov0=np.eye(2)
#cov1[1][1]=10

cov0=cov1
#print(cov0)
#print(cov1)"""
"""cov1[0][0]=5;cov1[0][1]=4.5;cov1[1][0]=7;cov1[1][1]=8;
cov0[0][0]=1;cov0[0][1]=0.5;cov0[1][0]=3;cov0[1][1]=4;

cov0=np.transpose(cov0)
cov1=np.transpose(cov1)
print(cov0)
print(cov1)"""
"""--------------calculating gaussian----------------------------"""
prc0=c0/nr
prc1=c1/nr

cov0inv=np.linalg.inv(cov0)
#print(np.matmul(cov0inv,cov0))
cov1inv=np.linalg.inv(cov1)
#print(cov1inv)
cov0det=np.linalg.det(cov0)
cov0det=m.sqrt(abs(cov0det))
#print(cov0det)
cov1det=np.linalg.det(cov1)
cov1det=m.sqrt(abs(cov1det))
#print(cov1det)
p=(m.sqrt(2*m.pi))**nc

nrt=len(tsdata)
nct=len(tsdata[0])

lrnlabel=[]
pgc0=[]
pgc1=[]
pb=[]
for i in range(0,nrt):
	#calculate pxgc0
	x=(np.matrix(tsdata[i])-mu0)
	pr=np.matmul(x,np.transpose(cov0inv))
	z=np.matmul(pr,np.transpose(x))
	pxgc0=(np.exp(-z/2))*prc0
	pxgc0=pxgc0/(p*cov0det)
	pgc0.append(float(pxgc0))
	#calculate pxgc1
	x=(np.matrix(tsdata[i])-mu1)
	pr=np.matmul(x,np.transpose(cov1inv))
	z=np.matmul(pr,np.transpose(x))
	pxgc1=(np.exp(-z/2))*prc1
	pxgc1=pxgc1/(p*cov1det)
	pgc1.append(float(pxgc1))
	#calculate pc0gx
	pc0gx=pxgc0/(pxgc0+pxgc1)
	#learning label
	if(pc0gx>0.5):
		lrnlabel.append(0)
	else:
		lrnlabel.append(1)

"""-------------calcuating misclassification--------------------------"""
misc=[[0,0],[0,0]]
for i in range(0,nrt):
	if(lrnlabel[i]==0 and tslabel[i]==0):
		misc[0][0]=misc[0][0]+1
	elif(lrnlabel[i]==0 and tslabel[i]==1):
		misc[0][1]=misc[0][1]+1
	elif(lrnlabel[i]==1 and tslabel[i]==0):
		misc[1][0]=misc[1][0]+1
	else:
		misc[1][1]=misc[1][1]+1

ct0=0
ct1=0
for i in range(0,nrt):
	if(tslabel[i]==0):
		ct0=ct0+1
	else:
		ct1=ct1+1
print("confusion matrix:-")
print(misc)
misc[0][0]=(misc[0][0]/ct0)*100
misc[0][1]=(misc[0][1]/ct1)*100
misc[1][0]=(misc[1][0]/ct0)*100
misc[1][1]=(misc[1][1]/ct1)*100
print("percentage:-")
print(misc)

"""-------------------------Ploting Contour---------------------------"""
#ploting contour
import matplotlib.cm as cm
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

matplotlib.rcParams['xtick.direction'] = 'out'
matplotlib.rcParams['ytick.direction'] = 'out'


cov0=np.array(cov0)
cov1=np.array(cov1)

mu0=np.array(mu0)
mu1=np.array(mu1)
"""--------function for gaussian calculation------------"""
def bivariate_normal(position, mu, cov):
    n = mu.shape[0]
    cov_det = np.linalg.det(cov)
    cov_inv = np.linalg.inv(cov)
    d = np.sqrt((2*np.pi)**n * cov_det)
    fac = np.einsum('...k,kl,...l->...', position-mu, cov_inv, position-mu)
    return np.exp(-fac / 2) / d

"""-----------function end----------------------------"""
sd=np.linspace(min(tsdata[:,0]),max(tsdata[:,0]),1000)
sd1=np.linspace(min(tsdata[:,1]),max(tsdata[:,1]),1000)
X, Y = np.meshgrid(sd,sd1)
dis = np.empty(X.shape + (2,))
dis[:, :, 0] = X
dis[:, :, 1] = Y


Z0 = bivariate_normal(dis,mu0,cov0)
plt.contour(X,Y,Z0)

Z1 = bivariate_normal(dis,mu1,cov1)
plt.contour(X,Y,Z1)

# difference of Gaussians
Z = (Z1 - Z0)
plt.contour(X,Y,Z)

plt.title('Iso-probability Contour')
plt.scatter(tsdata[:,0],tsdata[:,1])
plt.ylabel('x2')
plt.xlabel('x1')
plt.show()
