import pandas as pd
import numpy as np 
import math as m


df=pd.read_csv('P1_data_train.csv',sep=',',header=None)
trdata=df.values
df=pd.read_csv('P1_labels_train.csv',sep=',',header=None)
trlabel=df.values
df=pd.read_csv('P1_labels_test.csv',sep=',',header=None)
tslabel=df.values
df=pd.read_csv('P1_data_test.csv',sep=',',header=None)
tsdata=df.values

rt=len(trdata)
ct=len(trdata[0])

"""--------------------calculate mean and varience-------------------------"""
mu5=0
mu6=0
c5=0
c6=0
cov5=0
cov6=0
x5=[]
x6=[]
for i in range(0,rt):
	if trlabel[i]==5:
		mu5=mu5+trdata[i]
		x5.append(trdata[i])
		c5=c5+1
	else:
		mu6=mu6+trdata[i]
		x6.append(trdata[i])
		c6=c6+1


x5=np.matrix(x5)
x6=np.matrix(x6)

mu5=mu5/c5
mu5=np.matrix(mu5)
cov5=np.matmul(np.transpose(x5-mu5),(x5-mu5))
cov5=cov5/c5;

mu6=mu6/c6
mu6=np.matrix(mu6)
cov6=np.matmul(np.transpose(x6-mu6),(x6-mu6))
cov6=cov6/c6;
"""print("mean 5:-")
print(mu5)
print("\nmean 6:-")
print(mu6)
print('\ncovariance matrix for 5:-')
print(cov5)
print('\ncovarience matrix for 6:-')
print(cov6)"""
"""-----------------------change covarience----------------------"""
#cov6=cov5




"""--------------calculating gaussian----------------------------"""
prc5=c5/rt;
prc6=c6/rt;


rts=len(tsdata)
cts=len(tsdata[0])

cov5inv=np.linalg.inv(cov5)
cov5det=np.linalg.det(cov5)
cov5det=m.sqrt(abs(cov5det))
p5=(m.sqrt(2*m.pi))**cts
d5=p5*cov5det
#print("\n")
cov6inv=np.linalg.inv(cov6)
cov6det=np.linalg.det(cov6)
cov6det=m.sqrt(abs(cov6det))
p6=(m.sqrt(2*m.pi))**cts
d6=p6*cov6det
lrnlabel=[]
for i in range(0,rts):
	#calculate pxgc5
	x=(np.matrix(tsdata[i])-mu5)
	pr=np.matmul(x,np.transpose(cov5inv))
	z=np.matmul(pr,np.transpose(x))
	pxgc5=(np.exp(-z/2))*prc5
	pxgc5=pxgc5/(d5)
	#calculate pxgc6
	x=(np.matrix(tsdata[i])-mu6)
	pr=np.matmul(x,np.transpose(cov6inv))
	z=np.matmul(pr,np.transpose(x))
	pxgc6=(np.exp(-z/2))*prc6
	pxgc6=pxgc6/(d6)
	#calculate pc5gx
	pc5gx=pxgc5/(pxgc5+pxgc6)
	#learning label
	if(pc5gx>0.5):
		lrnlabel.append(5)
	else:
		lrnlabel.append(6)

"""-------------calcuating misclassification--------------------------"""
misc=[[0,0],[0,0]]
for i in range(0,rts):
	if(lrnlabel[i]==5 and tslabel[i]==5):
		misc[0][0]=misc[0][0]+1
	elif(lrnlabel[i]==5 and tslabel[i]==6):
		misc[0][1]=misc[0][1]+1
	elif(lrnlabel[i]==6 and tslabel[i]==5):
		misc[1][0]=misc[1][0]+1
	else:
		misc[1][1]=misc[1][1]+1

ct5=0
ct6=0
for i in range(0,rts):
	if(tslabel[i]==5):
		ct5=ct5+1
	else:
		ct6=ct6+1
print("confusion matrix:-")
print(misc)
misc[0][0]=(misc[0][0]/ct5)*100
misc[0][1]=(misc[0][1]/ct6)*100
misc[1][0]=(misc[1][0]/ct5)*100
misc[1][1]=(misc[1][1]/ct6)*100
print("percentage:-")
print(misc)
	


    
