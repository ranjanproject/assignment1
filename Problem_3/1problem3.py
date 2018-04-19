import pandas as pd 
import numpy as np 
import math as m 
import matplotlib.pyplot as plt 


df=pd.read_csv('Wage_dataset.csv',sep=',',header=None)
wd=df.values
df=pd.read_csv('Wage_original.csv',sep=',',header=None)
wo=df.values

nr=len(wd)
nc=len(wd[0])

year=wd[:,0]
age=wd[:,1]
ed=wd[:,4]
wage=wd[:,10]
year=np.array(year)
age=np.array(age)
ed=np.array(ed)
wage=np.array(wage)

"""print(year)
print("\n")
print(age)
print("\n")
print(ed)
print("\n")
print(wage)"""
"""print((wd[0]))
print("\n")
print((wo[1]))"""

"""---------------------------------wage vs age linear regression-----------------------------"""
print("wage vs age:-")
print("Enter the degree of polynomial to be fit:")
dim=int(input())
#dim=10
dim=dim+1
yage=[]
a=np.zeros((dim,dim))
#print(a)
for i in range(0,dim):
	yage.append(np.sum(wage*(age**i)))
	for j in range(0,dim):
		a[i][j]=np.sum(age**(i+j))

#print(a)
agvwg_par=np.matmul(np.linalg.inv(np.matrix(a)),np.transpose(np.matrix(yage)))
#print(agvwg_par)

learnwage=[]

for i in range(0,nr):
	sm=0
	for j in range(0,dim):
		sm=sm+agvwg_par[j]*(age[i]**j)
		sm=float(sm)
	learnwage.append(sm)

learnwage=np.array(learnwage)
#print(learnage)
#print(age)
plt.title('Wage vs Age plot')
plt.ylabel('Wage')
plt.xlabel('Age')
plt.plot(age,wage,'bo',age,learnwage,'r^')
plt.show()

"""---------------------------------wage vs year linear regression--------------------------------"""
print("wage vs year:-")
print("Enter the degree of polynomial to be fit:")
dim=int(input())
#dim=10
dim=dim+1
yr=[]
a=np.zeros((dim,dim))

for i in range(0,dim):
	yr.append(np.sum(wage*(year**i)))
	for j in range(0,dim):
		a[i][j]=np.sum(year**(i+j))


yrvwg_par=np.matmul(np.linalg.inv(np.matrix(a)),np.transpose(np.matrix(yr)))

learnwage=[]

for i in range(0,nr):
	sm=0
	for j in range(0,dim):
		sm=sm+yrvwg_par[j]*(year[i]**j)
		sm=float(sm)
	learnwage.append(sm)

learnwage=np.array(learnwage)

plt.title('Wage vs year plot')
plt.ylabel('Wage')
plt.xlabel('year')
plt.plot(year,wage,'bo',year,learnwage,'r^')
plt.show()

"""-------------------------------wage vs education linear regression---------------------------------"""
print("wage vs education:-")
print("Enter the degree of polynomial to be fit:")
dim=int(input())
dim=dim+1
yed=[]
a=np.zeros((dim,dim))

for i in range(0,dim):
	yed.append(np.sum(wage*(ed**i)))
	for j in range(0,dim):
		a[i][j]=np.sum(ed**(i+j))


edvwg_par=np.matmul(np.linalg.inv(np.matrix(a)),np.transpose(np.matrix(yed)))

learnwage=[]

for i in range(0,nr):
	sm=0
	for j in range(0,dim):
		sm=sm+edvwg_par[j]*(ed[i]**j)
		sm=float(sm)
	learnwage.append(sm)

learnwage=np.array(learnwage)

plt.title('Wage vs education plot')
plt.ylabel('Wage')
plt.xlabel('education')
plt.plot(ed,wage,'bo',ed,learnwage,'r^')
plt.show()