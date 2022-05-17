
import numpy as np
import random
import pandas as pd
import math 
from sklearn import datasets
from sklearn import preprocessing
import scipy.io as sio
from svdd import SVDD

# load data

def data():

        nindex = np.loadtxt('../Matrix_data/Nindex.txt',dtype=int)
        index = np.loadtxt('../Matrix_data/Index.txt',dtype=int)
        Sim = np.loadtxt('../MDAfeature/Simfeatures.txt')
        PPMI = np.loadtxt('../MDAfeature/PPMIfeatures.txt')
        Pfeature=[]
        for i in range(0,index.shape[0]):
            ind1=np.concatenate((Sim[index[i,0],:],PPMI[index[i,0],:]),axis=0)
            ind2=np.concatenate((Sim[index[i,1],:],PPMI[index[i,1],:]),axis=0)
            pfeature=np.hstack((ind1,ind2))
            Pfeature.append(pfeature)

        Nfeature=[]
        for j in range(0,nindex.shape[0]):
            nind1=np.concatenate((Sim[nindex[j,0],:],PPMI[nindex[j,0],:]),axis=0)
            nind2=np.concatenate((Sim[nindex[j,1],:],PPMI[nindex[j,1],:]),axis=0)
            nfeature=np.hstack((nind1,nind2))
            Nfeature.append(nfeature)
        
        
        Pfeature = np.array(Pfeature)
        Nfeature = np.array(Nfeature)
        trainData = Pfeature
        testData = Nfeature
        trainLabel = np.ones((len(Pfeature),1))
        testLabel = np.ones((len(Nfeature),1))*-1
    
        return trainData, testData,trainLabel, testLabel
    

# PrepareData data
trainData, testData,trainLabel, testLabel= data()


# set SVDD parameters
parameters = {"positive penalty": 0.9,
              "negative penalty": [],
              "kernel": {"type": 'gauss', "width":1/120},
              "option": {"display": 'on'}}




# construct an SVDD model
svdd = SVDD(parameters)
# train SVDD model
svdd.train(trainData, trainLabel)
# test SVDD model

distance, accuracy = svdd.test(testData, testLabel)
radius=svdd.model["radius"]
indis1=np.where(distance>=radius)
indis2=np.where(distance>=1.01*radius)
indis3=np.where(distance>=1.02*radius)
indis4=np.where(distance>=1.03*radius)
indis5=np.where(distance>=1.035*radius)
dis1=indis1[1]
dis2=indis2[1]
dis3=indis3[1]
dis4=indis4[1]
dis5=indis5[1]
nfeature1=testData[dis1]
nfeature2=testData[dis2]
nfeature3=testData[dis3]
nfeature4=testData[dis4]
nfeature5=testData[dis5]


pind=len(trainData)
nind1=pind
ind1=random.sample(range(0,len(nfeature5)),nind1)
ind2=random.sample(range(0,len(nfeature5)),nind1)
ind3=random.sample(range(0,len(nfeature5)),nind1)
ind4=random.sample(range(0,len(nfeature5)),nind1)
ind5=random.sample(range(0,len(nfeature5)),nind1)
ind6=random.sample(range(0,len(nfeature5)),nind1)
neg1=nfeature5[ind1]
neg2=nfeature5[ind2]
neg3=nfeature5[ind3]
neg4=nfeature5[ind4]
neg5=nfeature5[ind5]
neg6=nfeature5[ind6]
pfeature1= np.array(trainData)
pfeature2= np.array(trainData)
pfeature3= np.array(trainData)
pfeature4= np.array(trainData)
pfeature5= np.array(trainData)
pfeature6= np.array(trainData)
plabel=np.ones((pind, 1))
nlabel=np.zeros((nind1, 1))
np.savetxt('../best_select/Pfeature1.txt',pfeature1)
np.savetxt('../best_select/Pfeature2.txt',pfeature2)
np.savetxt('../best_select/Pfeature3.txt',pfeature3)
np.savetxt('../best_select/Pfeature4.txt',pfeature4)
np.savetxt('../best_select/Pfeature5.txt',pfeature5)
np.savetxt('../best_select/Pfeature6.txt',pfeature6)
np.savetxt('../best_select/Nfeature1.txt',neg1)
np.savetxt('../best_select/Nfeature2.txt',neg2)
np.savetxt('../best_select/Nfeature3.txt',neg3)
np.savetxt('../best_select/Nfeature4.txt',neg4)
np.savetxt('../best_select/Nfeature5.txt',neg5)
np.savetxt('../best_select/Nfeature6.txt',neg6)
np.savetxt('../best_select/Plabel1.txt',plabel)
np.savetxt('../best_select/Plabel2.txt',plabel)
np.savetxt('../best_select/Plabel3.txt',plabel)
np.savetxt('../best_select/Plabel4.txt',plabel)
np.savetxt('../best_select/Plabel5.txt',plabel)
np.savetxt('../best_select/Plabel6.txt',plabel)
np.savetxt('../best_select/Nlabel1.txt',nlabel)
np.savetxt('../best_select/Nlabel2.txt',nlabel)
np.savetxt('../best_select/Nlabel3.txt',nlabel)
np.savetxt('../best_select/Nlabel4.txt',nlabel)
np.savetxt('../best_select/Nlabel5.txt',nlabel)
np.savetxt('../best_select/Nlabel6.txt',nlabel)