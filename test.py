# -*- coding: utf-8 -*-
"""
Created on Sun Jun 19 21:18:25 2016

@author: Jerry Wong
"""

import importlib
import jax
import numpy as np
import pandas as pd
import timeSeries
import matplotlib.pyplot as plt

importlib.reload(jax)
importlib.reload(timeSeries)

#X = np.array([[[1,-1],[-1,1]],[[-1,1],[1,-1]]])
#Y = np.array([[-1],[1]])
#f = np.array([[[0.5,0],[0,0.5]]])
#bias = np.array([1])
#
#l = jax.jlayer('tanh',[1])
#l.getInput(X)
#l.getFilter(f)
#l.getBias(bias)
#l.DOut = [1]

dataSet = np.array(pd.read_csv(r'F:\mywd\macLea\Python\autoDiff\tra.csv', sep=',',header = None),
dtype=np.float64)
dataSet[:,64][dataSet[:,64]!=5]=-1
dataSet[:,64][dataSet[:,64]==5]=1
trainX = (dataSet[:,:-1]-32)/8
trainY = dataSet[:,64]
trainX.shape = [3823,8,8]
trainY.shape = [3823,1]
#3823
#
testDataSet = np.array(pd.read_csv(r'F:\mywd\macLea\Python\autoDiff\tes.csv', sep=',',header = None),
dtype=np.float64)
testDataSet[:,64][testDataSet[:,64]!=5]=-1
testDataSet[:,64][testDataSet[:,64]==5]=1
testX = (testDataSet[:,:-1]-32)/8
testY = testDataSet[:,64]
testX.shape = [1797,8,8]
testY.shape = [1797,1]
#1797

#dataSet = np.array(pd.read_csv(r'F:\mywd\GSA\data\data.csv',sep = ',',header = 0),dtype = np.float64)[:,1:4]
#dataSet[:,2] = dataSet[:,2]/0.875
#
#X = timeSeries.feature(dataSet[:,0:-1],1410,list(range(120)))
#Y = timeSeries.lable(dataSet[:,[-1]],1410,list(range(120)))

#nsample = 300000
#ind = np.random.random_integers(0,X.shape[0]-1,nsample)
#ind1 = ind[0:210000]
#ind2 = ind[210000:nsample ]
#
#trainX = X[ind1,:]
#trainY = Y[ind1,:]
#testX = X[ind2,:]
#testY = Y[ind2,:]


#trainX = np.array([[1,-1],[-1,1]])
#trainY = np.array([[-1],[1]])
#testX = trainX
#testY = trainY




n = jax.jnet(trainX,trainY)
n.setTestSet(testX,testY)
n.getLayer([jax.jlayer('tanh',[1]),jax.jlayer('vanila',[1])])#jax.jlayer('tanh',[3,3]),jax.jlayer('softPlus',[4,4]),
n.initialFilter()
randomFilter = n.saveFilter()
#n.forecast()
n.train(2500)
filter1 = n.saveFilter()
#n.loadFilter(randomFilter)
#n.forecast()
#print(100*n.accuracy)
#n.loadFilter(filter1)
print(100*n.trainRSquare)
n.forecast()
print(100*n.accuracy)
print(100*n.RSquare)







#filter10000 lossRec10000 testLossRec10000






print('-----------------------------------------------')