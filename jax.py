# -*- coding: utf-8 -*-
"""
Deep Learning library based on tensor
Support feed forward neural network only
This library is layer-based
Created on Sun Jun 19 17:39:32 2016

@author: Jerry Wong
"""

import pandas as pd

import matplotlib.pyplot as plt
from matplotlib import cm

import random as rd
from random import randint

import math
from math import isnan

import pylab

import numpy as np
from numpy import array,mean,eye,ones,log,exp,tanh,dot,tensordot,linspace,squeeze

from numpy.linalg import inv

import time
import os

import statistics as stat

#       np.pad(np.array([a]),(2,3),'constant',constant_values = (0,0))

class jlayer:
#    neuronTypeList = ['sigmoid','tanh','softPlus','vanila']
#    structure = {}
    input = np.array([])
    neuronVal = np.array([])
    DIn = [1]
    DOut = [1]
    bias = [0]
    
    def __init__(self,neuronType,structure):
        self.neuronType = neuronType
        self.DIn = structure
#        self.inputShape = inputShape
#        self.structure = dict.fromkeys(self.neuronTypeList,[0])
        
    def getInput(self,newInput):
        self.input = newInput
        self.m = self.input.shape[0]
        self.DIn = list(self.input.shape[1:])
        
        if self.neuronType == 'sigmoid':
            self.neuronVal = 1/(1+exp(-self.input))
            self.neuronGrad = np.multiply(self.neuronVal,1-self.neuronVal)
        elif self.neuronType =='tanh':
            self.neuronVal = tanh(self.input)
            self.neuronGrad = 1 - np.multiply(self.neuronVal,self.neuronVal)
        elif self.neuronType == 'softPlus':
            self.neuronVal = log(1+exp(self.input))
            self.neuronGrad = 1/(1+exp(-self.input))
        else:
            self.neuronVal = self.input
            self.neuronGrad = np.ones(self.input.shape)
        
    def initialFilter(self):
        self.filter = np.random.random(self.DOut+self.DIn)-0.5
        self.bias = np.zeros(self.DOut)
        
    def updateFilter(self,step):
        self.filter = self.filter-step
        
    def getFilter(self,newFilter):
        self.filter = newFilter
        
    def getBias(self,bias):
        self.bias = bias
        #shape of bias units should be the same as DOut
        
    def updateBias(self,step):
        self.bias = self.bias-step
    
    def output(self):
#        dotInd = list(range(len(self.DIn)))
        if self.neuronType == 'sigmoid':
            self.neuronVal = 1/(1+exp(-self.input))
            self.neuronGrad = np.multiply(self.neuronVal,1-self.neuronVal)
        elif self.neuronType =='tanh':
#            print(self.input)
#            print(tanh(self.input))
            self.neuronVal = tanh(self.input)
#            print(self.neuronVal)
            self.neuronGrad = 1 - np.multiply(self.neuronVal,self.neuronVal)
        elif self.neuronType == 'softPlus':
            self.neuronVal = log(1+exp(self.input))
            self.neuronGrad = 1/(1+exp(-self.input))
        else:
            self.neuronVal = self.input
            self.neuronGrad = np.ones(self.input.shape)
            
        il = len(self.input.shape)
        ol = len(self.filter.shape)
#        print(il)
#        print(ol)
        self.outputTmp = tensordot(self.neuronVal,self.filter,(list(range(il))[1:],list(range(ol))[len(self.DOut):]))
        return self.bias+self.outputTmp
        
    def grad(self):
#        il = len(self.input.shape)
#        ol = len(self.filter.shape)
#        print(il)
#        print(ol)
        self.neuronGradTmp = self.neuronGrad
        
        self.filterTmp = self.filter
#        print(self.filter.shape)
        self.neuronGradTmp.shape = [self.m]+[1]*len(self.DOut)+self.DIn
        self.filterTmp.shape = [1]+self.DOut+self.DIn
        self.outputTmp = np.multiply(self.neuronGradTmp,self.filterTmp)
        self.filterTmp.shape = self.DOut+self.DIn
#        print(self.filter.shape)
        return self.outputTmp
        
    def biasGrad(self):
        return np.ones([self.m]+list(self.bias.shape))
        
        

class jnet:
    X = array([None])
    Y = array([None])
    dW = array([None])
    def __init__(self,X,Y):
        self.backX = X
        self.backY = Y
        self.X = X
        self.Y = Y
        self.m = self.backX.shape[0]
        
    def getLayer(self,newLayerList):
        self.layerList = newLayerList
        self.nLayer = len(self.layerList)
        
        
    def initialFilter(self):
        self.layerList[0].DOut = list(self.Y.shape[1:])
        self.layerList[-1].DIn = list(self.X.shape[1:])
        for i in list(range(self.nLayer))[1:]:
            self.layerList[i].DOut = self.layerList[i-1].DIn
            
        for item in self.layerList:
            item.initialFilter()
        
        self.layerList[0].filter = np.ones(self.layerList[0].filter.shape)
        self.layerList[0].bias = np.zeros(self.layerList[0].bias.shape)
        
        
    def output(self):
        self.yhat = np.zeros(self.Y.shape)
        currentInput = self.X
        for i in list(range(self.nLayer)):
            self.layerList[-i-1].getInput(currentInput)
            currentInput = self.layerList[-i-1].output()
        
        self.yhat = currentInput
        self.error = self.Y-self.yhat
        self.trainRSquare = 1-stat.pvariance(np.squeeze(self.error))/stat.pvariance(np.squeeze(self.Y))
        return currentInput
    
    def train(self,itemax):
        self.lossRec = [None]
        self.testLossRec = [None]
        self.batchSize = self.m
        for k in list(range(itemax)):
            
            if k%np.floor(itemax/10) == 0:
                print(k*100/itemax,'%')
#            self.forecast()
#            self.testLossRec.append(np.linalg.norm(self.forecastError)/self.mt)
#            ind = rd.sample(range(0,self.backX.shape[0]-1),self.batchSize);
#            (0,self.backX.shape[0]-1,self.batchSize)
#            self.X = self.backX[ind,:]
#            self.Y = self.backY[ind,:]
            
            self.output()
            self.lossRec.append(np.linalg.norm(self.error)/self.batchSize)
            self.jacobian = None
            self.mu = 0.0001#/self.batchSize*(100/(100+k))
            self.dW = [None]*self.nLayer
#            self.dB = [None]*self.nLayer
            for i in list(range(self.nLayer))[0:-1]:
                
                rs = len(self.layerList[0].DOut)
                r0 = len(self.layerList[i-1].DOut)
                r1 = len(self.layerList[i].DOut)
                r2 = len(self.layerList[i].DIn)
                
                strs = bytes(list(range(106,106+rs))).decode('utf-8')
                str1 = bytes(list(range(106,106+r0))).decode('utf-8')
                str2 = bytes(list(range(106+r0,106+r0+r1))).decode('utf-8')
                str3 = bytes(list(range(106+r0+r1,106+r0+r1+r2))).decode('utf-8')
#                print(strs)
                parameter1 = 'i'+strs+str2+','+'i'+str2+str3+'->'+'i'+strs+str3
                parameter2 = 'i'+strs+','+'i'+strs+str3+'->'+'i'+str3
                parameter3 = 'i'+str3+','+'i...'+'->'+str3+'...'
                parameter4 = 'i'+str3+','+'i'+str3+'->'+str3
                if i == 0:
                    self.jacobian = self.layerList[0].grad()
                else:
#                    print(parameter1)
                    self.jacobian = np.einsum(parameter1,self.jacobian,self.layerList[i].grad())
                
                self.weightedJac = np.einsum(parameter2,-self.error,self.jacobian)
                
                self.contractedJac = np.einsum(parameter3,self.weightedJac,self.layerList[i+1].neuronVal)
#                print(parameter4)
#                print(self.layerList[i+1].biasGrad().shape)
#                print(self.weightedJac.shape)
                self.contractedBiasGrad = np.einsum(parameter4,self.weightedJac,self.layerList[i+1].biasGrad())
#                print(self.contractedBiasGrad.shape)
          
#                print(self.layerList[i].neuronVal.shape)
#                print(self.contractedJac.shape)
                self.dW[i+1]=(self.contractedJac)
#                self.dB[i+1]=(self.contractedBiasGrad)
    
#                print(self.dB[i+1].shape)
#                print(self.layerList[i+1].bias)                
                
                self.layerList[i+1].updateFilter(self.dW[i+1]*self.mu)
#                self.layerList[i+1].updateBias(self.dB[i+1]*self.mu*0)
            
        self.fig = plt.figure()
        plt.plot(self.lossRec,'b--')
#        plt.plot(self.testLossRec,'r--')
        self.X = self.backX
        self.Y = self.backY
                
                
    def setTestSet(self,Xt,Yt):
        self.Xt = Xt
        self.Yt = Yt
        self.mt = self.Xt.shape[0]
    
    def forecast(self):
        self.forecastResult = np.zeros(self.Yt.shape)
        currentInput = self.Xt
        for i in list(range(self.nLayer)):
            self.layerList[-i-1].getInput(currentInput)
            currentInput = self.layerList[-i-1].output()
    
        self.forecastResult = currentInput
        self.forecastError = self.Yt-self.forecastResult
        self.accuracy = sum(np.sign(self.forecastResult) == np.sign(self.Yt))/self.mt
        self.RSquare = 1-stat.pvariance(np.squeeze(self.forecastError))/stat.pvariance(np.squeeze(self.Yt))
#        print(self.accuracy)
        
    def saveFilter(self):
        self.filter = [None]*self.nLayer
        self.bias = [None]*self.nLayer
        for i in list(range(self.nLayer)):
#            print(self.layerList[i].filter)
            self.filter[i] = self.layerList[i].filter
            self.bias[i] = self.layerList[i].bias
        
        return([self.filter,self.bias])
            
    def loadFilter(self,newFilter):
        for i in list(range(self.nLayer)):
            self.layerList[i].getFilter(newFilter[0][i])
            self.layerList[i].getBias(newFilter[1][i])
        
        



print('JAX loaded')