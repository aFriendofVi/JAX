# -*- coding: utf-8 -*-
"""
Time Series module
Created on Wed Jun 29 18:40:23 2016

@author: Jerry Wong
"""
import pandas as pd
import numpy as np

def feature(data,k,lag):
    if type(data) != np.ndarray:
        data = np.array(data)
        
    s = int( np.floor(data.shape[0]/k))
#    print(s)
    
    n = [len(lag),data.shape[1]]
    
    m = k-max(lag)-1
#    print(m)
#    print(type(m))
#    print(n)
#    print(type(n))
#    print([m*s]+n)
    result = np.zeros([m*s]+n)

    for day in list(range(s)):
#        print(list(range(s)))
#        print('day ',day)
        for i in list(range(m)):
#            print('====================')
#            print(day*k+max(lag)+i-np.array(lag)+1)
#            print(data[day*k+max(lag)+i-np.array(lag)+1,0:-1])
#            print(data[day*k+max(lag)+i-np.array(lag),0:-1])
            result[day*m+i,:] = data[day*k+max(lag)+i-np.array(lag)+1,:]-data[day*k+max(lag)+i-np.array(lag),:]
           
    
#    np.random.shuffle(result)
    return result
    
def lable(data,k,lag):
    
    if type(data) != np.ndarray:
        data = np.array(data)
        
    s = int( np.floor(data.shape[0]/k))
#    print(s)
    
    n = list(data.shape)[1:]
    
    m = k-max(lag)-1
#    print(m)
#    print(type(m))
#    print(n)
#    print(type(n))
#    print([m*s]+n)
    result = np.zeros([m*s]+n)

    for day in list(range(s)):
#        print(list(range(s)))
#        print('day ',day)
        for i in list(range(m)):
#            print('====================')
#            print(day*k+max(lag)+i-np.array(lag)+1)
#            print(data[day*k+max(lag)+i-np.array(lag)+1,0:-1])
#            print(data[day*k+max(lag)+i-np.array(lag),0:-1])
            result[day*m+i,:] = data[day*k+max(lag)+i+1,:]
        
    return result
    
#aX = generate(dataSet[:,0:2],1410,list(range(61)))