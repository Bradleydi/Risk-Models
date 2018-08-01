# -*- coding: utf-8 -*-
"""
Created on Fri Jun 22 13:36:43 2018

@author: Administrator
"""

import math
import numpy as np
import time
import pandas as pd
from datetime import datetime
import pickle
import csv

pd.options.mode.chained_assignment = None
from pylab import * 
mpl.rcParams['font.sans-serif']=['SimHei']
mpl.rcParams['axes.unicode_minus']=False




def Z_score(data_):
    
    for c in range(data_.shape[0]):
        if type(data_.iloc[c])==str and data_.iloc[c]=='CWSSService: corrupted response.':
            data_.iloc[c] = np.nan
    
    data_ = pd.DataFrame([float(x) for x in data_])
    median_ = data_.dropna().median()[0]
    MAD = np.abs(data_.dropna() - median_).median()[0]
    max_1 = np.max(data_.values)
    min_1 = np.min(data_.values)
    max_2 = median_+3*1.83*MAD
    min_2 = median_-3*1.83*MAD
    ratio_1 = np.max([(max_1-max_2),0])/(0.5*1.483*MAD)
    ratio_2 = np.max([(min_2-min_1),0])/(0.5*1.483*MAD)
    
    for i in range(data_.shape[0]):
        if data_.iloc[i][0] > max_2 and ratio_1>0:
            data_.iloc[i] = max_2 + (data_.iloc[i]-max_2)/ratio_1
        elif data_.iloc[i][0] < min_2 and ratio_2>0:
            data_.iloc[i] = min_2 - (min_2-data_.iloc[i])/ratio_2
    
    mean_ = data_.dropna().mean()
    std_ = data_.dropna().std()
    
    data_ = (data_-mean_)/std_
    data_ = data_.fillna(0)
    
    return data_

def preprocessing(data_frame,names):
    
    '''Z_socre'''
    for name in names:
        data_frame[name] = np.array(Z_score(data_frame[name])[0].tolist())
    
    '''Cap & Ind'''
    data_frame = data_frame.dropna()
    ind_ = data_frame['ind'].tolist()
    a = np.zeros([data_frame.shape[0],29])
    
    for i in range(data_frame.shape[0]):
        a[i,int(ind_[i]-1)]=1

    b = data_frame['lncap'].values
    X = np.concatenate((np.array([b]).T, a), axis=1)
    y = data_frame[names[1:]].as_matrix()
    
    results = sm.OLS(y, X).fit()
    #beta = np.array(results.params)
    data_frame[names[1:]] = results.resid
    
    for name in names[1:]:
        data_frame[name] = np.array(Z_score(data_frame[name])[0].tolist())
    
    
    return data_frame


def factors_combine(data_frame,factors):
    
    for i in range(len(factors)):
        
        if len(factors[i])==1:
            continue
        else:
            data_frame[factor_names[i]] = data_frame[factors[i]].mean(axis=1)
    
    names_ = ['Unnamed: 0']+factor_names+['ind','trade_s','ret','trade_date']
    
    return data_frame[names_]



def sys_orth(data_frame,names):
    
    F = data_frame[names].as_matrix()
    M = F.T@F
    D,U = np.linalg.eigh(M)
    S = U@np.diag(1./np.sqrt(D))@U.T
    F_t = F@S
    data_frame[names] = F_t
    
    return data_frame















