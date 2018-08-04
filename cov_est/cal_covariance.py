# -*- coding: utf-8 -*-
"""
Created on Thu Jul 12 12:34:12 2018

@author: Administrator


"""


import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import time
import scipy.stats as sps
import pandas as pd
import datetime
import pickle
import csv

import numpy.linalg as lin
import scipy.linalg as lin2
from arch.univariate import arch_model

from sklearn.covariance import LedoitWolf, OAS, ShrunkCovariance, \
    log_likelihood, empirical_covariance

import cvxpy as cp



def fill_return(X):

    n = X.shape[0]
    p = X.shape[1]

    xnav = np.zeros((n,p))
    xnav[:] = np.nan

    for j in range(p):
        ind = np.where(~np.isnan(X[:, j]))[0]
        xnav[ind, j] = np.cumprod(1 + X[ind, j])


    tmp = X[0, :].copy()
    tmp[np.isnan(tmp)] = 0
    XX = pd.DataFrame(xnav).interpolate().fillna(method='bfill').pct_change().values
    XX[0, :] = tmp

    return XX

	
def fill_return2(X):

    n = X.shape[0]
    p = X.shape[1]
    XX = X.copy()
    ss = np.argsort(np.sum(~np.isnan(XX),axis=0))

    for j in range(p):
        ind1 = np.where(np.isnan(XX[:, j]))[0]
        if ind1.size>0:
            ind2 = np.where(~np.isnan(XX[:, j]))[0]
            np.random.seed(ss[j])
            XX[ind1,j] = np.random.randn(len(ind1))*np.std(XX[ind2,j])+np.mean(XX[ind2,j])


    return XX


def pav(data):

    T, N = data.shape
    v = data.copy()
    for j in range(N):
        v[:,j] = pav_1d(v[:,j])


    return v

def pav_1d(y):

    y = np.asarray(y)

    n_samples = len(y)
    v = y
    lvls = np.arange(n_samples)
    lvlsets = np.c_[lvls, lvls]
    flag = 1
    while flag:
        deriv = np.diff(v)
        if np.all(deriv >= 0):
            break

        viol = np.where(deriv < 0)[0]
        start = lvlsets[viol[0], 0]
        last = lvlsets[viol[0] + 1, 1]
        s = 0
        n = last - start + 1
        for i in range(start, last + 1):
            s += v[i]

        val = s / n
        for i in range(start, last + 1):
            v[i] = val
            lvlsets[i, 0] = start
            lvlsets[i, 1] = last

    return v.reshape((len(v),))


def direct_kernel(X):

    n, p = X.shape
    X = X - np.mean(X, axis=0)
    sample = X.T.dot(X) / n
    sample = (sample+sample.T)/2
    [lamb, u] = lin2.eigh(sample)

    sort_idx = np.argsort(lamb)
    lamb = lamb[sort_idx]
    u = u[:,sort_idx]

    lamb = lamb[np.max([1,p-n+2])-1:]


    L=np.tile(lamb.reshape((len(lamb),1)),(1,len(lamb)))
    h = n**(-0.35)

    tmp = 4*(L.T**2)*h*h -(L-L.T)**2
    tmp[tmp<0] = 0
    ftilde =np.mean(np.sqrt(tmp)/(2*np.pi*(L.T**2)*h*h),axis =1)


    tmp2 = (L-L.T)**2-4*(L.T**2)*h*h
    tmp2[tmp2<0] = 0
    Hftilde =np.mean((np.sign(L-L.T)*np.sqrt(tmp2)-L+L.T)/(2*np.pi*(L.T**2)*h*h), axis =1)

    if p<=n-1:
        dtilde = lamb/((np.pi*(p/n)*lamb*ftilde)**2 +(1-(p/n)-np.pi*(p/n)*lamb*Hftilde)**2)
    else:
        Hftilde0 =(1-np.sqrt(1-4*h*h))/(2*np.pi*h*h)*np.mean(1/lamb)
        dtilde0 = 1/(np.pi*(p-n+1)/(n-1)*Hftilde0)
        dtilde1 = lamb/((np.pi**2)*(lamb**2)*(ftilde**2+Hftilde**2))
        dtilde =np.hstack((dtilde0*np.ones(p-n+1), dtilde1))
        dtilde =dtilde.reshape((len(dtilde),1))

    if dtilde.ndim ==1:
        dtilde = dtilde.reshape((len(dtilde),1))

    dhat = pav(dtilde)
    sigmahat = u.dot(np.tile(dhat,(1,p))*(u.T))

    return sigmahat


def LW_est(X):
    
    '''
    Ledoit-Wolf optimal shrinkage coefficient estimate
    X_size = (n_samples, n_features)
    '''
    
    lw = LedoitWolf()
    cov_lw = lw.fit(X).covariance_ 
    
    return cov_lw


def OAS_est(X):

    '''
    OAS coefficient estimate
    X_size = (n_samples, n_features)
    '''
    oa = OAS()
    cov_oa = oa.fit(X).covariance_ 
    
    return cov_oa







def CCC_GARCH(X,K,method=1):

    n, p = X.shape

    # xnav = np.zeros((n,p))
    # xnav[:] = np.nan

    cond_vol = np.zeros((n,p))
    cond_vol[:] = np.nan

    if method == 1:
        pred_vol = np.zeros((K,p))
        pred_vol[:] = np.nan

        for j in range(p):
            ind = np.where(~np.isnan(X[:,j]))[0]
            xx = X[ind,j]
            # xnav[ind,j] = np.cumprod(1+xx)
            md = arch_model(xx - np.mean(xx), mean='Zero', p=1, q=1)
            res = md.fit(disp = 'off',show_warning=False)
            cond_vol[ind, j] = res.conditional_volatility
            pred_vol[:, j] = md.forecast(res.params,horizon = K).variance.iloc[-1,:].values
        cond_vol = pd.DataFrame(cond_vol).interpolate().fillna(method='bfill').values  # interplolate to get missing value



    if method == 2:
        for j in range(p):
            ind = np.where(~np.isnan(X[:, j]))[0]
            xx = X[ind, j]
            # xnav[ind, j] = np.cumprod(1 + xx)
            md = arch_model(xx - np.mean(xx), mean='Zero', p=1, q=1)
            res = md.fit(disp='off', show_warning=False)
            cond_vol[ind, j] = res.conditional_volatility
        cond_vol = pd.DataFrame(cond_vol).interpolate().fillna(method='bfill').values  # interplolate to get missing value


    # tmp = X[0,:].copy()
    # tmp[np.isnan(tmp)] = 0
    # XX = pd.DataFrame(xnav).interpolate().fillna(method = 'bfill').pct_change().values
    # XX[0,:] = tmp

    XX = X - np.nanmean(X,axis = 0)
    Y = fill_return2(XX/cond_vol)

    # Y = pd.DataFrame(Y).interpolate().fillna(method='bfill').values
    # Y[np.isnan(Y)] = 0
    rho = direct_kernel(Y)
    aa = np.diag(1/np.sqrt(np.diag(rho)))
    rho = np.dot(aa.dot(rho),aa)



    if method == 1:
        cond_cov = np.zeros((p, p, K))
        cond_cov[:] = np.nan
        for k in range(K):
            tmp = np.diag(np.sqrt(pred_vol[k, :])).dot(rho).dot(np.diag(np.sqrt(pred_vol[k, :])))
            cond_cov[:, :, k] = (tmp + tmp.T) / 2

        return np.sum(cond_cov, axis=2)




    if method == 2:
        cond_cov2 = np.zeros((p, p, K))
        cond_cov2[:] = np.nan
        for k in range(K):
            tmp = np.diag(cond_vol[-k - 1, :]).dot(rho).dot(np.diag(cond_vol[-k - 1, :]))
            cond_cov2[:, :, k] = (tmp + tmp.T) / 2

        return np.sum(cond_cov2, axis=2)


def portfolio_opt(risk):
    
    P = cp.Variable(risk.shape[0])
    
    constraints = [
        0 <= P, P <= 0.5,
        cp.sum_entries(P) == 1
    ]
    
    objective = cp.Minimize(cp.quad_form(P, risk))
    
    prob = cp.Problem(objective, constraints)
    result = prob.solve()
    
    return P
