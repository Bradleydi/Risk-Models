import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import sklearn
import time
import scipy.stats as sps
import pandas as pd
import datetime
import pickle
import csv
import cvxpy as cp

from scipy import linalg
from risk_adj import Covariance_NW, NW_adjusted, Eigen_adjusted

def EWMA(n=252,tau=42,norm=1):
    
    lambd = 0.5**(1./tau)
    
    w = np.array([lambd**n for n in range(252)][::-1])
    
    if norm:
        return w/w.sum()
    else:
        return w


def Get_data_spr():
    data_spr = pd.read_csv('factors_000905_2010_2018_spr.csv')
    data_spr = data_spr.sort_values(['TRADEDATE'])
    dates = data_spr['TRADEDATE'].drop_duplicates().tolist()
    data_spr_ = data_spr[['STOCKCODE','TRADEDATE','specific_r']]
    
    stocks_all = list(set(data_spr_['STOCKCODE'].tolist()))
    spr_data = pd.DataFrame(np.arange(len(stocks_all)))
    spr_data.index = stocks_all
    
    for i in range(len(dates)):
        
        data_frame = data_spr_[data_spr_['TRADEDATE']==dates[i]][['STOCKCODE','specific_r']]
        data_frame.index = data_frame.STOCKCODE
       
        del data_frame['STOCKCODE']
        
        spr_data = pd.concat([spr_data,data_frame],axis=1)
    
    spr_data = spr_data.sort_index()
    del spr_data[0]
    
    spr_data.columns = np.arange(spr_data.shape[1])
    
    return spr_data

def Get_data_r():
    data_spr = pd.read_csv('factors_000905_2010_2018_spr.csv')
    data_spr = data_spr.sort_values(['TRADEDATE'])
    dates = data_spr['TRADEDATE'].drop_duplicates().tolist()
    data_spr_ = data_spr[['STOCKCODE','TRADEDATE','RET']]
    
    stocks_all = list(set(data_spr_['STOCKCODE'].tolist()))
    spr_data = pd.DataFrame(np.arange(len(stocks_all)))
    spr_data.index = stocks_all
    
    for i in range(len(dates)):
        
        data_frame = data_spr_[data_spr_['TRADEDATE']==dates[i]][['STOCKCODE','RET']]
        data_frame.index = data_frame.STOCKCODE
       
        del data_frame['STOCKCODE']
        
        spr_data = pd.concat([spr_data,data_frame],axis=1)
    
    spr_data = spr_data.sort_index()
    del spr_data[0]
    
    spr_data.columns = np.arange(spr_data.shape[1])
    
    return spr_data









data_factor = pd.read_csv('factor_r_sqrt_000905.csv')
data_factor = data_factor.iloc[:,2:]
data_factor = data_factor.dropna()

length = 252
n_forward = 21
tau = 90

Bias=[]
stat_all=[]

for i in range(length,data_factor.shape[0]-n_forward,10):
    
    if i%20==0: print("i: ",i)
    
    data_cov, U, F_NW, R_i, Std_i = NW_adjusted(data_factor,tau=tau,length=length,n_start=i,NW=0)
    
    Bias.append(R_i/Std_i)
    
    stat = Eigen_adjusted(F_NW,U,Std_i,length=length,N_mc=1000)
    stat_all.append(stat)

    


Bias = np.array(Bias)
Bias_eigen = [np.std(Bias[:,x]) for x in range(Bias.shape[1])]
gamma_k = np.array(stat_all).mean(axis=0)

gamma_k_new = Gamma_fitting(gamma_k,amp=2.,n_start_fitting=15)






spr_data = Get_data_spr()



weights = EWMA(n=252,tau=42,norm=1)

length = 252
n_forward = 21
tau = 90
n_dates = data_factor.shape[0]

BF_t_all=[]
CSV = []
lambda_F=[]
lambda_F_all=[]
BF_t_vra_all=[]

portfolio={}

for i in range(length,n_dates-n_forward,1):
    
    
    date_i = dates[i]
    
    data_cov, U, F_NW, R_i, Std_i = NW_adjusted(data_factor,tau=tau,n_start=i,NW=1)
    
    s, U = linalg.eigh(F_NW)
    
    # eigen adjusted
    F_eigen = U@(np.diag(np.power(gamma_k_new,2))@np.diag(s))@U.T
    
    f_kt = data_factor.iloc[i,:].values
    b_ = f_kt / np.sqrt(np.diag(F_eigen))
    BF_t = np.sqrt(np.mean(b_@b_))
    BF_t_all.append(BF_t)
        
    
    if i>length+252 and (i-length-252)%10==0:
        lambda_F = np.sqrt( np.power(BF_t_all[i-length-252:i-length],2) @ weights )
        lambda_F_all.append( lambda_F )
    
        F_VRA = np.diag([lambda_F**2]*40)@F_eigen
        b_vra = f_kt / np.sqrt(np.diag(F_VRA))
        BF_t = np.sqrt(np.mean(b_vra@b_vra))
        BF_t_vra_all.append(BF_t)
    
        X = data_spr[data_spr['TRADEDATE']==date_i]
        X = X.sort_values(by=['STOCKCODE'])
        stocks_i =  X['STOCKCODE'].tolist()
        X = X.iloc[:,4:-2]
        X = np.concatenate([np.ones((X.shape[0],1)), X ],axis=1)
        
        
        Risk_1 = X @ F_VRA @ X.T
        
        data_temp = spr_data.iloc[:,i-length:i]
        data_temp = data_temp.loc[stocks_i]
        data_temp = data_temp.fillna(0)
        specific_risk = np.diag(data_temp.var(axis=1).values)
        
        Risk = Risk_1 + specific_risk

        P = cp.Variable(Risk.shape[0])
        objective = cp.Minimize(cp.quad_form(P, Risk))
        '''
        constraints = [
            0 <= P, P <= 0.05,
            cp.sum_entries(P) == 1
        ]
        '''
        constraints = [
            0 <= P, P <= 0.1,
            cp.sum_entries(P) == 1
        ]
        
        prob = cp.Problem(objective, constraints)
        result = prob.solve()
        
        port_i = pd.DataFrame(P.value,index=stocks_i)
        port_i = port_i[port_i[0]>0.001]
        
        portfolio[str(date_i)] = port_i
        
        print(date_i,port_i.shape[0])






