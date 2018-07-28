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


from risk_adj import Covariance_NW, NW_adjusted, Eigen_adjusted

# read factors data
data = pd.read_csv('factor_r_sqrt_000905.csv')
data = data.iloc[:,2:]
data = data.dropna()

# parameters
length = 252
n_forward = 21
tau = 90

Bias=[]
stat_all=[]

# Calulate bias statistics
for i in range(length,data.shape[0]-n_forward,1):
    
    if i%20==0: print("i: ",i)
    
    data_cov, U, F_NW, R_i, Std_i = NW_adjusted(data,tau=tau,length=length,n_start=i,n_forward=n_forward,NW=1)
    
    Bias.append(R_i/Std_i)
    
    # eigen adjusted N_mc is # of MC simulations
    stat = Eigen_adjusted(F_NW,U,Std_i,length=252,N_mc=1000)
    stat_all.append(stat)
    
# Bias statistics of eigenfactors using the unadjusted covariance matrix 
# Figure 4.1 of UNE4
Bias = np.array(Bias)
Bias_eigen = [np.std(Bias[:,x]) for x in range(Bias.shape[1])]
plt.plot(Bias_eigen,'-*')

# Simulated volatility bias 
# Figure 4.3 of UNE4, plot every simulation result
for i in range(len(stat_all)):
    plt.plot(stat_all[i])

# Mean simulated volatility bias 
# Figure 4.3 of UNE4
gamma_k = np.array(stat_all).mean(axis=0)
plt.plot(gamma_k)



# Fitting the simulated volatility bias v(k) using a parabola, then scale the fit values in proportion to their deviation from 1
# Page 41 of UNE4 

def Gamma_fitting(gamma_k,para=2,n_start_fitting = 16):
    # para: the scaling parameter "a"
    # n_start_fitting: assign zero weight to the first "n_start_fitting" eigenfactor, here we choose 16 while UNE4 chooses 15
    y = gamma_k[n_start_fitting:]
    x = np.array(range(n_start_fitting,40))
    z = np.polyfit(x, y, 2)
    p = np.poly1d(z)
    gamma_k_new = [p(xi) for xi in range(n_start_fitting)]+list(y)
    
    gamma_k_new = para*(np.array(gamma_k_new)-1)+1
    
    return gamma_k_new

# New v(k)
gamma_k_new = Gamma_fitting(gamma_k,para=2,n_start_fitting=16)
plt.plot(gamma_k_new,'-*')



# Eigen-adjused step
# Here we use gamma_k_new to adjust F_NW
# The gamma_k_new should be calucalted with in-sample data, but it costs long CPU time. 
# Thus, we use gamma_k_new from previous step, with future information, for simplicity.


b2=[]   # bias

for i in range(length,data.shape[0]-n_forward,1):
    
    data_cov, U, F_NW, R_i, Std_i = NW_adjusted(data,tau=tau,length=length,n_start=i,n_forward=n_forward,NW=0)
    
    s, U = linalg.eigh(F_NW)
    
    F_eigen = U @ (np.diag(np.power(gamma_k_new,2))@np.diag(s)) @ U.T

    s2, U2 = linalg.eigh(F_eigen)
    
    R_eigen2 = R_i #np.dot(U2.T,R_i)
    b2.append(R_eigen2/np.sqrt(s2))

# Bias statistics of eigenfactors using the eigen-adjusted covariance matrix
# Figure 4.4 of UNE4
B2=np.array(b2).std(axis=0)
plt.plot(B2,'-*')



# Volatility Regime Adjustment Step

# parameters
tau = 42
lambd = 0.5**(1./tau)
w = np.array([lambd**n for n in range(252)][::-1])

# BF is the factor cross-sectional bias statistic, formula (4.3) of UNE4
# lambda_F is the factor volatility multiplier, formula (4.4) of UNE4
# CSV is the factor cross-sectional volatility, formula (4.6) of UNE4

BF_t_all=[]
lambda_F=[]
lambda_F_all=[]
BF_t_vra_all=[]
CSV = []

for i in range(length,data.shape[0]-n_forward,1):
    
    data_cov, U, F_NW, R_i, Std_i = NW_adjusted(data,tau=tau,length=length,n_start=i,n_forward=n_forward,NW=0)
    
    s, U = linalg.eigh(F_NW)
    
    F_eigen = U@(np.diag(np.power(gamma_k_new,2))@np.diag(s))@U.T
    
    s2, U2 = linalg.eigh(F_eigen)
    
    f_kt = data.iloc[i,:].values
    b_ = f_kt / np.sqrt(np.diag(F_eigen))
    BF_t = np.sqrt(np.mean(b_@b_))
    BF_t_all.append(BF_t)
    
    CSV.append (np.sqrt(np.mean(f_kt@f_kt)) )
    
    
    if i>length+252:
        lambda_F = np.sqrt( np.power(BF_t_all[i-length-252:i-length],2)@w / w.sum() )
        lambda_F_all.append( lambda_F )
    
        F_VRA = np.diag([lambda_F**2]*40)@F_eigen
        b_vra = f_kt / np.sqrt(np.diag(F_VRA))
        BF_t = np.sqrt(np.mean(b_vra@b_vra))
        BF_t_vra_all.append(BF_t)


# plot CSV, Figure 4.6 of UNE4
plt.plot(pd.DataFrame(CSV).rolling(30).mean())

# plot factor volatility multiplier, Figure 4.6 of UNE4
plt.plot(lambda_F_all)

BF_befor_after = pd.DataFrame(BF_t_all).rolling(120).mean()
BF_befor_after[1] = [np.nan]*(len(BF_t_all)-len(BF_t_vra_all))+BF_t_vra_all
BF_befor_after = BF_befor_after.dropna()
BF_befor_after[1] = BF_befor_after[1].rolling(120).mean()

# Figure 4.7 of UNE4
plt.plot(BF_befor_after)




