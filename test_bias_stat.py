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


data = pd.read_csv('factor_r_sqrt_000905.csv')
data = data.iloc[:,2:]
data = data.dropna()

length = 252
n_forward = 21
tau = 90

Bias=[]
R_stat=[]

for i in range(length,data.shape[0]-n_forward,1):
    if i%100==0: print(i)
    data_cov, U, F_NW, R_i, Std_i = NW_adjusted(data,tau=tau,length=length,n_start=i,n_forward=n_forward,NW=1)
    R_stat.append(np.mean(R_i))
    Bias.append(R_i/Std_i)
    

Bias = np.array(Bias)
Bias_eigen = [np.std(Bias[:,x]) for x in range(Bias.shape[1])]

plt.plot(Bias_eigen,'-*')
