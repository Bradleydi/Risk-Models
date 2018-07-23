# -*- coding: utf-8 -*-
"""
Created on Wed Jul 11 17:07:46 2018

@author: Administrator
"""


#from WindPy import *
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

#w.start()

from cal_covariance import fill_return,fill_return2,LW_est, OAS_est, portfolio_opt, direct_kernel, CCC_GARCH


'''
data_frame: T(frames) by N(stocks)
dates_: dates by week / month / day
constraints of optimization can be modified in cal_covariance.portfolio_opt
'''


P = [{}]*5


for i in range(len(dates_)):
    
    data_frame2 = fill_return2(data_frame.values)
    data_frame2 = pd.DataFrame(data_frame2,columns=data_frame.columns)
    
    empirical_cov = np.cov(data_frame2.T)
    OAS_cov = OAS_est(data_frame2)
    LW_cov = LW_est(data_frame2)
    NLS_cov = direct_kernel(data_frame2)
    GAR_cov = CCC_GARCH(data_frame2.values,20)
    
    P_emp = portfolio_opt(empirical_cov)
    P_lw = portfolio_opt(LW_cov)
    P_oas = portfolio_opt(OAS_cov)
    P_nls = portfolio_opt(NLS_cov)
    P_gar = portfolio_opt(GAR_cov)
    
    
    temp = pd.DataFrame(P_emp.value,index=data_frame2.columns)
    temp = temp[temp[0]>0.001]
    P[0][date_i] = temp
    
    temp = pd.DataFrame(P_lw.value,index=data_frame2.columns)
    temp = temp[temp[0]>0.001]
    P[1][date_i] = temp
      
    temp = pd.DataFrame(P_oas.value,index=data_frame2.columns)
    temp = temp[temp[0]>0.001]
    P[2][date_i] = temp
    
    temp = pd.DataFrame(P_nls.value,index=data_frame2.columns)
    temp = temp[temp[0]>0.001]
    P[3][date_i] = temp
    
    temp = pd.DataFrame(P_gar.value,index=data_frame2.columns)
    temp = temp[temp[0]>0.001]
    P[4][date_i] = temp
    
