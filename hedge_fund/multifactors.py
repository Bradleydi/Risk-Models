from WindPy import w
from scipy import stats
import numpy as np
import pandas as pd
import statsmodels.api as sm

w.start()

from preprocessing_factors import Z_score, preprocessing, factors_combine, sys_orth




def cal_IC(data,dates,factor_names):
    
    IC_all = []
    
    for i in range(len(dates)):
        
        data_frame = data[data['trade_date']==dates[i]].copy()
        
        IC_i = []
        for name in factor_names:
            IC_i.append( stats.spearmanr(data_frame[name], data_frame['ret'])[0] )
        
        IC_all.append(IC_i)
        
    IC_all = pd.DataFrame(IC_all)
    IC_all.index = dates
    
    return IC_all


if __name__ == '__main__':

    w.start()
    
    FLAG_PREPROCESS = 0
    
    names1 = ['lncap', 'ep', 'bp', 'sp', 'm10', 'm60', 'np_gr', 'oper_gr','opprofit_gr', 'turnover20', 'turnover60', 'west_roe_yoy',
              'west_netprofit_CAGR', 'west_sales_CAGR', 'roe_yoy', 'np_yoy','sales_yoy', 'op_yoy', 'roe', 'roa', 'std20', 'std60']
    
    factors = [ ['lncap'],['ep','bp','sp'],['m10','m60'],['np_gr','oper_gr','opprofit_gr'],['turnover20','turnover60'],
                ['west_roe_yoy','west_netprofit_CAGR','west_sales_CAGR'],['roe_yoy','np_yoy','sales_yoy','op_yoy'],['roe','roa'],['std20','std60']]
    
    factor_names = ['lncap','evalu','mtm','growth','turnover','west','value_yoy','value','std']
    
    if FLAG_PREPROCESS:
    
        data = pd.read_csv('data_month_zz500.csv',encoding='ANSI')
        dates = data['trade_date'].drop_duplicates().tolist()
        N_stocks = data.shape[0]/len(dates)
        
        data_new = []
        
        for i in range(0,len(dates)):
            
            print(dates[i])
            
            data_frame = data.iloc[i*N_stocks:(i+1)*N_stocks,:].copy()
            data_frame = preprocessing(data_frame,names1)
            data_frame = factors_combine(data_frame,factors)
            data_frame = sys_orth(data_frame,names1)
            
            data_new.append(data_frame)
        
        
        data_new = pd.concat(data_new)
        data_new.to_csv('factor_month_zz500_2.csv')
    
    
    data = pd.read_csv('factor_month_zz500_2.csv',encoding='ANSI', index_col=0)
    dates = data['trade_date'].drop_duplicates().tolist()
    
    names = factor_names 
    
    IC_all = cal_IC(data,dates,names)
    
    T_p = 12
    N_top = 100
    
    # ICIR 加权
    #ICIR = IC_all.rolling(T_p).mean()/IC_all.rolling(T_p).std()
    
    # IC衰减加权
    ICIR = IC_all.ewm(T_p).mean()
    
    r=[1.]
    corr_=[]
    
    for i in range(T_p+1,len(dates)):
        
        print(dates[i])
        
        data_frame = data[data['trade_date']==dates[i]].copy()
        factors = data_frame[names[:]].as_matrix()
        
        scores = factors @ ICIR.iloc[i-1,].values
        
        data_frame['scores'] = scores
        data_frame = data_frame.sort_values(by=['scores'],ascending=0)
        data_r_top = data_frame[data_frame['trade_s']!='-1']
        r_ = data_r_top['ret'].iloc[:N_top].mean()
        corr_.append(stats.spearmanr(data_frame['ret'],data_frame['scores'])[0])
        
        r.append(1+r_)
    
    r = np.cumprod(r)
    
    raw_hs300 = w.wsd("000905.SH", "close", dates[T_p+1], '2018-07-27', "Period=M")
    hs300 = pd.DataFrame(raw_hs300.Data[0])
    hs300.index = raw_hs300.Times
    hs300[0] = hs300[0]/hs300[0][0]
    
    hs300[1] = r
    data_plot = hs300
    data_plot[2] = data_plot[1]/data_plot[0]
    
    plt.plot(data_plot[0])
    plt.plot(data_plot[1])
    plt.plot(data_plot[2],'r--')
    plt.grid()
    plt.show()
    
    plt.bar(np.arange(len(corr_)),corr_)
    plt.show()
