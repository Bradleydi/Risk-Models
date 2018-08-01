from WindPy import w
from scipy import stats
import numpy as np
import pandas as pd
import statsmodels.api as sm

w.start()


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


names1 = ['lncap', 'ep', 'bp', 'sp', 'm10', 'm60', 'np_gr', 'oper_gr',
       'opprofit_gr', 'turnover20', 'turnover60', 'west_roe_yoy',
       'west_netprofit_CAGR', 'west_sales_CAGR', 'roe_yoy', 'np_yoy',
       'sales_yoy', 'op_yoy', 'roe', 'roa', 'std20', 'std60']

factors = [ ['lncap'],['ep','bp','sp'],['m10','m60'],['np_gr','oper_gr','opprofit_gr'],['turnover20','turnover60'],
            ['west_roe_yoy','west_netprofit_CAGR','west_sales_CAGR'],['roe_yoy','np_yoy','sales_yoy','op_yoy'],['roe','roa'],['std20','std60']]

factor_names = ['lncap','evalu','mtm','growth','turnover','west','value_yoy','value','std']


data = pd.read_csv('factor_month_zz500_2.csv',encoding='ANSI', index_col=0)
dates = data['trade_date'].drop_duplicates().tolist()

###
names = factor_names # names1

IC_all = []

for i in range(len(dates)):
    
    print(dates[i])
    
    data_frame = data[data['trade_date']==dates[i]].copy()
    
    IC_i = []
    for name in names:
        IC_i.append( stats.spearmanr(data_frame[name], data_frame['ret'])[0] )
        
    IC_all.append(IC_i)
    
IC_all = pd.DataFrame(IC_all)
IC_all.index = dates


T_p = 12


#ICIR = IC_all.rolling(T_p).mean()/IC_all.rolling(T_p).std()

ICIR = IC_all.ewm(T_p).mean()

r=[1.]
corr_=[]

for i in range(T_p+1,len(dates)):
    
    print(dates[i])
    
    data_frame = data[data['trade_date']==dates[i]].copy()
    factors = data_frame[names[:]].as_matrix()
    
    ICIR_i = ICIR.iloc[i-1,:]
    weight = factors@ICIR_i.values
    
    data_r = data_frame[['ret','trade_s']]
    data_r['w'] = list(weight)
    data_r = data_r.sort_values(by=['w'],ascending=0)
    data_r_top = data_r.iloc[:100,:]
    data_r_top = data_r_top[data_r_top['trade_s']!='-1']
    r_ = data_r_top['ret'].mean()
    
    corr_.append(stats.spearmanr(data_r.iloc[:,0],data_r.iloc[:,2])[0])
    
    r.append(r[-1]*(1+r_))
    

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

plt.bar(np.arange(len(corr_)),corr_)
