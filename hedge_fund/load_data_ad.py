# -*- coding: utf-8 -*-
"""
Created on Wed Jul 25 10:19:28 2018

@author: Administrator
"""

from WindPy import w
import numpy as np
import pandas as pd


def get_LNCAP(stocks, date):
    date_i = date.strftime("%Y%m%d")
    rawdata = w.wss(stocks, "ev", "unit=1;tradeDate=" + date_i)
    data = pd.DataFrame(rawdata.Data).T
    data.index = stocks

    data = np.log(data)
    data.columns = ['lncap']

    return data

def get_valuation_MTM(stocks,date):
    
    date_i = date.strftime("%Y%m%d")
    rawdata = w.wss(stocks, "pe_ttm,pb_lf,ps_ttm,tech_revs10,tech_revs60,fa_nppcgr_ttm,fa_orgr_ttm,fa_oigr_ttm,tech_turnoverrate20,tech_turnoverrate60,west_avgroe_YOY,west_netprofit_CAGR,west_sales_CAGR","tradeDate="+date_i)
    data = pd.DataFrame(rawdata.Data).T
    data.index = stocks
    data.columns = ['ep','bp','sp','m10','m60','np_gr','oper_gr','opprofit_gr','turnover20','turnover60','west_roe_yoy','west_netprofit_CAGR','west_sales_CAGR']
    data.iloc[:,:3]= 1./data.iloc[:,:3]
    return data


def get_roe_rpt(stocks, date):
    date_i = date.strftime("%Y%m%d")

    rawdata = w.wss(stocks, "latelyrd_bt", "tradeDate=" + date_i)
    lately_rpdate = pd.DataFrame(rawdata.Data).T
    lately_rpdate.index = stocks

    dates_rp = list(set(lately_rpdate[0].tolist()))

    if len(dates_rp) > 1:

        for i in range(len(dates_rp)):

            date_rp_i = dates_rp[i]

            stocks_i = lately_rpdate[lately_rpdate[0] == date_rp_i].index.tolist()
            rawdata1 = w.wss(stocks_i, "yoyroe,qfa_yoynetprofit,qfa_yoysales,qfa_yoyop,qfa_roe_deducted,qfa_roa",
                             "rptDate=" + date_rp_i.strftime("%Y%m%d"))
            data1 = pd.DataFrame(rawdata1.Data).T
            data1.index = stocks_i

            if i == 0:
                data = data1
            else:
                data = pd.concat([data, data1])

    else:
        date_rp_1 = dates_rp[0]

        rawdata1 = w.wss(stocks, "yoyroe,qfa_yoynetprofit,qfa_yoysales,qfa_yoyop,qfa_roe_deducted,qfa_roa",
                         "rptDate=" + date_rp_1.strftime("%Y%m%d"))
        data1 = pd.DataFrame(rawdata1.Data).T
        data1.index = stocks

        data = data1

    data.columns = ['roe_yoy', 'np_yoy', 'sales_yoy', 'op_yoy', 'roe', 'roa']

    return data


def get_stdevr(stocks, date):
    date_i = date.strftime("%Y%m%d")
    date_20 = w.tdaysoffset(-20, date, "").Data[0][0].strftime("%Y%m%d")
    date_60 = w.tdaysoffset(-60, date, "").Data[0][0].strftime("%Y%m%d")

    rawdata1 = w.wss(stocks, "stdevr", "startDate=" + date_20 + ";endDate=" + date_i + ";period=1;returnType=1")
    data1 = pd.DataFrame(rawdata1.Data).T
    rawdata2 = w.wss(stocks, "stdevr", "startDate=" + date_60 + ";endDate=" + date_i + ";period=1;returnType=1")
    data2 = pd.DataFrame(rawdata2.Data).T

    data = pd.concat([data1, data2], axis=1)
    data.index = stocks
    data.columns = ['std20', 'std60']

    return data

def get_close(stocks, date):

    date_i = date.strftime("%Y%m%d")

    rawdata = w.wss(stocks, "close", "unit=1;tradeDate=" + date_i)
    data = pd.DataFrame(rawdata.Data).T
    data.index = stocks
    data.columns = ['close']

    return data

def get_ind(stocks, date):
    
    date_i = date.strftime("%Y%m%d")
    rawdata = w.wss(stocks, "indexcode_citic", "tradeDate=" + date_i + ";industryType=1")
    data = pd.DataFrame(rawdata.Data).T
    
    ind_=[]
    for x in data[0].tolist():
        if x is not None:
            ind_.append(int(x[6:-3]))
        else:
            ind_.append(np.nan)
    
    data[0] = ind_

    data.index = stocks
    data.columns = ['ind']

    return data



def get_ret(stocks,date1,date2):
    
    rawdata1 = w.wsd(stocks, "close", date1, date1, "PriceAdj=F")
    rawdata2 = w.wsd(stocks, "close", date2, date2, "PriceAdj=F")
    data1 = pd.DataFrame(rawdata1.Data).T
    data2 = pd.DataFrame(rawdata2.Data).T
    data = data2/data1-1
    data.index = rawdata1.Codes
    data.columns=['ret']
    
    return data


def get_trade_status(stocks,date):
    
    rawdata = w.wsd(stocks, "trade_status", date, date, "Period=D;PriceAdj=F")
    data = pd.DataFrame(rawdata.Data).T
    data.index = rawdata.Codes
    data = data.replace('停牌一天', -1)
    data.columns = ['trade_s']
    
    return data
    
    
def get_data(dates,code):
    
    data_all=[]
    
    for i in range(len(dates)-1):
        
        date = dates[i]
        date2 = dates[i+1]
        
        print(i,dates[i])
        
        stocks = w.wset("sectorconstituent","date="+date.strftime("%Y-%m-%d")+";windcode="+code+";field=wind_code").Data[0]
        
        data = [get_LNCAP(stocks, date), get_valuation_MTM(stocks, date),
                get_roe_rpt(stocks, date), get_stdevr(stocks, date), get_ind(stocks, date), get_trade_status(stocks,date), get_ret(stocks,date,date2)]
        
        data = pd.concat(data, axis=1, join='outer')  # merge six dataframes on a day
        data['trade_date'] = date.strftime("%Y-%m-%d")
    
        data_all.append(data)
    
    data_all = pd.concat(data_all)
    
    return data_all



if __name__ == '__main__':

    w.start()
    
    dates_week = w.tdays("2010-01-01", "2018-07-24", "Period=W").Data[0]
    dates_month = w.tdays("2010-01-01", "2018-07-27", "Period=M").Data[0]
    
    data = get_data(dates_month,code='000905.SH')
    
    data.to_csv('data_month_zz500.csv')
    













