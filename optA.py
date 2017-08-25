import mt_auto.beta.mt_auto as mt
import pandas as pd
import numpy as np
import datetime as dt
import calendar
import os
import sys
 
 
    
peak_sum_cols = ['trade_volume','open_interest']
peak_mean_cols: 
cp_ratio_cols:

    
def next_monthlies(d): # must test len of option price history to ensure 1: that purchase is possible on d2 and 2: that there is at least 12 days of data
    f,g,h,k = d + dt.timedelta(days=95),d + dt.timedelta(days=180),d + dt.timedelta(days=270),d + dt.timedelta(days=360)
    f,g,h,k = dt.date(f.year, f.month, 15),dt.date(g.year, g.month, 15),dt.date(h.year, h.month, 15),dt.date(k.year, k.month, 15)
    f,g,h,k = (f + dt.timedelta(days=(calendar.FRIDAY - f.weekday()) % 7)),(g + dt.timedelta(days=(calendar.FRIDAY - g.weekday()) % 7)),(h + dt.timedelta(days=(calendar.FRIDAY - h.weekday()) % 7)),(k + dt.timedelta(days=(calendar.FRIDAY - k.weekday()) % 7))
    f,g,h,k = pd.to_datetime(f,infer_datetime_format=True),pd.to_datetime(g,infer_datetime_format=True),pd.to_datetime(h,infer_datetime_format=True),pd.to_datetime(k,infer_datetime_format=True)
    return f,g,h,k
     
def one_monthly(d): # must test len of option price history to ensure 1: that purchase is possible on d2 and 2: that there is at least 12 days of data
    f = d + dt.timedelta(days=32)
    f = dt.date(f.year, f.month, 15)
    f = (f + dt.timedelta(days=(calendar.FRIDAY - f.weekday()) % 7))
    f = pd.to_datetime(f,infer_datetime_format=True)
    return f
 
def two_monthly(d): # must test len of option price history to ensure 1: that purchase is possible on d2 and 2: that there is at least 12 days of data
    f = d + dt.timedelta(days=65)
    f = dt.date(f.year, f.month, 15)
    f = (f + dt.timedelta(days=(calendar.FRIDAY - f.weekday()) % 7))
    f = pd.to_datetime(f,infer_datetime_format=True)
    return f
     
def three_monthly(d): # must test len of option price history to ensure 1: that purchase is possible on d2 and 2: that there is at least 12 days of data
    f = d + dt.timedelta(days=95)
    f = dt.date(f.year, f.month, 15)
    f = (f + dt.timedelta(days=(calendar.FRIDAY - f.weekday()) % 7))
    f = pd.to_datetime(f,infer_datetime_format=True)
    return f
 
def four_monthly(d): # must test len of option price history to ensure 1: that purchase is possible on d2 and 2: that there is at least 12 days of data
    f = d + dt.timedelta(days=125)
    f = dt.date(f.year, f.month, 15)
    f = (f + dt.timedelta(days=(calendar.FRIDAY - f.weekday()) % 7))
    f = pd.to_datetime(f,infer_datetime_format=True)
    return f
 
def five_monthly(d): # must test len of option price history to ensure 1: that purchase is possible on d2 and 2: that there is at least 12 days of data
    f = d + dt.timedelta(days=155)
    f = dt.date(f.year, f.month, 15)
    f = (f + dt.timedelta(days=(calendar.FRIDAY - f.weekday()) % 7))
    f = pd.to_datetime(f,infer_datetime_format=True)
    return f
     
def six_monthly(d): # must test len of option price history to ensure 1: that purchase is possible on d2 and 2: that there is at least 12 days of data
    f = d + dt.timedelta(days=180)
    f = dt.date(f.year, f.month, 15)
    f = (f + dt.timedelta(days=(calendar.FRIDAY - f.weekday()) % 7))
    f = pd.to_datetime(f,infer_datetime_format=True)
    return f
     
def date_diff(d1,d2):
    return (d1-d2).days
     
class stock(object):
     
    def __init__(self,symb,direct):
        self.symb = symb
        self.direct = direct
        self.stk = mt.dicts(direct).stk_dict(self.symb)
         
    def get_underlying(self,d):
        return self.stk.loc[d,'Open'], self.stk.loc[d,'High'], self.stk.loc[d,'Low']
         
    def get_open(self,d):
        return self.stk.loc[d,'Open']
     
    def get_hi(self,d):
        return self.stk.loc[d,'High']
         
    def get_lo(self,d):
        return self.stk.loc[d,'Low']
         
class option_analyzer(object):
    peak_cols = [str(i) + '_Week_Exp_Max_'+col for i in np.arange(9) for col in ['U_Hi_Factor','U_Lo_Factor','trade_volume','open_interest']]
    opt_cols = ['Underlying_Open','Underlying_High','Underlying_Low','1_Month_Exp','1_Month_Exp','3_Month_Exp','4_Month_Exp','5_Month_Exp','6_Month_Exp']
    def __init__(self,symb,direct):
        self.symb = symb
        self.direct = direct
        self.options = ['call','put']
        self.stk = stock(self.symb,self.direct)
        report_path = self.direct.db_dir + 'Bollocks\\'
        if os.path.isfile(report_path):
            self.report = pd.read_csv()
        else:
            self.report = pd.DataFrame(columns=report_cols)
         

 
    def option_analysis_gen_cols(opt_data):
        for col in opt_cols:
            opt_data[col] = 0
        opt_data['Days_to_Exp'] = opt_data['expiration']-opt_data['quote_date']
        opt_data['Days_to_Exp'] = [d.days for d in opt_data['Days_to_Exp']]
        exp_week = (opt_data['Days_to_Exp']/7)-1
        opt_data['Exp_Week'] = round(exp_week,0)+(exp_week%7>0)
        opt_data['Exp_Week'] = opt_data['Exp_Week'].astype(int)
        opt_data = opt_data[opt_data['Exp_Week'] <= 30]
        for d in opt_data['quote_date'].unique():
            d = pd.to_datetime(d,infer_datetime_format=True)
            opt_data.loc[opt_data['quote_date']==d,['Underlying_Open','Underlying_High','Underlying_Low']] = stk.get_underlying(d)
            opt_data.loc[(opt_data['quote_date']==d) & (opt_data['expiration']==one_monthly(d)),'1_Month_Exp'] = True
            opt_data.loc[(opt_data['quote_date']==d) & (opt_data['expiration']==two_monthly(d)),'2_Month_Exp'] = True
            opt_data.loc[(opt_data['quote_date']==d) & (opt_data['expiration']==three_monthly(d)),'3_Month_Exp'] = True
            opt_data.loc[(opt_data['quote_date']==d) & (opt_data['expiration']==four_monthly(d)),'4_Month_Exp'] = True 
            opt_data.loc[(opt_data['quote_date']==d) & (opt_data['expiration']==five_monthly(d)),'5_Month_Exp'] = True 
            opt_data.loc[(opt_data['quote_date']==d) & (opt_data['expiration']==six_monthly(d)),'6_Month_Exp'] = True          
        opt_data['U_Hi_Factor'] = (opt_data['Underlying_High']-opt_data['Underlying_Open'])/opt_data['Underlying_Open']
        opt_data['U_Lo_Factor'] = (opt_data['Underlying_Low']-opt_data['Underlying_Open'])/opt_data['Underlying_Open']
        opt_data['Moneyness'] = round(((opt_data['strike']/opt_data['Underlying_Open'])-1)*100,0)
        return opt_data
   
     
    def oa_callput_ratios_sum(self,opt_data,col):
        #volume & OI
        opt_data = opt_data[(opt_data['Moneyness'] <= 5) & (opt_data['Moneyness'] >= -5) & (opt_data['Exp_Week']<=8),['quote_date','Exp_Week',col]]
        for d in opt_data['quote_date'].unique():
            data = opt_data[opt_data['quote_date']==d]
            for exp in np.arange(9):
                data = data[data['Exp_Week']==exp]
                for stx in np.arange(6):
                    call = data[col][{data['option_type'].isin(['c','C'])) & (opt_data['Moneyness'] == stx)].sum()
                    put = data[col][{data['option_type'].isin(['p','P'])) & (opt_data['Moneyness'] == (stx*-1))].sum()
                    report.loc[d,str(stx)+'_Stx_CP_'+col+'_Ratio_Week_'+str(exp)] = call/put
        return report
                                     
     def oa_callput_ratios_mean(self,opt_data,col):
        #hi/lo factors
        opt_data = opt_data[(opt_data['Moneyness'] <= 5) & (opt_data['Moneyness'] >= -5) & (opt_data['Exp_Week']<=8),['quote_date','Exp_Week',col]]
        for d in opt_data['quote_date'].unique():
            data = opt_data[opt_data['quote_date']==d]
            for exp in np.arange(9):
                data = data[data['Exp_Week']==exp]
                for stx in np.arange(6):
                    call = data[col][{data['option_type'].isin(['c','C'])) & (opt_data['Moneyness'] == stx)].mean()
                    put = data[col][{data['option_type'].isin(['p','P'])) & (opt_data['Moneyness'] == (stx*-1))].mean()
                    report.loc[d,str(stx)+'_Stx_CP_'+col+'_Ratio_Week_'+str(exp)] = call/put
        return report
      
    def option_sma(self,report,col):
        sma_range = (5,10,21,34,55,89,144,233)
        sma_combos = it.permutations(sma_range,2)
        for p in sma_range:
            s = pd.Series(report[col].rolling(center=False,window=p).mean(), name=col+'_SMA_' + str(p))
            ratio = pd.Series(report[col]/s, name=col+'_SMA_' + str(p)+'_Daily_Ratio')
            report = report.join(s)
            report = report.join(ratio)
        return report
     
    factor_cols = {'call':{'hi':'U_Hi_Factor','lo':'U_Lo_Factor'},'put':{'hi':'U_Lo_Factor','lo':'U_Hi_Factor'}
    #cycle through days, exps, and option types to generate data.
                  
    def oa_option_hi_lo_factors(self,opt_data):
       report = pd.DataFrame()
       opt_data = opt_data[(opt_data['Moneyness'] <= 5) & (opt_data['Moneyness'] >= -5) & (opt_data['Exp_Week']<=8),['open','high','low','U_Hi_Factor','U_Lo_Factor' ]]
       for d in opt_data['quote_date'].unique():
           raw_data = opt_data[(opt_data['quote_date']==d)]
           for expir in raw_data['expiration'].unique():
               data = raw_data[raw_data['expiration']=expir]
               for typ in data['option_type'].unique():
                   data = data[data['option_type']==typ]
                   report.at[d,typ+'_Hi_Factor'] = ((data['high']-data['open'])/-data['open'])/data['open'])/data[factor_cols[typ]['hi']]
                   report.at[d,typ+'_Lo_Factor'] = ((data['low']-data['open'])/data['open'])/data[factor_cols[typ]['lo']]
       return report
                   
    def daily_volumes_sum(self,report,opt_data):
        for col in ['trade_volume','open_interest']:
            for name, group in opt_data.groupby(['quote_date','Exp_Week'])[col]:
                report[str(name[1])+'_Week_Exp_'+col] = 0
                report.loc[name[0],str(name[1])+'_Week_Exp_'+col] = group.sum()
        return report
                   
                   
    def daily_volumes_mean(self,report,opt_data):
        col = 'implied_volatility_eod'
        for name, group in opt_data.groupby(['quote_date','Exp_Week'])[col]:
            report[str(name[1])+'_Week_Exp_'+col] = 0
            report.loc[name[0],str(name[1])+'_Week_Exp_'+col] = group.mean()
        return report
    
    def oa_peak_sum(self,report,opt_data,col):
        report = pd.DataFrame(columns=peak_cols)
        opt_data = opt_data[(opt_data['Moneyness'] <= 10) & (opt_data['Moneyness'] >= -10) & (opt_data['Exp_Week']<=8)]
        group = opt_data.groupby(['Moneyness','Exp_Week'],axis=0)[col].sum()
        for exp in group.index.get_level_values(level=1):
            report.loc[d,str(exp)+'_Week_Exp_Max_'+col] = int(group.loc[:,exp].idxmax())
        return report
       
    def oa_peak_mean(self,report,opt_data,col):
        report = pd.DataFrame(columns=peak_cols)
        opt_data = opt_data[(opt_data['Moneyness'] <= 10) & (opt_data['Moneyness'] >= -10) & (opt_data['Exp_Week']<=8)]
        group = opt_data.groupby(['Moneyness','Exp_Week'],axis=0)[col].mean()
        idx = pd.IndexSlice     
        for exp in group.index.get_level_values(level=1):
            report.loc[d,str(exp)+'_Week_Exp_Max_'+col] = int(group.loc[:,exp].idxmax())
        return report
         
       
      dets = ...
       
      def update_opt_data(max_exp):
         addendum = load_calls_puts(max_exp.year)
         addendum = option_analysis_gen_cols(addendum)
        return addendum
         
      def load_calls_puts(chrono):
        for option in self.options:
          calls = pd.read_csv(''.join([self.direct.db_dir,symb,'\\Options\\',self.symb,'_',str(chrono.year),'_Calls.csv']),'rb',delimiter=',',parse_dates=['expiration','quote_date'],infer_datetime_format=True).sort_values(by='quote_date')
          puts = pd.read_csv(''.join([self.direct.db_dir,symb,'\\Options\\',self.symb,'_',str(chrono.year),'_Puts.csv']),'rb',delimiter=',',parse_dates=['expiration','quote_date'],infer_datetime_format=True).sort_values(by='quote_date')
          return pd.concat([calls,puts],axis=0)
           
      def option_analysis(self,dates):
        opt_analysis = pd.DataFrame(index=dates)
        start = dates.iloc[0]
        in_mem = start
        opt_data = load_calls_puts(start.year)
                   
        opt_data = opt_data[opt_data['quote_date'].isin(dates)]           
        opt_data = option_analysis_gen_cols(opt_data)
        dv_sum = daily_volumes_sum(self,report,opt_data)
        sma_cols = dv_sum.columns
        dv_mean = daily_volumes_mean(self,report,opt_data)
        sma_cols.extend(dv_mean.columns)
        opt_analysis = pd.concat([opt_analysis,dv_sum],axis=1)
        opt_analysis = pd.concat([opt_analysis,dv_mean],axis=1)
        opt_analysis = pd.concat([opt_analysis,oa_option_hi_lo_factors(opt_data)],axis=1)  
        for d in dates:
          max_exp = six_monthly(d)
          if d.year != max_exp.year:
            addendum = update_opt_data(max_exp)
            opt_data = pd.concat([opt_data[d:],addendum],axis=0)
            in_mem = opt_data['expiration'].max()
          optdata = opt_data[opt_data['quote_date'] == d]
          for col in sma_cols:
             opt_analysis = pd.concat([opt_analysis,option_sma(optdata,col)],axis=1)
          for col in peak_sum_cols:
             opt_analysis = pd.concat([opt_analysis,oa_peak_sum(opt_analysis,opt_data,col)],axis=1)
          for col in peak_mean_cols:
             opt_analysis = pd.concat([opt_analysis,oa_peak_mean(opt_analysis,opt_data,col)],axis=1)
          for col in cp_ratio_cols:
             opt_analysis = pd.concat([opt_analysis,oa_callput_ratios_mean(opt_data,col)],axis=1)
        opt_analysis.to_csv(...
        pickle.dump(opt_analysis...)
        return opt_analysis
          
             
'''
 
 
1. Daily Volume at weeks 0-8 over various moving averages (e.g. 30/60/90/200 day SMAs) - add Daily Market
2. Strike/Moneyness of highest Volume peak for each day at exps weeks 0-8 - add Daily Market
3. Strike/Moneyness of highest Open Interest peak for each day at exps weeks 0-8 - add Daily Market
4. Strikes/Moneyness with highest Hi & Lo factors for each day at exps weeks 0-8 - add Daily Market
5. Hi & Lo Factors - Gets the mean Hi/Lo factors for exps weeks 0-8 for each day for strikes within 2.5% of the money per option type - add Daily Market
6. Call vs Put volume
7. Call vs Put Hi/Lo Factors
     
 
 
1. pre-analysis 
    1. daily volume at weeks 1-8
    2. Strike/Moneyness of highest Volume peak for each day at exps weeks 0-8 - add Daily Market
    3. Strike/Moneyness of highest Open Interest peak for each day at exps weeks 0-8 - add Daily Market
    4. Strikes/Moneyness with highest Hi & Lo factors for each day at exps weeks 0-8 - add Daily Market
    5. Hi & Lo Factors - Gets the mean Hi/Lo factors for exps weeks 0-8 for each day for strikes within 2.5% of the money per option type - add Daily Market
    6. Call vs Put volume
    7. Call vs Put Hi/Lo Factors
2. final-analysis 
    1. Volume/OI SMAs
    2. Ratios of above
    3. Crossovers of above
'''
