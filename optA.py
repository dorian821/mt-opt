1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
30
31
32
33
34
35
36
37
38
39
40
41
42
43
44
45
46
47
48
49
50
51
52
53
54
55
56
57
58
59
60
61
62
63
64
65
66
67
68
69
70
71
72
73
74
75
76
77
78
79
80
81
82
83
84
85
86
87
88
89
90
91
92
93
94
95
96
97
98
99
100
101
102
103
104
105
106
107
108
109
110
111
112
113
114
115
116
117
118
119
120
121
122
123
124
125
126
127
128
129
130
131
132
133
134
135
136
137
138
139
140
141
142
143
144
145
146
147
148
149
150
151
152
153
154
155
156
157
158
159
160
161
162
163
164
165
166
167
168
169
170
171
172
173
174
175
176
177
178
179
180
181
182
183
184
185
186
187
188
189
190
191
192
193
194
195
196
197
198
199
200
201
202
203
204
205
206
207
208
209
210
211
212
213
214
215
216
217
218
219
220
221
222
223
224
225
226
227
228
229
230
231
232
233
234
235
236
237
238
239
240
241
242
243
import mt_auto.beta.mt_auto as mt
import pandas as pd
import numpy as np
import datetime as dt
import calendar
import os
import sys
 
 
    
   sma_cols = [   ]
    
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
    peak_cols = [str(i) + '_Week_Exp_Max_'+col for i in np.arange(8)+1 for col in ['U_Hi_Factor','trade_volume','open_interest']]
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
         
    def daily_volumes(self,report,opt_data):
        for col in ['trade_volume','open_interest']:
            for name, group in opt_data.groupby(['quote_date','Exp_Week'])[col]:
                report[str(name[1])+'_Week_Exp_'+col] = 0
                report.loc[name[0],str(name[1])+'_Week_Exp_'+col] = group.sum()
        return report
    
    def oa_peak_sum(self,report,opt_data,col):
        report = pd.DataFrame(columns=peak_cols)
        opt_data = opt_data[(opt_data['Moneyness'] <= 10) & (opt_data['Moneyness'] >= -10) & (opt_data['Exp_Week']<=8)]
        group = opt_data.groupby(['Moneyness','Exp_Week'],axis=0)[col].sum()
        idx = pd.IndexSlice     
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
   
     
    def oa_callput_ratios(self,opt_data,col):
        opt_data = opt_data[(opt_data['Moneyness'] <= 5) & (opt_data['Moneyness'] >= -5) & (opt_data['Exp_Week']<=8)]
        for d in opt_data['quote_date']:
            data = opt_data[opt_data['quote_date']==d]
            for exp in np.arange(9):
                data = data[data['Exp_Week']==exp]
                for stx in np.arange(6):
                    call = data[col][{data['option'].isin('c')) & (opt_data['Moneyness'] == stx)].sum()
                    put = data[col][{data['option'].isin('p')) & (opt_data['Moneyness'] == stx)].sum()
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
    for d in opt_data['quote_date'].unique():
        raw_data = opt_data[(opt_data['quote_date']==d) & (opt_data['strike']<(opt_data['Underlying_Open']*1.025)) & (opt_data['strike']<(opt_data['Underlying_Open']*.975))&(expiration filter)]
        for expir in raw_data['expiration'].unique():
            data = raw_data[raw_data['expiration']=expir]
            for typ in data['option_type'].unique():
                data = data[data['option_type']==typ]
                option_report.at[d,typ+'_Hi_Factor'] = ((data['high']-data['open'])/-data['open'])/data['open'])/data[factor_cols[typ]['hi']]
                option_report.at[d,typ+'_Lo_Factor'] = ((data['low']-data['open'])/data['open'])/data[factor_cols[typ]['lo']]
 
         
       
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
           
      def option_analysis(dates,):
        opt_analysis = pd.DataFrame(columns=opt_analysis_cols)
        start = dates.iloc[0]
        in_mem = start
        opt_data = load_calls_puts(start.year)
        opt_data = option_analysis_gen_cols(opt_data)
        for d in dates:
          max_exp = get_max_exp(d)
          if d.year != max_exp.year:
            addendum = update_opt_data(max_exp)
            opt_data = pd.concat([opt_data[d:],,axis=0)
            in_mem = opt_data['expiration'].max()
          optdata = opt_data[opt_data['quote_date'] == d]
          for col in sma_cols:
            opt_analysis.at[d,sma_cols] = option_analysis_sma_ratios(optdata,col)
          for col in peak_cols:
            opt_analysis.at[d,peak_cols] = option_analysis_peak(optdata,col,max())
          for col in mean_cols:
              opt_analysis.at[d,mean_cols] = option_analysis_mean(optdata,col,max())
          for col in cp_ratio_cols:
            opt_analysis.at[d,cp_ratio_cols] = option_analysis_callput_ratios(opt_data,col)
             
             
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
