import pandas as pd
import numpy as np
import os
import datetime as dt
import pandas_datareader.data as web
from yahoo_finance import Share
import scipy as sp
from datetime import datetime
import calendar
import sys


#			['underlying_symbol', 'root', 'expiration', 'strike', 'option_type',
#			 'open', 'high', 'low', 'close', 'trade_volume', 'bid_size_1545',
#			 'bid_1545', 'ask_size_1545', 'ask_1545', 'underlying_bid_1545',
#			 'underlying_ask_1545', 'implied_underlying_price_1545',
#			 'active_underlying_price_1545', 'implied_volatility_1545', 'delta_1545',
#			 'gamma_1545', 'theta_1545', 'vega_1545', 'rho_1545', 'bid_size_eod',
#			 'bid_eod', 'ask_size_eod', 'ask_eod', 'underlying_bid_eod',
#			 'underlying_ask_eod', 'vwap', 'open_interest', 'delivery_code']


def mt_reporter(data):
	report = pd.DataFrame(index=np.arange(1))
	report['Expiration'] = str(data.ix[0]['expiration'].date())
	report['Day_1_Volume'] = data.ix[0]['trade_volume']
	report['Days_to_Expiration'] = ((data.ix[0]['expiration'] - data.index[0])/ np.timedelta64(1, 'D')).astype(int)
	report['Option_Symbol'] = ''.join([str(data.ix[0]['expiration'].date()),',',data.ix[0]['option_type'],',',str(data.ix[0]['strike']),',',symb])
	report['Trade_Date'] = data.index[0]
	tddate = data.index[0]
	data = data[1:20]
	
	data['mid_1545'] = (data['bid_1545'] + data['ask_1545'])/2
	if len(data) < 19:
		ix = pd.DatetimeIndex(start=data.index[-1] + dt.timedelta(days=1), end=data.index[-1] + dt.timedelta(days=19-len(data)), freq='D')
		dx = pd.DataFrame(index=ix,columns=data.columns)
		data = pd.concat([data,dx],axis=0)
	data = data[['open','high','low','close','mid_1545','trade_volume']].replace(to_replace=0, value=np.nan).fillna(method='bfill',axis=1)
	if data.ix[0]['open'] != 0:
		d2op = data.ix[0]['open']
		ds = np.arange(2,len(data)+2)
		lolabels = ["D2Lo/D2op",	"D3Lo/D2op",	"D4Lo/D2op",	"D5Lo/D2op",	"D6Lo/D2op",	"D7Lo/D2op",	"D8Lo/D2op",	"D9Lo/D2op",	"D10Lo/D2op",	"D11Lo/D2op",	"D12Lo/D2op",	"D13Lo/D2op",	"D14Lo/D2op",	"D15Lo/D2op",	"D16Lo/D2op",	"D17Lo/D2op",	"D18Lo/D2op",	"D19Lo/D2op",	"D20Lo/D2op"]
		hilabels = ["D2Hi/D2op",	"D3Hi/D2op",	"D4Hi/D2op",	"D5Hi/D2op",	"D6Hi/D2op",	"D7Hi/D2op",	"D8Hi/D2op",	"D9Hi/D2op",	"D10Hi/D2op",	"D11Hi/D2op",	"D12Hi/D2op",	"D13Hi/D2op",	"D14Hi/D2op",	"D15Hi/D2op",	"D16Hi/D2op",	"D17Hi/D2op",	"D18Hi/D2op",	"D19Hi/D2op",	"D20Hi/D2op"]
		loratios = pd.DataFrame(index=lolabels, columns=np.arange(1))
		hiratios = pd.DataFrame(index=hilabels, columns=np.arange(1))
		lows = data['low'].reshape(19,1)
		highs = data['high'].reshape(19,1)
		loratios[[0]] = lows/d2op
		hiratios[[0]] = highs/d2op
		
		d2 = data.ix[1]
		#d2 = d2[['open','high','low','close','bid_1545','ask_1545']].replace(to_replace=0, value=np.nan).fillna(method='bfill')
		report['d2op'] = data.ix[0]['open']	
		report['d2hi'] = data.ix[0]['high']
		report['d2lo'] = data.ix[0]['low']
		report['d2cl'] = data.ix[0]['close']
		report['d2ivst'] = data.ix[0]['trade_volume']*d2op*10
		per1 = data[1:5]
		#per1 = per1[['open','high','low','close','bid_1545']].replace(to_replace=0, value=np.nan).fillna(method='bfill',axis=1)
		#per1 = per1[['open','high','low','close']].resample('5D').agg({'open': 'first', 'high': 'max','low': 'min', 'close': 'last'}) #need to reformat, giving false values
		report['per1op'] = per1.ix[0]['open']
		per1op = per1.ix[0]['open']
		report['per1hi'] = per1['high'].max()
		report['per1lo'] = per1['low'].min()
		report['per1cl'] = per1.ix[-1]['close']
		report['per1ivst'] = data[2:6]['trade_volume'].mean()*per1op*10
		per2 = data[5:10]
		#per2 = per2[['open','high','low','close','bid_1545']].replace(to_replace=0, value=np.nan).fillna(method='bfill',axis=1)
		#per2 = per2[['open','high','low','close']].resample('4D').agg({'open': 'first', 'high': 'max','low': 'min', 'close': 'last'})#need to reformat, giving false values
		report['per2op'] = per2.ix[0]['open']
		per2op = per2.ix[0]['open']
		report['per2hi'] = per2['high'].max()
		report['per2lo'] = per2['low'].min()
		report['per2cl'] = per2.ix[-1]['close']
		report['per2ivst'] = data[6:11]['trade_volume'].mean()*per2op*10
		try:
			per3 =  data.ix[10:]
			#per3 = per3[['open','high','low','close','bid_1545']].replace(to_replace=0, value=np.nan).fillna(method='bfill',axis=1)
			#per3 = per3[['open','high','low','close']].resample('11D').agg({'open': 'first', 'high': 'max','low': 'min', 'close': 'last'}) #need to reformat, giving false values
			report['per3op'] = per3.ix[0]['open']
			per3op = per3.ix[0]['open']
			report['per3hi'] = per3['high'].max()
			report['per3lo'] = per3['low'].min()
			report['per3cl'] = per3.ix[-1]['close']
			report['per3ivst'] = data[10:]['trade_volume'].mean()*per3op*10
			report['Avg_Investible_Volume'] = (report['per3ivst']+report['per2ivst']+report['per1ivst']+report['d2ivst'])/4
		except:
			report['per3op'] = 'NA'
			per3op = 'NA'
			report['per3hi'] = 'NA'
			report['per3lo'] = 'NA'
			report['per3cl'] = 'NA'
			report['per3ivst'] = 0
			report['Avg_Investible_Volume'] = (report['per2ivst']+report['per1ivst']+report['d2ivst'])/3

		report['Trade_Hi'] = hiratios.max()
		report['Trade_Lo'] = loratios.min()
		report['per1hi_percent'] = report['per1hi']/d2op
		report['per2hi_percent'] = report['per2hi']/d2op
		report['per3hi_percent'] = report['per3hi']/d2op
		report['per1lo_percent'] = report['per1lo']/d2op
		report['per2lo_percent'] = report['per2lo']/d2op
		report['per3lo_percent'] = report['per3lo']/d2op	
		report['per1cl_percent'] = report['per1cl']/d2op
		report['per2cl_percent'] = report['per2cl']/d2op
		report['per3cl_percent'] = report['per3cl']/d2op
		loratios = loratios.T
		hiratios = hiratios.T
		report['Trade_Hi_Profit'] = report['Trade_Hi'] -1
		rprt = pd.concat([report,hiratios,loratios],axis=1)
	else: 
		rprt = np.nan
	return rprt	
	
def next_monthly(d): # must test len of option price history to ensure 1: that purchase is possible on d2 and 2: that there is at least 12 days of data
	f = d + dt.timedelta(days=30)
	f = (f + dt.timedelta(days=(calendar.FRIDAY - f.weekday()) % 7))
	e = d + dt.timedelta(days=(calendar.FRIDAY - f.weekday()) % 7)
	return e,f
	
def find_strike(array,value,movement):
	if len(array) > 1:
		array = np.sort(array, axis=0, kind='quicksort', order=None)
		idx = np.abs(array-value).argmin()
		stx = array[idx+movement]
	else: 
		stx = array
	return stx

def 20k_sortby(data,volume,days_to_expiration):
	underlying = pd.Series(data=((data['underlying_bid_eod']+data['underlying_ask_eod'])/2),index=data.index)
	stx = pd.Series(data=(data['strike'] -underlying),index=data.index)
	d2x = pd.Series(data=(data['expiration']-data['quote_date']/np.timedelta64(1, 'D')).astype(int), index=data.index)
	sorted_data = data[(data['trade_volume']>=volume)&(d2x<=days_to_expiration)&(np.abs(stx)<underlying*1.05)]
	return sorted_data

types = ('put','call')
symbs = ('AAPL','')
years = ('2016','2017')
c = 'trade_volume'
relation = '>='
value = 20000


	
	
for symb in symbs:
	dir = 'C:\\Users\\asus\\Documents\\Quant\\Database\\' + symb + '\\Options\\'
	os.makedirs('C:\\Users\\asus\\Dropbox\\Outlines\\MTAUTO-PYTHON\\Sort_By\\', exist_ok=True)
	rprtsortbypath = 'C:\\Users\\asus\\Dropbox\\Outlines\\MTAUTO-PYTHON\\Sort_By\\' 'Sort_By_Summary_Report_' + symb + '_' + c + '-' + str(value) + '.csv'	
	rawsortbypath = 'C:\\Users\\asus\\Dropbox\\Outlines\\MTAUTO-PYTHON\\Sort_By\\' 'Sort_By_Raw_Data_' + symb + '_' + c + str(value) +'.csv'
	report = pd.DataFrame()
	rawreport = pd.DataFrame()
	for type in types:
		if symb == '^VIX':
			symb = 'VXX'
		for year in years:
			new_year = pd.to_datetime(year+'-01-01', infer_datetime_format=True)
			if type == 'call':
				diropt = ''.join([dir,symb,'_',year,'_Calls.csv'])
				x = 1
			elif type == 'put':
				diropt = ''.join([dir,symb,'_',year,'_Puts.csv'])
				x = -1
			try:
				data = pd.read_csv(diropt,'rb',delimiter=',',parse_dates=['expiration','quote_date'],infer_datetime_format=True).sort_index(axis=0)
			except: 
				continue
			#filter near the money options
			data.drop_duplicates(['expiration', 'strike','quote_date'],inplace=True)
			sorted_data = 20k_sortby(data=data,volume=value,days_to_expiration=5)

			

			for i in sorted_data.index:
				opt = data[(data['expiration'] == sort.ix[i]['expiration'])&(data['strike'] == sort.ix[i]['strike'])].set_index('quote_date')
				opt = opt[sorted_data.ix[i]['quote_date']:]
				if (len(opt) > 2):
					if (opt.ix[1]['open'] != 0): 
						rprt = mt_reporter(opt)
						report = pd.concat([report,rprt],axis=0)								
						rawreport = pd.concat([rawreport,opt],axis=0)
					else:
						continue
				else:
					continue
					
	report.to_csv(rprtsortbypath)
	rawreport.to_csv(rawsortbypath)					
				
				
				
				
				
				
				
''''''''''
exp1, exp2 = next_monthly(q)
days = sort[(sort['expiration'] >=exp1) & (sort['expiration'] <=exp2)].reset_index()
p = (days.ix[0]['underlying_bid_eod']+days.ix[0]['underlying_ask_eod'])/2
stx = find_strike(days['strike'].unique(),p,x)
for e in days['expiration'].unique():
opt = data[(data['expiration'] == e)]
opt = opt[(opt['strike'] == int(stx))].set_index('quote_date')
opt = opt[q:]					
if (len(opt) > 13):
if (opt.ix[1]['open'] != 0): 
rprt = mt_reporter(opt)
report = pd.concat([report,rprt],axis=0)
report['Expiration'] = e
rawreport = pd.concat([rawreport,opt],axis=0)
else:
continue
else:
continue
						
report.to_csv(rprtsortbypath)
rawreport.to_csv(rawsortbypath)					

#filter by criteria
#for unique quote date 			
#for each expiration from 0 - 2 weeks out  
#get 5 strikes surrounding ATM
#report
'''

'''''''''''
exps = sort['expiration'].unique()
for exp in exps:
strikes =  sort['strike'].unique()
for strike in strikes:					
srt = sort[(sort['expiration'] == exp) & (sort['strike'] == strike)]
opt = data[(data['expiration'] == exp) & (data['strike'] == strike)].reset_index(drop=False)
opt['Option_Symbol'] = ''.join([str(opt.ix[0]['expiration'].date()),',',opt.ix[0]['option_type'],',',str(opt.ix[0]['strike']),',',symb])
opt.set_index('quote_date',inplace=True)
start = np.min(srt['quote_date'])				
opt = opt[start:]					
if (len(opt) > 13):
if (opt.ix[1]['open'] != 0): 
rprt = mt_reporter(opt)
report = pd.concat([report,rprt],axis=0)
rawreport = pd.concat([rawreport,opt],axis=0)
'''''''''''''''		

					
