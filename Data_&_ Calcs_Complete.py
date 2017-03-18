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
import itertools as it

start = pd.to_datetime('2014-01-01', infer_datetime_format=True)
end =  pd.to_datetime('2016-12-31', infer_datetime_format=True)
symbs = ('SPY','QQQ','IWM','AAPL','^VIX','NFLX','C','X','MSFT','BABA','AMZN') #'SPY','QQQ','IWM','AAPL','^VIX')  #,'BAC','NFLX','AMZN','GLD','GDX','NFLX','C','X','MSFT','BABA','AMZN')'FB',) C,X,MSFT, BABA, AMZN, FB

os.makedirs('C:\\Users\\asus\\Dropbox\\Outlines\\MTAUTO-PYTHON\\Back-Tester\\Level_Class_Reports\\', exist_ok=True)
completed = os.listdir('C:\\Users\\asus\\Dropbox\\Outlines\\MTAUTO-PYTHON\\Back-Tester\\Level_Class_Reports\\')
if len(completed) > 0:
	#start = pd.Series(pd.to_datetime(x, infer_datetime_format=True).date() for x in completed).max() + dt.timedelta(days=1)	
	dets = web.DataReader('SPY', 'yahoo', start, end).index	
else:
	dets = web.DataReader('SPY', 'yahoo', start, end).index

x = pd.read_csv('C:\\Users\\asus\\Dropbox\\Outlines\\Quantile Test.csv','rb', delimiter=',')
initiate_time = dt.datetime.now()
c = 0

		
###########################################################Data & Calcs#####################################################
def day_2_ratios(data):
	D2Op = pd.Series(data['Open'].shift(-1),name='D2Op')
	trange = np.arange(20)+1
	for i in trange:
		j = i * -1
		lo = pd.Series((data['Low'].shift(j)/D2Op),name='D'+ str(i+1) + 'Lo/D2Op')
		hi = pd.Series((data['High'].shift(j)/D2Op),name='D'+ str(i+1) + 'Hi/D2Op')
		data = data.join(lo)
		data = data.join(hi)
	per1lo = pd.Series(data['Low'].shift(-5).rolling(4).min()/D2Op, name ='Per1Lo/D2Op')
	per1hi = pd.Series(data['High'].shift(-5).rolling(4).max()/D2Op, name ='Per1Hi/D2Op') #Include Close of D2
	per1cl = pd.Series(data['Close'].shift(-5)/D2Op, name='Per1Cl/D2Op')
	per2lo = pd.Series(data['Low'].shift(-9).rolling(4).min()/D2Op, name ='Per2Lo/D2Op')
	per2hi = pd.Series(data['High'].shift(-9).rolling(4).max()/D2Op, name ='Per2Hi/D2Op')
	per2cl = pd.Series(data['Close'].shift(-9)/D2Op, name='Per2Cl/D2Op')
	per3lo = pd.Series(data['Low'].shift(-14).rolling(5).min()/D2Op, name ='Per3Lo/D2Op')
	per3hi = pd.Series(data['High'].shift(-14).rolling(5).max()/D2Op, name ='Per3Hi/D2Op')
	per3cl = pd.Series(data['Close'].shift(-14)/D2Op, name='Per3Cl/D2Op')
	per4lo = pd.Series(data['Low'].shift(-19).rolling(5).min()/D2Op, name ='Per4Lo/D2Op')
	per4hi = pd.Series(data['High'].shift(-19).rolling(5).max()/D2Op, name ='Per4Hi/D2Op')
	per4cl = pd.Series(data['Close'].shift(-19)/D2Op, name='Per4Cl/D2Op')
	per5lo = pd.Series(data['Low'].shift(-24).rolling(5).min()/D2Op, name ='Per5Lo/D2Op')
	per5hi = pd.Series(data['High'].shift(-24).rolling(5).max()/D2Op, name ='Per5Hi/D2Op')
	per5cl = pd.Series(data['Close'].shift(-24)/D2Op, name='Per5cl/D2Op')
	trhi = pd.Series(data['High'].shift(-19).rolling(18).max()/D2Op, name ='Trade_Hi')
	trlo = pd.Series(data['Low'].shift(-19).rolling(18).min()/D2Op, name ='Trade_Lo')
	per12hi = pd.Series(data['High'].shift(-9).rolling(8).max()/D2Op, name ='Per1-2Hi/D2Op')
	per34hi = pd.Series(data['High'].shift(-19).rolling(10).max()/D2Op, name ='Per3-4Hi/D2Op')
	per12lo = pd.Series(data['Low'].shift(-9).rolling(8).min()/D2Op, name ='Per1-2Lo/D2Op')
	per34lo = pd.Series(data['Low'].shift(-19).rolling(10).min()/D2Op, name ='Per3-4Lo/D2Op')
	data = data.join(per12hi)
	data = data.join(per34hi)
	data = data.join(per12lo)
	data = data.join(per34lo)
	data = data.join(per1lo)
	data = data.join(per1hi)
	data = data.join(per1cl)
	data = data.join(per2lo)
	data = data.join(per2hi)
	data = data.join(per2cl)
	data = data.join(per3lo)
	data = data.join(per3hi)
	data = data.join(per3cl)
	data = data.join(per4lo)
	data = data.join(per4hi)
	data = data.join(per4cl)
	data = data.join(per5lo)
	data = data.join(per5hi)
	data = data.join(per5cl)
	data = data.join(trhi)
	return data

def stock_updater(symb,data):
	d = dt.date.today()
	stockcolumns = ['Open','High','Low','Close','Volume','Adj Close']
	stk = Share(symb)
	stock = pd.Series({'Open':stk.get_open(),'High': stk.get_days_high(),'Low': stk.get_days_low(),'Close': stk.get_price(),'Volume': stk.get_volume(), 'Adj Close':stk.get_price()}, name=d)
	stock = pd.to_numeric(stock)
	data = data.append(stock)
	return data
	
def SMA(data,column,ndays): 
	s = pd.Series(data[column].rolling(center=False,window=ndays).mean(), name='SMA_' + str(ndays)) 
	data = data.join(s)
	return data
	
def SMAs(data,column): 
	sma_range = (5,10,21,34,55,89,144,233)
	sma_combos = it.permutations(sma_range,2)
	for p in sma_range:
		s = pd.Series(data[column].rolling(center=False,window=p).mean(), name='SMA_' + str(p)) 
		data = data.join(s)
	for p in sma_range[1:]:
		s = pd.Series((data['SMA_'+str(p)]-data['SMA_5'])/data[column], name='SMA_' + str(p) + '_Over_SMA_5')
		data = data.join(s)
	for p in sma_combos:
		s = pd.Series(data=((data['SMA_'+str(p[0])]>data['SMA_'+str(p[1])]) & (data['SMA_'+str(p[0])].shift(1)<data['SMA_'+str(p[1])].shift(1))), name='SMA_' + str(p[0]) + '_X-Over_SMA_' + str(p[1]))
		data = data.join(s)
	return data
	
def EMAs(data,column): 
	ema_range = (5,10,21,34,55,89,144,233)
	ema_combos = it.permutations(ema_range,2)
	for p in ema_range:
		s = pd.Series(pd.ewma(data['Close'], span = p, min_periods = p - 1), name = 'EMA_' + str(p)) 
		data = data.join(s)
	for p in ema_range[1:]:
		s = pd.Series((data['EMA_'+str(p)]-data['EMA_5'])/data[column], name='EMA_' + str(p) + '_Over_EMA_5')
		data = data.join(s)
	for p in ema_combos:
		s = pd.Series(data=((data['EMA_'+str(p[0])]>data['EMA_'+str(p[1])]) & (data['EMA_'+str(p[0])].shift(1)<data['EMA_'+str(p[1])].shift(1))), name='EMA_' + str(p[0]) + '_X-Over_EMA_' + str(p[1]))
		data = data.join(s)
	return data
	
def MACD(data,nday1,nday2,sign):
	quick = pd.Series(pd.ewma(data['Close'], span = nday1, min_periods = nday1 - 1), name = 'EMA_' + str(nday1))
	data = data.join(quick)
	slow = pd.Series(pd.ewma(data['Close'], span = nday2, min_periods = nday2 - 1), name = 'EMA_' + str(nday2))
	data = data.join(slow)
	macd = pd.Series(data=quick-slow,index=data.index,name='MACD_'+str(nday1)+'-'+str(nday2))
	data = data.join(macd)
	signal = pd.Series(pd.ewma(macd, span = sign, min_periods = sign - 1), name = 'Signal_EMA_' + str(sign))
	data = data.join(signal)
	return data
	

def anchored_divergence(data,col,r,diff):
	zzz = pd.Series()
	data[col] = pd.Series(data[col].rolling(window=5).mean(),data.index,name=col)
	anchors = pd.Series(index=data.index,name=col+'_Anchors')
	flipped = pd.Series(data=np.abs(data[col])-diff,index=data.index,name='Flipped')
	flipped.ix[:r+1] = np.nan
	#data = data.join(flipped)
	for i in (flipped.index[flipped<=0]):
		z = flipped[i:i+dt.timedelta(days=r)].argmin()
		zzz = zzz.append(pd.Series([z]))
	zeros = pd.Series(zzz.unique(),name='Zeros')
	#data = data.join(zeros)
	#zeros.reset_index(drop=True,inplace=True)
	for z in zeros.index:
		if z == zeros.index[-1]:
			break
		zos = zeros[z:z+2] #.reset_index(drop=True) #,inplace=True)
		if data.ix[zos[z]:zos[z+1]][col].max() > 100:
			anchor = data.ix[zos[z]:zos[z+1]][col].idxmax()			
		elif data.ix[zos[z]:zos[z+1]][col].min() < -100:
			anchor = data.ix[zos[z]:zos[z+1]][col].idxmin()
		else:
			continue
			#print ("Error: No max or min found")
		anchors.at[anchor] = 1
	anchors.fillna(value=np.nan, inplace=True)
	data = data.join(anchors)
	anchor_indices = pd.Series(data=np.where((data[col+'_Anchors'] == 1), data.index, data.index[0]),index=data.index,name=col+'_Anchor_Indices').astype(data.index.dtype)
	anchor_indices[anchor_indices == data.index[0]] = np.nan
	anchor_indices.fillna(method='ffill', inplace=True)
	anchor_indices.fillna(value=0,inplace=True)
	data = data.join(anchor_indices)
	anchor_values = pd.Series(data=np.where((data[col+'_Anchors'] == 1), data[col], np.nan),index=data.index,name=col+'_Anchor_Values')
	anchor_values.fillna(method='ffill', inplace=True)
	anchor_values.fillna(value=0,inplace=True)
	data = data.join(anchor_values)
	typ_values = pd.Series(data=np.where((data[col+'_Anchors'] == 1), data['Typical'], np.nan),index=data.index,name=col+'_Typ_Values')
	typ_values.fillna(method='ffill', inplace=True)
	typ_values.fillna(value=0,inplace=True)
	data = data.join(typ_values)
	anchored_slopes = pd.Series(data=(data[col]-data[col+'_Anchor_Values'])/(((data.index-data[col+'_Anchor_Indices']) / np.timedelta64(1, 'D')).astype(int)),index=data.index,name=col+'_Anchored_Slopes')
	anchored_slopes.fillna(value=.0001,inplace=True)
	data = data.join(anchored_slopes)	
	typ_slopes = pd.Series(data=(data['Typical']-data[col+'_Anchor_Values'])/(((data.index-data[col+'_Anchor_Indices']) / np.timedelta64(1, 'D')).astype(int)),index=data.index,name=col+'_Typ_Slopes')
	typ_slopes[np.isinf(typ_slopes)] = .0001 
	data = data.join(typ_slopes)
	divergence_angle = pd.Series(data=np.arctan((data[col+'_Typ_Slopes']-data[col+'_Anchored_Slopes'])/(1+(data[col+'_Typ_Slopes']*data[col+'_Anchored_Slopes']))),index=data.index,name=col+'_Anchored_Divergence')
	data = data.join(divergence_angle)
	divergence_slope = pd.Series(data=divergence_angle.rolling(window=3).apply(three_linest),index=data.index,name=col+'_Anchored_Divergence_3p_Slope')
	data = data.join(divergence_slope)
	divergence = pd.Series(data= np.where((divergence_slope> -.0019) & (divergence_slope< 0),True,False),index=data.index,name=col + '_Divergence_2ndDiv')
	data = data.join(divergence)
	return data

def anchored_divergence_bool(data,col,r,diff,cutoff): #cutoff of 10%
	low_l = data[col].mean()-((data[col].max()-data[col].min())*cutoff)
	high_l = data[col].mean()+((data[col].max()-data[col].min())*cutoff)
	zzz = pd.Series()
	data[col] = pd.Series(data[col].rolling(window=5).mean(),data.index,name=col)
	anchors = pd.Series(index=data.index,name=col+'_Anchors')
	flipped = pd.Series(data=np.abs(data[col])-diff,index=data.index,name='Flipped')
	flipped.ix[:r+1] = np.nan
	#data = data.join(flipped)
	for i in (flipped.index[flipped<=0]):
		z = flipped[i:i+dt.timedelta(days=r)].argmin()
		zzz = zzz.append(pd.Series([z]))
	zeros = pd.Series(zzz.unique(),name='Zeros')
	#data = data.join(zeros)
	#zeros.reset_index(drop=True,inplace=True)
	for z in zeros.index:
		if z == zeros.index[-1]:
			break
		zos = zeros[z:z+2] #.reset_index(drop=True) #,inplace=True)
		if data.ix[zos[z]:zos[z+1]][col].max() > high_l:
			anchor = data.ix[zos[z]:zos[z+1]][col].idxmax()			
		elif data.ix[zos[z]:zos[z+1]][col].min() < low_l:
			anchor = data.ix[zos[z]:zos[z+1]][col].idxmin()
		else:
			continue
			#print ("Error: No max or min found")
		anchors.at[anchor] = 1
	anchors.fillna(value=np.nan, inplace=True)
	data = data.join(anchors)
	anchor_indices = pd.Series(data=np.where((data[col+'_Anchors'] == 1), data.index, data.index[0]),index=data.index,name=col+'_Anchor_Indices').astype(data.index.dtype)
	anchor_indices[anchor_indices == data.index[0]] = np.nan
	anchor_indices.fillna(method='ffill', inplace=True)
	anchor_indices.fillna(value=0,inplace=True)
	data = data.join(anchor_indices)
	anchor_values = pd.Series(data=np.where((data[col+'_Anchors'] == 1), data[col], np.nan),index=data.index,name=col+'_Anchor_Values')
	anchor_values.fillna(method='ffill', inplace=True)
	anchor_values.fillna(value=0,inplace=True)
	data = data.join(anchor_values)
	high_divergence = pd.Series(data=[(data['High']>data['High'].shift(1)) & (data[col]<=data[col+'_Anchor_Values'])],name=col+'High_Divergence')
	data = data.join(high_divergence)
	high_div_count = pd.Series(data=(np.where(data[col+'High_Divergence'] == True,data.index - data[col+'_Anchor_Indices'],np.nan),name=col+'High_Div_Count')
	data = data.join(high_div_count)			   
	low_divergence = pd.Series(data=[(data['Low']<data['Low'].shift(1)) & (data[col]>=data[col+'_Anchor_Values'])],name=col+'Low_Divergence')
	data = data.join(low_divergence)
	low_div_count = pd.Series(data=(np.where(data[col+'Low_Divergence'] == True,data.index - data[col+'_Anchor_Indices'],np.nan),name=col+'Low_Div_Count')
	data = data.join(low_div_count)			   
	return data

#Slope
def three_linest(y,x=np.arange(3),deg=1):
	l = np.polyfit(y=y, x=x, deg=1)[0]
	return l

def five_linest(y,x=np.arange(5),deg=1):
	l = np.polyfit(y=y, x=x, deg=1)[0]
	return l
	
def seven_linest(y,x=np.arange(7),deg=1):
	l = np.polyfit(y=y, x=x, deg=1)[0]
	return l
	
def slope(data,series, ndays, name):
	name = name + str(ndays)
	slope, intercept, r_value, p_value, std_err = pd.Series(pd.rolling_apply(arg=series, func=sp.stats.linregress(y=np.datetime64(series.index), x=series), window=3), name=name)
	data = data.join(slope)
	return data
	
def sloped(data,series, ndays, name):
	name = name + str(ndays)
	slope, intercept, r_value, p_value, std_err = pd.Series(pd.rolling_apply(arg=series, func=np.polyfit(y=np.datetime64(series.index), x=series), window=3), name=name)
	data = data.join(slope)
	data = data.fillna(value=0)
	return data
	
def SMA_slope(data,sma,ndays):
	slope = pd.Series((data[sma].rolling(window=3).apply(func=three_linest)),name=sma+'_'+str(ndays)+'_Slope')
	data = data.join(slope)
	return data
	
	#three_linest
 
# Exponentially-weighted Moving Average 
def EMA(data, ndays): 
	EMA = pd.Series(pd.ewma(data['Close'], span = ndays, min_periods = ndays - 1, name = 'EMA_' + str(ndays)) )
	return EMA

#Stochastics 
#D1_%K (14Per)
def FSTOCH(data, ndays=14):
	l = pd.Series(pd.rolling_min(data['Low'], ndays), name=str(ndays) + '_Min_Low')
	h = pd.Series(pd.rolling_max(data['High'], ndays), name=str(ndays) + '_Max_Hi')
	k = pd.Series(100 * (data['Close'] - l) / (h - l), name = '%K')
	data = data.join(k)
	data = data.join(l)
	data = data.join(h)
	data = data.fillna(value=0)
	return data
	
def SLSTOCH(data, ndays=3):
	d = pd.Series(data['%K'].rolling(center=False,window=ndays).mean(), name='%D')
	data = data.join(d)
	data = data.fillna(value=0)
	return data
	
def fstoch(lowp, highp, closep, period=14):
	""" calculate slow stochastic
	Fast stochastic calculation
	%K = (Current Close - Lowest Low)/(Highest High - Lowest Low) * 100
	%D = 3-day SMA of %K
	"""
	low_min = pd.rolling_min(lowp, period)
	high_max = pd.rolling_max(highp, period)
	k_fast = 100 * (closep - low_min)/(high_max - low_min)
	k_fast = k_fast.dropna()
	return k_fast
	
def sstoch(k,period):
	d = SMA(stk,k, period)
	data = data.fillna(value=0)
	return d

#D1_DIR
def STOCHDIR(data):
	d = pd.Series(data['%K'] - data['%D'], name='D1_DIR')
	dir = pd.Series(data=(data['%K'] - data['%D'])>0, name='D1_DIR_Bool')	
	data = data.join(d)
	data = data.join(dir)
	data = data.fillna(value=0)
	return data

#D1_FAUXSTO
def FAUXSTO(data):
	d = pd.Series((data['%K'] - data['%K'].shift(+1))*(data['%D'] - data['%D'].shift(+1)), name='D1_FAUXSTO')
	data = data.join(d)
	data = data.fillna(value=0)
	return data

#ABSOL_UP_D1_STO
def STOCHABSOLUP(data):
	d = pd.Series((data['D1_FAUXSTO'] > 0) & (data['D1_DIR'] > 0), name='ABSOL_UP_D1_STO')
	data = data.join(d)
	data = data.fillna(value=0)
	return data

#FRST_UP_OR_DWN_D1_%K_3PER_SLP
def K_UP_DOWN(data):
	km = pd.Series(pd.rolling_apply(arg=data['%K'],func=three_linest,window=3),name='%K_3PER_SLP')
	data = data.join(km)
	up = pd.Series(((data['%K_3PER_SLP'] > 0) & (data['%K_3PER_SLP'].shift(+1) < 0)), name='FIRST_UP_D1_%K_3PER_SLP')
	data = data.join(up)
	down = pd.Series(((km < 0) & (km.shift(+1) > 0)), name='FIRST_DOWN_D1_%K_3PER_SLP')
	data = data.join(down)
	return data


#CCI
def CCI(data, ndays): 
	TP = (data['High'] + data['Low'] + data['Close']) / 3 
	CCI = pd.Series(((TP - TP.rolling(center=False,window=ndays).mean())/ (0.015 * pd.rolling_std(TP, ndays))), name='CCI_'+str(ndays))
	name = 'CCI'
	data = data.join(CCI)
	data = data.fillna(value=0)	
	return data
	
def TYP(data):
	typ = pd.Series(((data['High'] + data['Low'] + data['Close']) / 3), name='Typical')
	data = data.join(typ)
	data = data.fillna(value=0)
	return data

#CCI Divergence
def CCI_DIVERG_T(data,ndays):
	ccislope = pd.rolling_apply(arg=data['CCI_'+str(ndays)],func=three_linest,window=3)
	typslope =  pd.rolling_apply(arg=data['Typical'],func=three_linest,window=3)
	updiverg = pd.Series(((ccislope > 0) & (typslope < 0)), name='CCI_'+str(ndays)+'_UP_DIVERGENCE_3')
	data = data.join(updiverg)
	downdiverg = pd.Series(((ccislope < 0) & (typslope > 0)), name='CCI_'+str(ndays)+'_DOWN_DIVERGENCE_3')
	data = data.join(downdiverg)
	data = data.fillna(value=0)
	return data
	
def CCI_DIVERG_F(data,ndays):
	ccislope = pd.rolling_apply(arg=data['CCI_'+str(ndays)],func=five_linest,window=5)
	typslope =  pd.rolling_apply(arg=data['Typical'],func=five_linest,window=5)
	updiverg = pd.Series(((ccislope > 0) & (typslope < 0)), name='CCI_'+str(ndays)+'_UP_DIVERGENCE_5')
	data = data.join(updiverg)
	downdiverg = pd.Series(((ccislope < 0) & (typslope > 0)), name='CCI_'+str(ndays)+'_DOWN_DIVERGENCE_5')
	data = data.join(downdiverg)
	data = data.fillna(value=0)
	return data
	
def CCI_DIVERG_S(data,ndays):
	ccislope = pd.rolling_apply(arg=data['CCI_'+str(ndays)],func=seven_linest,window=7)
	typslope =  pd.rolling_apply(arg=data['Typical'],func=seven_linest,window=7)
	updiverg = pd.Series(((ccislope > 0) & (typslope < 0)), name='CCI_'+str(ndays)+'_UP_DIVERGENCE_7')
	data = data.join(updiverg)
	downdiverg = pd.Series(((ccislope < 0) & (typslope > 0)), name='CCI_'+str(ndays)+'_DOWN_DIVERGENCE_7')
	data = data.join(downdiverg)
	data = data.fillna(value=0)
	return data

#Bollinger Bands
def BBANDS(data, ndays, nstdev):
 
	MA = pd.Series(data['Close'].rolling(center=False,window=ndays).mean())
	SD = pd.Series(pd.rolling_std(data['Close'], ndays))
	b1 = MA + (nstdev * SD)
	B1 = pd.Series(b1, name = 'Upper_BollingerBand') 
	data = data.join(B1) 
	b2 = MA - (nstdev * SD)
	B2 = pd.Series(b2, name = 'Lower_BollingerBand') 
	data = data.join(B2)
	#data = data.fillna(value=0)
	return data
	
#Bollinger Band Rank
def BBAND_RANK(data):
	rank = pd.Series(data=((data['Upper_BollingerBand']-data['Typical'])/(data['Upper_BollingerBand']-data['Lower_BollingerBand'])),index=data.index,name='BBand_Rank')
	data = data.join(rank)
	return data
	
 
#Bollinger Band Pinch XPAND
def BB_PINCH_XPAND(data):
	bp = pd.Series((data['Upper_BollingerBand'] > data['Upper_BollingerBand'].shift(+1)) & (data['Lower_BollingerBand'] < data['Lower_BollingerBand'].shift(+1)),name='BBANDS_PINCH')
	data = data.join(bp)
	bx = pd.Series((data['Upper_BollingerBand'] < data['Upper_BollingerBand'].shift(+1)) & (data['Lower_BollingerBand'] > data['Lower_BollingerBand'].shift(+1)),name='BBANDS_XPAND')
	data = data.join(bx)
	data = data.fillna(value=0)
	return data
 
#Blowest
def LLOWEST(data, ndays):
	l = pd.Series(pd.rolling_min(data['Low'],ndays), name='Lowest_Low_in_' + str(ndays))
	data = data.join(l)
	data = data.fillna(value=0)
	return data
	
def HLOWEST(data, ndays):
	l = pd.Series(pd.rolling_min(data['High'],ndays), name = 'Lowest_Hi_in_' + str(ndays))
	data = data.join(l)
	data = data.fillna(value=0)
	return data
	
def BLOWEST(data, ndays): #not named, must be explicitly assigned to column ['BLOWEST_' + ndays]
	bl = pd.Series((data['Lowest_Hi_in_' + str(ndays)] == data['High']) & (data['Low'] == data['Lowest_Low_in_' + str(ndays)]), name = 'BLOWEST_' + str(ndays))
	data = data.join(bl)
	data = data.fillna(value=0)
	return data
	
#D1_H_LO_LOWEST_IN_X
def HLLOWESTIN(data, ndays):
	loin = pd.rolling_min(data['Low'], ndays)
	hloin = pd.rolling_min(data['High'], ndays)
	hlin = pd.Series((hloin == data['High']) & (data['Low'] == loin), name = 'D1_H_LO_LOWEST_IN_' + str(ndays))
	data = data.join(hlin)
	data = data.fillna(value=0)
	return data
	
def COUNT_BLOWEST(data, ndays, pdays):
	ct = pd.DataFrame(data=np.where(data[''.join(['BLOWEST_' + str(ndays)])] == True,1,0),index=data.index,columns={''.join(['COUNT_BLOWEST_' + str(ndays)])})
	cb = ct.rolling(center=False, window=pdays).sum() #.name('COUNT_BLOWEST_' + str(ndays))
	data = data.join(cb)
	data = data.fillna(value=0)
	return data

	cb = pd.rolling_apply(arg=data, window=ndays, func=data['BLOWEST_' + str(ndays)][data['BLOWEST_' + str(ndays)] == True].count())


#ANN_LO_BY_DEFLAT
def ANNLOW(data, ndays=250):
	l = pd.Series(pd.rolling_min(data['Low'],ndays, min_periods=1), name='Annual_Low_250_Days')
	data = data.join(l)
	return data

def ANNLOWDEFLAT(data, ndays=250):
	ald = pd.Series((data['Annual_Low_250_Days'].shift(+1) > data['Annual_Low_250_Days']) & (data['Annual_Low_250_Days'] == data['Low']), name='ANN_LO_BY_DEFLAT')
	data = data.join(ald)
	return data	

def ANNLOWATTRIT(data, ndays=250):
	alt = pd.Series((data['Annual_Low_250_Days'].shift(+1) < data['Annual_Low_250_Days']) & (data['Annual_Low_250_Days'] != data['Low']), name='ANN_LO_BY_DEFLAT')
	data = data.join(alt)
	return data
	
#OC - Commodity Channel Indicator?

#RSI
def RSI(data, ndays):
	close = data['Close']
	delta = close.diff()
	delta = delta.fillna(method= 'bfill')
	up, down = delta.copy(), delta.copy()
	up[up < 0] = 0
	down[down > 0] = 0
	roll_up1 = pd.Series(up).ewm(ignore_na=False,adjust=True,min_periods=0,com=ndays).mean()
	roll_down1 = pd.Series(down.abs()).ewm(ignore_na=False,adjust=True,min_periods=0,com=ndays).mean()
	RS1 = roll_up1 / roll_down1
	RSI1 = 100.0 - (100.0 / (1.0 + RS1))
	RSI = pd.Series(RSI1, name='RSI_' + str(ndays))
	data = data.join(RSI)
	return data
	
#Extras
# Ease of Movement 
def EVM(data, ndays): 
	dm = ((data['High'] + data['Low'])/2) - ((data['High'].shift(1) + data['Low'].shift(1))/2)
	br = (data['Volume'] / 100000000) / ((data['High'] - data['Low']))
	EVM = dm / br 
	EVM_MA = pd.Series(pd.rolling_mean(EVM, ndays), name = 'EVM') 
	data = data.join(EVM_MA) 
	return data 
	
#Measures	
def directionality(data,ndays):
	nrange = (np.arange(int(ndays/5))+1)*5
	for n in nrange:
		high = pd.Series(data=data['High'].shift(int(n)*-1).rolling(window=n).max(),index=data.index)
		low = pd.Series(data=data['Low'].shift(int(n)*-1).rolling(window=n).min(),index=data.index)
		high = pd.Series(data=((high-data['Typical'])/data['Typical']),index=data.index)
		low = pd.Series(data=((low-data['Typical'])/data['Typical']),index=data.index)	
		dirat = pd.Series(data=high-low,index=data.index,name=str(n)+'_Day_Directionality(%Up/%Down)')
		data = data.join(dirat)
	return data

def momentum_peak(data,ndays): #ndays must be multiple of 5
	nrange = (np.arange(int(ndays/5))+1)*5
	for n in nrange:
		high = pd.Series(data=(((data['High'].shift(n*-1).rolling(window=n).max()-data['Typical'])/data['Typical'])/n), index=data.index)
		low = pd.Series(data=(((data['Low'].shift(n*-1).rolling(window=n).min()-data['Typical'])/data['Typical'])/n), index=data.index)
		peak = pd.Series(data=(np.where(high>abs(low),high,low)*100,index=data.index,name=str(n)+'_Day_Peak_Momentum(%/Day)')#reformat for positive bias
		data = data.join(peak)
	return data
	
def volatility_stdev(data,ndays):
	nrange = (np.arange(int(ndays/5))+1)*5
	merge = pd.concat([data['High'],data['Low']],axis=0).sort_index(axis=0)
	for n in nrange:
		merger = merge.shift((int(n)*-2)-2)
		volatility = pd.Series(data=(merge.rolling(window=(int(n)*2)).apply(np.std)),name=str(n)+'_Day_St.Dev.')
		volatility = volatility[~volatility.index.duplicated()]
		data = data.join(volatility)
	return data
	
			
def candlesticker(data):
	new_data = pd.DataFrame()			 
	base = data['Close'].shift(-1)
	op = pd.Series(data=data['Open']/base,name='Candle_Open')
	new_data = new_data.join(op)			 
	hi = pd.Series(data=data['High']/base,name='Candle_High')
	new_data = new_data.join(hi)				 
	lo = pd.Series(data=data['Low']/base,name='Candle_Low')
	new_data = new_data.join(lo)				 
	cl = pd.Series(data=data['Close']/base,name='Candle_Close')
	new_data = new_data.join(cl)	
	return new_data
	
def normalizer_setwidth(data):
	new_data = pd.DataFrame()			 
	for col in data.columns:		 
		mx = data[col].max()
		mn = data[col].min()
		mx = mx + ((mx-mn)/100)
		mn = mn - ((mx-mn)/100)
		normalized = pd.Series(data=((data[col]-mn)/(mx-mn),name=col+'_Norm')
		new_data = new_data.join(normalized)
	return new_data			       

def normalizer_bool(data):
	new_data = pd.DataFrame()			 
	for col in data.columns:
		normalized = pd.Series(data=np.where(data[col]==True,.99,.01),name=col+'_Norm')
		new_data = new_data.join(normalized)		       
	return new_data

def normalizer_centered(data):
	new_data = pd.DataFrame(data=data/2)
	return new_data			       
				       

current_time = dt.datetime.now()
last_friday = (current_time.date()
	- dt.timedelta(days=current_time.weekday())
	+ dt.timedelta(days=4, weeks=-1))
strt = pd.to_datetime('1980-01-02', infer_datetime_format=True)

	
symb = 'SPY'
	#get stock data
	 #symbs = ('VXX','XIV','SPY','AAPL','QQQ','IWM','X','NFLX','AMZN','GLD','GDX','C')

stk = web.DataReader(symb, 'yahoo', strt, end)
dctargetdir = 'C:\\Users\\asus\\Dropbox\\Outlines\\MTAUTO-PYTHON\\Back-Tester\\Data_&_Calcs\\'
os.makedirs(dctargetdir, exist_ok=True)	
stk = TYP(stk)
stk = day_2_ratios(stk)	
stk = volatility_stdev(stk,20)
stk = momentum_peak(stk,20)
stk = directionality(stk,20)
stk = BBANDS(stk, 20, 2)
stk = BBAND_RANK(stk)
stk = BB_PINCH_XPAND(stk)
stk = FSTOCH(stk, ndays=14)
stk = SLSTOCH(data=stk,ndays=3)
stk = STOCHDIR(stk)
stk = FAUXSTO(stk)
stk = STOCHABSOLUP(stk)
stk = K_UP_DOWN(stk)
stk = LLOWEST(stk, 20)
stk = HLOWEST(stk, 20)
stk = BLOWEST(stk, 20)
stk = HLLOWESTIN(stk, 20)
stk = COUNT_BLOWEST(stk,20,20)
stk = LLOWEST(stk, 5)
stk = HLOWEST(stk, 5)
stk = BLOWEST(stk, 5)
stk = HLLOWESTIN(stk, 5)
stk = COUNT_BLOWEST(stk,5,5)
stk = ANNLOW(stk)
stk = ANNLOWDEFLAT(stk)
stk = CCI(stk, 14)
stk = CCI(stk, 40)
stk = CCI(stk, 89)
stk = CCI_DIVERG_T(stk,14)
stk = CCI_DIVERG_F(stk,14)
stk = CCI_DIVERG_T(stk,40)
stk = CCI_DIVERG_F(stk,40)
stk = CCI_DIVERG_S(stk,40)
stk = anchored_divergence(data=stk,col='CCI_40',r=30,diff=40)
stk = RSI(stk,20)
stk = RSI(stk,40)
stk = RSI(stk,89)
stk = anchored_divergence(data=stk,col='RSI_40',r=30,diff=40)
stk = SMAs(stk,'Typical')
stk = EMAs(stk,'Typical')
stk = MACD(data=stk,nday1=13,nday2=27,sign=8)
stk = anchored_divergence(data=stk,col='MACD_13-27',r=20,diff=.1)
stk = SMA_slope(data=stk,sma='SMA_5',ndays=3)

stk.to_csv('C:\\Users\\asus\\Dropbox\\Outlines\\MTAUTO-PYTHON\\MT-OPT\\SPY-DC.csv')

stk = pd.read_csv('C:\\Users\\asus\\Dropbox\\Outlines\\MTAUTO-PYTHON\\MT-OPT\\SPY-DC.csv','rb',delimiter=',')
	

dccolumns =			["Open",
					"High",
					"Low",
					"Close",
					"Volume",
					"Adj Close",
					"Typical",
					"D2Lo/D2Op",
					"D2Hi/D2Op",
					"D3Lo/D2Op",
					"D3Hi/D2Op",
					"D4Lo/D2Op",
					"D4Hi/D2Op",
					"D5Lo/D2Op",
					"D5Hi/D2Op",
					"D6Lo/D2Op",
					"D6Hi/D2Op",
					"D7Lo/D2Op",
					"D7Hi/D2Op",
					"D8Lo/D2Op",
					"D8Hi/D2Op",
					"D9Lo/D2Op",
					"D9Hi/D2Op",
					"D10Lo/D2Op",
					"D10Hi/D2Op",
					"D11Lo/D2Op",
					"D11Hi/D2Op",
					"D12Lo/D2Op",
					"D12Hi/D2Op",
					"D13Lo/D2Op",
					"D13Hi/D2Op",
					"D14Lo/D2Op",
					"D14Hi/D2Op",
					"D15Lo/D2Op",
					"D15Hi/D2Op",
					"D16Lo/D2Op",
					"D16Hi/D2Op",
					"D17Lo/D2Op",
					"D17Hi/D2Op",
					"D18Lo/D2Op",
					"D18Hi/D2Op",
					"D19Lo/D2Op",
					"D19Hi/D2Op",
					"D20Lo/D2Op",
					"D20Hi/D2Op",
					"D21Lo/D2Op",
					"D21Hi/D2Op",
					"Per1-2Hi/D2Op",
					"Per3-4Hi/D2Op",
					"Per1-2Lo/D2Op",
					"Per3-4Lo/D2Op",
					"Per1Lo/D2Op",
					"Per1Hi/D2Op",
					"Per1Cl/D2Op",
					"Per2Lo/D2Op",
					"Per2Hi/D2Op",
					"Per2Cl/D2Op",
					"Per3Lo/D2Op",
					"Per3Hi/D2Op",
					"Per3Cl/D2Op",
					"Per4Lo/D2Op",
					"Per4Hi/D2Op",
					"Per4Cl/D2Op",
					"Per5Lo/D2Op",
					"Per5Hi/D2Op",
					"Per5cl/D2Op",
					"Trade_Hi",
					"5_Day_St.Dev.",
					"10_Day_St.Dev.",
					"15_Day_St.Dev.",
					"20_Day_St.Dev.",
					"5_Day_Peak_Momentum",
					"10_Day_Peak_Momentum",
					"15_Day_Peak_Momentum",
					"20_Day_Peak_Momentum",
					"1_Day_Directionality",
					"2_Day_Directionality",
					"3_Day_Directionality",
					"4_Day_Directionality",
					"5_Day_Directionality",
					"6_Day_Directionality",
					"7_Day_Directionality",
					"8_Day_Directionality",
					"9_Day_Directionality",
					"10_Day_Directionality",
					"11_Day_Directionality",
					"12_Day_Directionality",
					"13_Day_Directionality",
					"14_Day_Directionality",
					"15_Day_Directionality",
					"16_Day_Directionality",
					"17_Day_Directionality",
					"18_Day_Directionality",
					"19_Day_Directionality",
					"20_Day_Directionality",
					"Upper_BollingerBand",
					"Lower_BollingerBand",
					"BBand_Rank",
					"BBANDS_PINCH",
					"BBANDS_XPAND",
					"%K",
					"14_Min_Low",
					"14_Max_Hi",
					"%D",
					"D1_DIR",
					"D1_FAUXSTO",
					"ABSOL_UP_D1_STO",
					"%K_3PER_SLP",
					"FIRST_UP_D1_%K_3PER_SLP",
					"FIRST_DOWN_D1_%K_3PER_SLP",
					"Lowest_Low_in_20",
					"Lowest_Hi_in_20",
					"BLOWEST_20",
					"D1_H_LO_LOWEST_IN_20",
					"COUNT_BLOWEST_20",
					"Lowest_Low_in_5",
					"Lowest_Hi_in_5",
					"BLOWEST_5",
					"D1_H_LO_LOWEST_IN_5",
					"COUNT_BLOWEST_5",
					"Annual_Low_250_Days",
					"ANN_LO_BY_DEFLAT",
					"CCI_14",
					"CCI_40",
					"CCI_89",
					"CCI_14_UP_DIVERGENCE_3",
					"CCI_14_DOWN_DIVERGENCE_3",
					"CCI_14_UP_DIVERGENCE_5",
					"CCI_14_DOWN_DIVERGENCE_5",
					"CCI_40_UP_DIVERGENCE_3",
					"CCI_40_DOWN_DIVERGENCE_3",
					"CCI_40_UP_DIVERGENCE_5",
					"CCI_40_DOWN_DIVERGENCE_5",
					"CCI_40_UP_DIVERGENCE_7",
					"CCI_40_DOWN_DIVERGENCE_7",
					"CCI_40_Anchors",
					"CCI_40_Anchor_Indices",
					"CCI_40_Anchor_Values",
					"CCI_40_Typ_Values",
					"CCI_40_Anchored_Slopes",
					"CCI_40_Typ_Slopes",
					"CCI_40_Anchored_Divergence",
					"CCI_40_Anchored_Divergence_3p_Slope",
					"CCI_40_Divergence_2ndDiv",
					"RSI_20",
					"RSI_40",
					"RSI_89",
					"RSI_40_Anchors",
					"RSI_40_Anchor_Indices",
					"RSI_40_Anchor_Values",
					"RSI_40_Typ_Values",
					"RSI_40_Anchored_Slopes",
					"RSI_40_Typ_Slopes",
					"RSI_40_Anchored_Divergence",
					"RSI_40_Anchored_Divergence_3p_Slope",
					"RSI_40_Divergence_2ndDiv",
					"SMA_5",
					"SMA_10",
					"SMA_21",
					"SMA_34",
					"SMA_55",
					"SMA_89",
					"SMA_144",
					"SMA_233",
					"SMA_10_Over_SMA_5",
					"SMA_21_Over_SMA_5",
					"SMA_34_Over_SMA_5",
					"SMA_55_Over_SMA_5",
					"SMA_89_Over_SMA_5",
					"SMA_144_Over_SMA_5",
					"SMA_233_Over_SMA_5",
					"SMA_5_X-Over_SMA_10",
					"SMA_5_X-Over_SMA_21",
					"SMA_5_X-Over_SMA_34",
					"SMA_5_X-Over_SMA_55",
					"SMA_5_X-Over_SMA_89",
					"SMA_5_X-Over_SMA_144",
					"SMA_5_X-Over_SMA_233",
					"SMA_10_X-Over_SMA_5",
					"SMA_10_X-Over_SMA_21",
					"SMA_10_X-Over_SMA_34",
					"SMA_10_X-Over_SMA_55",
					"SMA_10_X-Over_SMA_89",
					"SMA_10_X-Over_SMA_144",
					"SMA_10_X-Over_SMA_233",
					"SMA_21_X-Over_SMA_5",
					"SMA_21_X-Over_SMA_10",
					"SMA_21_X-Over_SMA_34",
					"SMA_21_X-Over_SMA_55",
					"SMA_21_X-Over_SMA_89",
					"SMA_21_X-Over_SMA_144",
					"SMA_21_X-Over_SMA_233",
					"SMA_34_X-Over_SMA_5",
					"SMA_34_X-Over_SMA_10",
					"SMA_34_X-Over_SMA_21",
					"SMA_34_X-Over_SMA_55",
					"SMA_34_X-Over_SMA_89",
					"SMA_34_X-Over_SMA_144",
					"SMA_34_X-Over_SMA_233",
					"SMA_55_X-Over_SMA_5",
					"SMA_55_X-Over_SMA_10",
					"SMA_55_X-Over_SMA_21",
					"SMA_55_X-Over_SMA_34",
					"SMA_55_X-Over_SMA_89",
					"SMA_55_X-Over_SMA_144",
					"SMA_55_X-Over_SMA_233",
					"SMA_89_X-Over_SMA_5",
					"SMA_89_X-Over_SMA_10",
					"SMA_89_X-Over_SMA_21",
					"SMA_89_X-Over_SMA_34",
					"SMA_89_X-Over_SMA_55",
					"SMA_89_X-Over_SMA_144",
					"SMA_89_X-Over_SMA_233",
					"SMA_144_X-Over_SMA_5",
					"SMA_144_X-Over_SMA_10",
					"SMA_144_X-Over_SMA_21",
					"SMA_144_X-Over_SMA_34",
					"SMA_144_X-Over_SMA_55",
					"SMA_144_X-Over_SMA_89",
					"SMA_144_X-Over_SMA_233",
					"SMA_233_X-Over_SMA_5",
					"SMA_233_X-Over_SMA_10",
					"SMA_233_X-Over_SMA_21",
					"SMA_233_X-Over_SMA_34",
					"SMA_233_X-Over_SMA_55",
					"SMA_233_X-Over_SMA_89",
					"SMA_233_X-Over_SMA_144",
					"EMA_5",
					"EMA_10",
					"EMA_21",
					"EMA_34",
					"EMA_55",
					"EMA_89",
					"EMA_144",
					"EMA_233",
					"EMA_10_Over_EMA_5",
					"EMA_21_Over_EMA_5",
					"EMA_34_Over_EMA_5",
					"EMA_55_Over_EMA_5",
					"EMA_89_Over_EMA_5",
					"EMA_144_Over_EMA_5",
					"EMA_233_Over_EMA_5",
					"EMA_5_X-Over_EMA_10",
					"EMA_5_X-Over_EMA_21",
					"EMA_5_X-Over_EMA_34",
					"EMA_5_X-Over_EMA_55",
					"EMA_5_X-Over_EMA_89",
					"EMA_5_X-Over_EMA_144",
					"EMA_5_X-Over_EMA_233",
					"EMA_10_X-Over_EMA_5",
					"EMA_10_X-Over_EMA_21",
					"EMA_10_X-Over_EMA_34",
					"EMA_10_X-Over_EMA_55",
					"EMA_10_X-Over_EMA_89",
					"EMA_10_X-Over_EMA_144",
					"EMA_10_X-Over_EMA_233",
					"EMA_21_X-Over_EMA_5",
					"EMA_21_X-Over_EMA_10",
					"EMA_21_X-Over_EMA_34",
					"EMA_21_X-Over_EMA_55",
					"EMA_21_X-Over_EMA_89",
					"EMA_21_X-Over_EMA_144",
					"EMA_21_X-Over_EMA_233",
					"EMA_34_X-Over_EMA_5",
					"EMA_34_X-Over_EMA_10",
					"EMA_34_X-Over_EMA_21",
					"EMA_34_X-Over_EMA_55",
					"EMA_34_X-Over_EMA_89",
					"EMA_34_X-Over_EMA_144",
					"EMA_34_X-Over_EMA_233",
					"EMA_55_X-Over_EMA_5",
					"EMA_55_X-Over_EMA_10",
					"EMA_55_X-Over_EMA_21",
					"EMA_55_X-Over_EMA_34",
					"EMA_55_X-Over_EMA_89",
					"EMA_55_X-Over_EMA_144",
					"EMA_55_X-Over_EMA_233",
					"EMA_89_X-Over_EMA_5",
					"EMA_89_X-Over_EMA_10",
					"EMA_89_X-Over_EMA_21",
					"EMA_89_X-Over_EMA_34",
					"EMA_89_X-Over_EMA_55",
					"EMA_89_X-Over_EMA_144",
					"EMA_89_X-Over_EMA_233",
					"EMA_144_X-Over_EMA_5",
					"EMA_144_X-Over_EMA_10",
					"EMA_144_X-Over_EMA_21",
					"EMA_144_X-Over_EMA_34",
					"EMA_144_X-Over_EMA_55",
					"EMA_144_X-Over_EMA_89",
					"EMA_144_X-Over_EMA_233",
					"EMA_233_X-Over_EMA_5",
					"EMA_233_X-Over_EMA_10",
					"EMA_233_X-Over_EMA_21",
					"EMA_233_X-Over_EMA_34",
					"EMA_233_X-Over_EMA_55",
					"EMA_233_X-Over_EMA_89",
					"EMA_233_X-Over_EMA_144",
					"EMA_13",
					"EMA_27",
					"MACD_13-27",
					"Signal_EMA_8",
					"MACD_13-27_Anchors",
					"MACD_13-27_Anchor_Indices",
					"MACD_13-27_Anchor_Values",
					"MACD_13-27_Typ_Values",
					"MACD_13-27_Anchored_Slopes",
					"MACD_13-27_Typ_Slopes",
					"MACD_13-27_Anchored_Divergence",
					"MACD_13-27_Anchored_Divergence_3p_Slope",
					"MACD_13-27_Divergence_2ndDiv",
					"SMA_5_3_Slope"]
			

criteria = 			["Per1-2Hi/D2Op",
					"Per3-4Hi/D2Op",
					"Per1-2Lo/D2Op",
					"Per3-4Lo/D2Op",
					"Per1Lo/D2Op",
					"Per1Hi/D2Op",
					"Per1Cl/D2Op",
					"Per2Lo/D2Op",
					"Per2Hi/D2Op",
					"Per2Cl/D2Op",
					"Per3Lo/D2Op",
					"Per3Hi/D2Op",
					"Per3Cl/D2Op",
					"Per4Lo/D2Op",
					"Per4Hi/D2Op",
					"Per4Cl/D2Op",
					"Per5Lo/D2Op",
					"Per5Hi/D2Op",
					"Per5cl/D2Op",
					"Trade_Hi",
					"5_Day_St.Dev.",
					"10_Day_St.Dev.",
					"15_Day_St.Dev.",
					"20_Day_St.Dev.",
					"5_Day_Peak_Momentum",
					"10_Day_Peak_Momentum",
					"15_Day_Peak_Momentum",
					"20_Day_Peak_Momentum",
					"1_Day_Directionality",
					"2_Day_Directionality",
					"3_Day_Directionality",
					"4_Day_Directionality",
					"5_Day_Directionality",
					"6_Day_Directionality",
					"7_Day_Directionality",
					"8_Day_Directionality",
					"9_Day_Directionality",
					"10_Day_Directionality",
					"11_Day_Directionality",
					"12_Day_Directionality",
					"13_Day_Directionality",
					"14_Day_Directionality",
					"15_Day_Directionality",
					"16_Day_Directionality",
					"17_Day_Directionality",
					"18_Day_Directionality",
					"19_Day_Directionality",
					"20_Day_Directionality"]
					
all_indicators =	["Upper_BollingerBand",
					"Lower_BollingerBand",
					"BBand_Rank",
					"BBANDS_PINCH",
					"BBANDS_XPAND",
					"%K",
					"14_Min_Low",
					"14_Max_Hi",
					"%D",
					"D1_DIR",
					"D1_FAUXSTO",
					"ABSOL_UP_D1_STO",
					"%K_3PER_SLP",
					"FIRST_UP_D1_%K_3PER_SLP",
					"FIRST_DOWN_D1_%K_3PER_SLP",
					"Lowest_Low_in_20",
					"Lowest_Hi_in_20",
					"BLOWEST_20",
					"D1_H_LO_LOWEST_IN_20",
					"COUNT_BLOWEST_20",
					"Lowest_Low_in_5",
					"Lowest_Hi_in_5",
					"BLOWEST_5",
					"D1_H_LO_LOWEST_IN_5",
					"COUNT_BLOWEST_5",
					"Annual_Low_250_Days",
					"ANN_LO_BY_DEFLAT",
					"CCI_14",
					"CCI_40",
					"CCI_89",
					"CCI_14_UP_DIVERGENCE_3",
					"CCI_14_DOWN_DIVERGENCE_3",
					"CCI_14_UP_DIVERGENCE_5",
					"CCI_14_DOWN_DIVERGENCE_5",
					"CCI_40_UP_DIVERGENCE_3",
					"CCI_40_DOWN_DIVERGENCE_3",
					"CCI_40_UP_DIVERGENCE_5",
					"CCI_40_DOWN_DIVERGENCE_5",
					"CCI_40_UP_DIVERGENCE_7",
					"CCI_40_DOWN_DIVERGENCE_7",
					"CCI_40_Anchors",
					"CCI_40_Anchor_Indices",
					"CCI_40_Anchor_Values",
					"CCI_40_Typ_Values",
					"CCI_40_Anchored_Slopes",
					"CCI_40_Typ_Slopes",
					"CCI_40_Anchored_Divergence",
					"CCI_40_Anchored_Divergence_3p_Slope",
					"CCI_40_Divergence_2ndDiv",
					"RSI_20",
					"RSI_40",
					"RSI_89",
					"RSI_40_Anchors",
					"RSI_40_Anchor_Indices",
					"RSI_40_Anchor_Values",
					"RSI_40_Typ_Values",
					"RSI_40_Anchored_Slopes",
					"RSI_40_Typ_Slopes",
					"RSI_40_Anchored_Divergence",
					"RSI_40_Anchored_Divergence_3p_Slope",
					"RSI_40_Divergence_2ndDiv",
					"SMA_5",
					"SMA_10",
					"SMA_21",
					"SMA_34",
					"SMA_55",
					"SMA_89",
					"SMA_144",
					"SMA_233",
					"SMA_10_Over_SMA_5",
					"SMA_21_Over_SMA_5",
					"SMA_34_Over_SMA_5",
					"SMA_55_Over_SMA_5",
					"SMA_89_Over_SMA_5",
					"SMA_144_Over_SMA_5",
					"SMA_233_Over_SMA_5",
					"SMA_5_X-Over_SMA_10",
					"SMA_5_X-Over_SMA_21",
					"SMA_5_X-Over_SMA_34",
					"SMA_5_X-Over_SMA_55",
					"SMA_5_X-Over_SMA_89",
					"SMA_5_X-Over_SMA_144",
					"SMA_5_X-Over_SMA_233",
					"SMA_10_X-Over_SMA_5",
					"SMA_10_X-Over_SMA_21",
					"SMA_10_X-Over_SMA_34",
					"SMA_10_X-Over_SMA_55",
					"SMA_10_X-Over_SMA_89",
					"SMA_10_X-Over_SMA_144",
					"SMA_10_X-Over_SMA_233",
					"SMA_21_X-Over_SMA_5",
					"SMA_21_X-Over_SMA_10",
					"SMA_21_X-Over_SMA_34",
					"SMA_21_X-Over_SMA_55",
					"SMA_21_X-Over_SMA_89",
					"SMA_21_X-Over_SMA_144",
					"SMA_21_X-Over_SMA_233",
					"SMA_34_X-Over_SMA_5",
					"SMA_34_X-Over_SMA_10",
					"SMA_34_X-Over_SMA_21",
					"SMA_34_X-Over_SMA_55",
					"SMA_34_X-Over_SMA_89",
					"SMA_34_X-Over_SMA_144",
					"SMA_34_X-Over_SMA_233",
					"SMA_55_X-Over_SMA_5",
					"SMA_55_X-Over_SMA_10",
					"SMA_55_X-Over_SMA_21",
					"SMA_55_X-Over_SMA_34",
					"SMA_55_X-Over_SMA_89",
					"SMA_55_X-Over_SMA_144",
					"SMA_55_X-Over_SMA_233",
					"SMA_89_X-Over_SMA_5",
					"SMA_89_X-Over_SMA_10",
					"SMA_89_X-Over_SMA_21",
					"SMA_89_X-Over_SMA_34",
					"SMA_89_X-Over_SMA_55",
					"SMA_89_X-Over_SMA_144",
					"SMA_89_X-Over_SMA_233",
					"SMA_144_X-Over_SMA_5",
					"SMA_144_X-Over_SMA_10",
					"SMA_144_X-Over_SMA_21",
					"SMA_144_X-Over_SMA_34",
					"SMA_144_X-Over_SMA_55",
					"SMA_144_X-Over_SMA_89",
					"SMA_144_X-Over_SMA_233",
					"SMA_233_X-Over_SMA_5",
					"SMA_233_X-Over_SMA_10",
					"SMA_233_X-Over_SMA_21",
					"SMA_233_X-Over_SMA_34",
					"SMA_233_X-Over_SMA_55",
					"SMA_233_X-Over_SMA_89",
					"SMA_233_X-Over_SMA_144",
					"EMA_5",
					"EMA_10",
					"EMA_21",
					"EMA_34",
					"EMA_55",
					"EMA_89",
					"EMA_144",
					"EMA_233",
					"EMA_10_Over_EMA_5",
					"EMA_21_Over_EMA_5",
					"EMA_34_Over_EMA_5",
					"EMA_55_Over_EMA_5",
					"EMA_89_Over_EMA_5",
					"EMA_144_Over_EMA_5",
					"EMA_233_Over_EMA_5",
					"EMA_5_X-Over_EMA_10",
					"EMA_5_X-Over_EMA_21",
					"EMA_5_X-Over_EMA_34",
					"EMA_5_X-Over_EMA_55",
					"EMA_5_X-Over_EMA_89",
					"EMA_5_X-Over_EMA_144",
					"EMA_5_X-Over_EMA_233",
					"EMA_10_X-Over_EMA_5",
					"EMA_10_X-Over_EMA_21",
					"EMA_10_X-Over_EMA_34",
					"EMA_10_X-Over_EMA_55",
					"EMA_10_X-Over_EMA_89",
					"EMA_10_X-Over_EMA_144",
					"EMA_10_X-Over_EMA_233",
					"EMA_21_X-Over_EMA_5",
					"EMA_21_X-Over_EMA_10",
					"EMA_21_X-Over_EMA_34",
					"EMA_21_X-Over_EMA_55",
					"EMA_21_X-Over_EMA_89",
					"EMA_21_X-Over_EMA_144",
					"EMA_21_X-Over_EMA_233",
					"EMA_34_X-Over_EMA_5",
					"EMA_34_X-Over_EMA_10",
					"EMA_34_X-Over_EMA_21",
					"EMA_34_X-Over_EMA_55",
					"EMA_34_X-Over_EMA_89",
					"EMA_34_X-Over_EMA_144",
					"EMA_34_X-Over_EMA_233",
					"EMA_55_X-Over_EMA_5",
					"EMA_55_X-Over_EMA_10",
					"EMA_55_X-Over_EMA_21",
					"EMA_55_X-Over_EMA_34",
					"EMA_55_X-Over_EMA_89",
					"EMA_55_X-Over_EMA_144",
					"EMA_55_X-Over_EMA_233",
					"EMA_89_X-Over_EMA_5",
					"EMA_89_X-Over_EMA_10",
					"EMA_89_X-Over_EMA_21",
					"EMA_89_X-Over_EMA_34",
					"EMA_89_X-Over_EMA_55",
					"EMA_89_X-Over_EMA_144",
					"EMA_89_X-Over_EMA_233",
					"EMA_144_X-Over_EMA_5",
					"EMA_144_X-Over_EMA_10",
					"EMA_144_X-Over_EMA_21",
					"EMA_144_X-Over_EMA_34",
					"EMA_144_X-Over_EMA_55",
					"EMA_144_X-Over_EMA_89",
					"EMA_144_X-Over_EMA_233",
					"EMA_233_X-Over_EMA_5",
					"EMA_233_X-Over_EMA_10",
					"EMA_233_X-Over_EMA_21",
					"EMA_233_X-Over_EMA_34",
					"EMA_233_X-Over_EMA_55",
					"EMA_233_X-Over_EMA_89",
					"EMA_233_X-Over_EMA_144",
					"EMA_13",
					"EMA_27",
					"MACD_13-27",
					"Signal_EMA_8",
					"MACD_13-27_Anchors",
					"MACD_13-27_Anchor_Indices",
					"MACD_13-27_Anchor_Values",
					"MACD_13-27_Typ_Values",
					"MACD_13-27_Anchored_Slopes",
					"MACD_13-27_Typ_Slopes",
					"MACD_13-27_Anchored_Divergence",
					"MACD_13-27_Anchored_Divergence_3p_Slope",
					"MACD_13-27_Divergence_2ndDiv",
					"SMA_5_3_Slope"]
					
balance_indicators =["BBand_Rank",
					"BBANDS_PINCH",
					"BBANDS_XPAND",
					"%K",
					"%D",
					"D1_DIR_Bool",
					"D1_FAUXSTO",
					"ABSOL_UP_D1_STO",
					"%K_3PER_SLP",
					"FIRST_UP_D1_%K_3PER_SLP",
					"FIRST_DOWN_D1_%K_3PER_SLP",
					"Lowest_Low_in_20",
					"Lowest_Hi_in_20",
					"BLOWEST_20",
					"D1_H_LO_LOWEST_IN_20",
					"COUNT_BLOWEST_20",
					"Lowest_Low_in_5",
					"Lowest_Hi_in_5",
					"BLOWEST_5",
					"D1_H_LO_LOWEST_IN_5",
					"COUNT_BLOWEST_5",
					"Annual_Low_250_Days",
					"ANN_LO_BY_DEFLAT",
					"CCI_14",
					"CCI_40",
					"CCI_89",
					"CCI_14_UP_DIVERGENCE_3",
					"CCI_14_DOWN_DIVERGENCE_3",
					"CCI_14_UP_DIVERGENCE_5",
					"CCI_14_DOWN_DIVERGENCE_5",
					"CCI_40_UP_DIVERGENCE_3",
					"CCI_40_DOWN_DIVERGENCE_3",
					"CCI_40_UP_DIVERGENCE_5",
					"CCI_40_DOWN_DIVERGENCE_5",
					"CCI_40_UP_DIVERGENCE_7",
					"CCI_40_DOWN_DIVERGENCE_7",
					"CCI_40_Anchored_Divergence",
					"CCI_40_Divergence_2ndDiv",
					"RSI_20",
					"RSI_40",
					"RSI_89",
					"RSI_40_Anchored_Divergence",
					"RSI_40_Divergence_2ndDiv",
					"SMA_10_Over_SMA_5",
					"SMA_21_Over_SMA_5",
					"SMA_34_Over_SMA_5",
					"SMA_55_Over_SMA_5",
					"SMA_89_Over_SMA_5",
					"SMA_144_Over_SMA_5",
					"SMA_233_Over_SMA_5",
					"SMA_5_X-Over_SMA_10",
					"SMA_5_X-Over_SMA_21",
					"SMA_5_X-Over_SMA_34",
					"SMA_5_X-Over_SMA_55",
					"SMA_5_X-Over_SMA_89",
					"SMA_5_X-Over_SMA_144",
					"SMA_5_X-Over_SMA_233",
					"SMA_10_X-Over_SMA_5",
					"SMA_10_X-Over_SMA_21",
					"SMA_10_X-Over_SMA_34",
					"SMA_10_X-Over_SMA_55",
					"SMA_10_X-Over_SMA_89",
					"SMA_10_X-Over_SMA_144",
					"SMA_10_X-Over_SMA_233",
					"SMA_21_X-Over_SMA_5",
					"SMA_21_X-Over_SMA_10",
					"SMA_21_X-Over_SMA_34",
					"SMA_21_X-Over_SMA_55",
					"SMA_21_X-Over_SMA_89",
					"SMA_21_X-Over_SMA_144",
					"SMA_21_X-Over_SMA_233",
					"SMA_34_X-Over_SMA_5",
					"SMA_34_X-Over_SMA_10",
					"SMA_34_X-Over_SMA_21",
					"SMA_34_X-Over_SMA_55",
					"SMA_34_X-Over_SMA_89",
					"SMA_34_X-Over_SMA_144",
					"SMA_34_X-Over_SMA_233",
					"SMA_55_X-Over_SMA_5",
					"SMA_55_X-Over_SMA_10",
					"SMA_55_X-Over_SMA_21",
					"SMA_55_X-Over_SMA_34",
					"SMA_55_X-Over_SMA_89",
					"SMA_55_X-Over_SMA_144",
					"SMA_55_X-Over_SMA_233",
					"SMA_89_X-Over_SMA_5",
					"SMA_89_X-Over_SMA_10",
					"SMA_89_X-Over_SMA_21",
					"SMA_89_X-Over_SMA_34",
					"SMA_89_X-Over_SMA_55",
					"SMA_89_X-Over_SMA_144",
					"SMA_89_X-Over_SMA_233",
					"SMA_144_X-Over_SMA_5",
					"SMA_144_X-Over_SMA_10",
					"SMA_144_X-Over_SMA_21",
					"SMA_144_X-Over_SMA_34",
					"SMA_144_X-Over_SMA_55",
					"SMA_144_X-Over_SMA_89",
					"SMA_144_X-Over_SMA_233",
					"SMA_233_X-Over_SMA_5",
					"SMA_233_X-Over_SMA_10",
					"SMA_233_X-Over_SMA_21",
					"SMA_233_X-Over_SMA_34",
					"SMA_233_X-Over_SMA_55",
					"SMA_233_X-Over_SMA_89",
					"SMA_233_X-Over_SMA_144",
					"MACD_13-27",
					"Signal_EMA_8",
					"MACD_13-27_Anchored_Divergence",
					"MACD_13-27_Divergence_2ndDiv",
					"SMA_5_3_Slope"]


stk = pd.read_csv('C:\\Users\\asus\\Dropbox\\Outlines\\MTAUTO-PYTHON\\MT-OPT\\SPY-DC.csv','rb',delimiter=',')					
type_dict = {}
bin_list = list([0,.1,.2,.3,.4,.5,.6,.7,.8,.9])				
sort = stk
sort.fillna(value=0,inplace=True)
#sort.set_index('Date',inplace=True)					
irange = np.arange(2)+3
final_report = pd.DataFrame()

for r in irange:
	indication_combos = it.combinations(balance_indicators, r)
	for ind_cmb in indication_combos:
		type_list = list()

		for ind in ind_cmb:
			if sort[ind].dtype == 'float64':
				type_dict[ind] = bin_list
				type_list.append(bin_list)
			elif sort[ind].dtype == 'bool':
				type_dict[ind] = [True,False]
				type_list.append([True,False])
		sort_combos	= list(it.product(*type_list))
		sort_report = pd.DataFrame(index=np.arange(len(sort_combos)))
		tx = -1
		for x in sort_combos:			
			tx = tx + 1
			sort_dict = {}
			sort_code = ''
			index_dict = {}
			for i in np.arange(r):
				if sort[ind_cmb[i]].dtype == 'float64':
					if x[i] == 0:
						sort_dict[i] = sort[sort[ind_cmb[i]] <= sort[ind_cmb[i]].quantile(.1)]						
						index_dict[i] = set(sort_dict[i].index)
						sort_code += ''.join([ind_cmb[i],'(',str(x[i]),')'])
					elif x[i] > 0:
						sort_dict[i] = sort[(sort[ind_cmb[i]] > sort[ind_cmb[i]].quantile(x[i])) & (sort[ind_cmb[i]] <= sort[ind_cmb[i]].quantile(x[i]+.1))]
						index_dict[i] = set(sort_dict[i].index)
						sort_code += ''.join([ind_cmb[i],'(',str(x[i]),')'])
				if 	sort[ind_cmb[i]].dtype == 'bool':
					sort_dict[i] = sort[sort[ind_cmb[i]] == x[i]]
					index_dict[i] = set(sort_dict[i].index)
					sort_code += ''.join([ind_cmb[i],'(',str(x[i]),')'])
			
			indx = sort.index
			for j in np.arange(r):
				indx = set(indx).intersection(index_dict[j])
			data = sort[sort.index.isin(indx)]
			sort_report.at[tx,'Sort_Code'] = sort_code
			sort_report.at[tx,'St_Dev_10Day_20th%'] = data['10_Day_St.Dev.'].quantile(.2)
			sort_report.at[tx,'Up_Directionality_10Day_20th%'] = data['10_Day_Directionality(%Up/%Down)'][data['10_Day_Directionality(%Up/%Down)']>0].quantile(.2)
			sort_report.at[tx,'Down_Directionality_10Day_80th%'] = data['10_Day_Directionality(%Up/%Down)'][data['10_Day_Directionality(%Up/%Down)']<0].quantile(.2)
			sort_report.at[tx,'Up_Momentum_10Day_20th%'] = data['10_Day_Peak_Momentum(%/Day)'][data['10_Day_Peak_Momentum(%/Day)']>0].quantile(.2)
			sort_report.at[tx,'Down_Momentum_10Day_80th%'] = data['10_Day_Peak_Momentum(%/Day)'][data['10_Day_Peak_Momentum(%/Day)']<0].quantile(.8)
			sort_report.at[tx,'Data_Set_Length'] = len(data)
		if os.path.isfile('C:\\Users\\asus\\Dropbox\\Outlines\\MTAUTO-PYTHON\\MT-OPT\\Data\\Raw.csv'):
			sort_report.to_csv('C:\\Users\\asus\\Dropbox\\Outlines\\MTAUTO-PYTHON\\MT-OPT\\Data\\Raw.csv',mode='a',header=None)
		else: 
			sort_report.to_csv('C:\\Users\\asus\\Dropbox\\Outlines\\MTAUTO-PYTHON\\MT-OPT\\Data\\Raw.csv',mode='w',header=sort_report.columns)
		sort_report = sort_report[sort_report['Data_Set_Length'] >=15]
		sort_std_selection = sort_report[sort_report['St_Dev_10Day_20th%'] >= sort_report['St_Dev_10Day_20th%'].quantile(.9)]
		sort_dir1_selection = sort_report[sort_report['Up_Directionality_10Day_20th%'] >= sort_report['Up_Directionality_10Day_20th%'].quantile(.9)]
		sort_dir2_selection = sort_report[sort_report['Down_Directionality_10Day_80th%'] >= sort_report['Down_Directionality_10Day_80th%'].quantile(.1)]
		sort_mom1_selection = sort_report[sort_report['Up_Momentum_10Day_20th%'] >= sort_report['Up_Momentum_10Day_20th%'].quantile(.9)]
		sort_mom2_selection = sort_report[sort_report['Down_Momentum_10Day_80th%'] <= sort_report['Down_Momentum_10Day_80th%'].quantile(.1)]
		final_report = pd.concat([sort_std_selection,sort_dir1_selection,sort_dir2_selection,sort_mom1_selection,sort_mom2_selection], axis=0)
		finalpath = "C:\\Users\\asus\\Dropbox\\Outlines\\MTAUTO-PYTHON\\MT-OPT\\SPY-MT_OPT_Report_Summary.csv"
		if os.path.isfile(finalpath):
			sort_report.to_csv(finalpath,mode='a',index=False,header=None)
		else: 
			sort_report.to_csv(finalpath,mode='w',index=False,header=final_report.columns)
		
np.save("C:\\Users\\asus\\Dropbox\\Outlines\\MTAUTO-PYTHON\\MT-OPT\\Type_Dictionary.npy",type_dict)
