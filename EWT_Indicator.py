import pandas as pd
import numpy as np
import os
import datetime as dt
import pandas_datareader.data as web
from yahoo_finance import Share
import scipy as sp
from datetime import datetime
import calendar
import itertools as it
from scipy import signal as sig
from scipy import stats
import matplotlib.pyplot as plt
from collections import Counter
#parameters
time_frames = (900,168,44) #(880,160,20)


base_time = pd.to_datetime('1901-01-01', infer_datetime_format=True)
start = pd.to_datetime('1980-01-02', infer_datetime_format=True)
end =  dt.datetime.today() - dt.timedelta(days=14)
current_time = dt.datetime.now()
last_friday = (current_time.date()
	- dt.timedelta(days=current_time.weekday())
	+ dt.timedelta(days=4))
strt = pd.to_datetime('1980-01-02', infer_datetime_format=True)

dets = web.DataReader('SPY', 'yahoo', start, last_friday)

stk = dets

def SMA(data,column,ndays): 
	s = pd.Series(data[column].rolling(center=False,window=ndays).mean(), name='SMA_' + str(ndays)) 
	data = data.join(s)
	return data
	
def TYP(data):
	typ = pd.Series(((data['High'] + data['Low'] + data['Close']) / 3), name='Typical')
	data = data.join(typ)
	data = data.fillna(value=0)
	return data
	
def three_linest(y,x=np.arange(3),deg=1):
	l = np.polyfit(y=y, x=x, deg=1)[0]
	return l
def seven_linest(y,x=np.arange(7),deg=1):
	l = np.polyfit(y=y, x=x, deg=1)[0]
	return l
def rolling_idxmax(data,idx,window):
	idx_max = data.ix[data.index[idx] - dt.timedelta(days=window):data.index[idx]]['High'].idxmax()
	return idx_max

def ema(values):
	window = len(values)
	weights = np.exp(np.linspace(-1., 0., window))
	weights /= weights.sum()
	a =  np.convolve(values, weights, mode='full')[:len(values)]
	a[:window] = a[window]
	return a

def EMA(data, ndays): 
	EMA = pd.Series(pd.ewma(data['Close'], span = ndays, min_periods = ndays - 1, name = 'EMA_' + str(ndays)) )
	return EMA
	
def get_low_idx(data,signal_series,window,ma):
	lows = pd.Series(index=np.arange(len(data)),name=ma+'_' + str(window) + '_Low_IDX')
	for i in signal_series.index:
		if signal_series[i] > 0:
			lows[i] = data.ix[tidx.shift(window)[i]:tidx.index[i]]['Low'].argmin()
	return lows	
	
def get_high_idx(data,signal_series,window,ma):
	highs = pd.Series(index=np.arange(len(data)),name=ma+'_' + str(window) + '_High_IDX')
	for i in signal_series.index:
		if signal_series[i] > 0:
			highs[i] = data.ix[tidx.shift(window)[i]:tidx.index[i]]['Low'].argmax()
	return highs	

	
def SMAs(data,column):
	sma_range = (5,8,13,21,34,55,89,144,233,512)
	sma_combos = it.permutations(sma_range,2)
	degree_dict = {8:8,13:7,21:6,34:5,55:4,89:3,144:2,233:1,512:0}
	for p in sma_range:
		s = pd.Series(data[column].rolling(center=False,window=p).mean(), name='SMA_' + str(p)).rolling(window=3).mean() 
		data = data.join(s)
		if p < 70:
			m = pd.Series(data['SMA_' + str(p)].rolling(center=False,window=3).apply(three_linest), name='SMA_' + str(p) + '_3pSlope')
			data = data.join(m)
		else: 
			m = pd.Series(data['SMA_' + str(p)].rolling(center=False,window=7).apply(seven_linest), name='SMA_' + str(p) + '_3pSlope')
			data = data.join(m)
		hi = pd.Series(data['High'].rolling(center=False,window=p).max())
		lo = pd.Series(data['Low'].rolling(center=False,window=p).min())
		l = pd.Series(data=(np.where(((m>0) & (m.shift(2)<0) & (np.abs(m.shift(2)-m)>.01)),lo,np.nan)), index=np.arange(len(data)), name='SMA_' + str(p) + '_Lows')
		l_idx = get_low_idx(data,l,p,'SMA').fillna(method='ffill')
		l = l.fillna(method='ffill')
		h = pd.Series(data=(np.where(((m<0) & (m.shift(2)>0) & (np.abs(m.shift(2)-m)>.01)),hi,np.nan)), index=np.arange(len(data)),name='SMA_' + str(p) + '_Highs')
		h_idx = get_high_idx(data,h,p,'SMA').fillna(method='ffill')
		h = h.fillna(method='ffill')
		pre_data = pd.concat([idx,h,l,h_idx,l_idx],axis=1).set_index('Date')		
		data = pd.concat([data,pre_data],axis=1)
	return data
	
def EMAs(data,column):
	sma_range = (5,8,13,21,34,55,89,144,233,512)
	sma_combos = it.permutations(sma_range,2)
	degree_dict = {8:8,13:7,21:6,34:5,55:4,89:3,144:2,233:1,512:0}
	for p in sma_range:
		s = pd.Series(pd.ewma(data['Close'], span = p, min_periods = p - 1), name = 'EMA_' + str(p)) 
		data = data.join(s)
		if p < 70:
			m = pd.Series(data['SMA_' + str(p)].rolling(center=False,window=3).apply(three_linest), name='EMA_' + str(p) + '_3pSlope')
			data = data.join(m)
		else: 
			m = pd.Series(data['SMA_' + str(p)].rolling(center=False,window=7).apply(seven_linest), name='EMA_' + str(p) + '_3pSlope')
			data = data.join(m)
		hi = pd.Series(data['High'].rolling(center=False,window=p).max())
		lo = pd.Series(data['Low'].rolling(center=False,window=p).min())
		l = pd.Series(data=(np.where(((m>0) & (m.shift(2)<0) & (np.abs(m.shift(2)-m)>.01)),lo,np.nan)), index=np.arange(len(data)), name='EMA_' + str(p) + '_Lows')
		l_idx = get_low_idx(data,l,p,'EMA').fillna(method='ffill')
		l = l.fillna(method='ffill')
		h = pd.Series(data=(np.where(((m<0) & (m.shift(2)>0) & (np.abs(m.shift(2)-m)>.01)),hi,np.nan)), index=np.arange(len(data)),name='EMA_' + str(p) + '_Highs')
		h_idx = get_high_idx(data,h,p,'EMA').fillna(method='ffill')
		h = h.fillna(method='ffill')
		pre_data = pd.concat([idx,h,l,h_idx,l_idx],axis=1).set_index('Date')		
		data = pd.concat([data,pre_data],axis=1)
	return data
	
idx = pd.Series(data=stk.index,index=np.arange(len(stk)))
tidx = pd.Series(data=stk.index,index=stk.index)

stk = TYP(stk)
stk = SMAs(stk,'Typical')
stk = EMAs(stk,'Typical')
stk.to_csv('C:\\Users\\asus\\Dropbox\\Outlines\\MTAUTO-PYTHON\\Test.csv')

plt.plot(stk.index,stk['Typical'])
for value in stk['EMA_89_Low_IDX'].unique():
	plt.axvline(x=value).set_color('r')
	

for value in stk['SMA_89_Low_IDX'].unique():
	plt.axvline(x=value)
	
plt.show()























stk = stk.fillna(0)
def highs_lows(data):
	for tf in time_frames
		highs = pd.Series(data['Typical'].rolling(window=tf).max(),name='Max_TF_'+str(tf))
		lows = pd.Series(data['Typical'].rolling(window=tf).min(),name='Min_TF_'+str(tf))
		data = pd.concat([data,highs,lows],axis=1)
		his = Counter(highs)
		his = his.most_common()
		high_dict = {}
		for x in hi_range:
			high_dict[''.join([str(x),'_',str(tf)])] = (highs==x).idxmax()
		los = los.tolist()
		low_dict = {}
		for x in lows.mode().index:
			low_dict[''.join(str(x),'_',str(tf))] = lows.index[los.index(lows.mode()[x])]
			
mode_range = [x[1] for x in his if x[1]> (tf*.33)]
hi_range = [x[0] for x in his if x[1]>(tf*.33)]
[value for value in his if any(i in his for i in filter_list)]

plt.plot(stk.index,stk['Typical'])
for key, value in high_dict.items():
	plt.axvline(x=high_dict[key])
plt.show()

		if p == 5:
			hi_index = pd.Series(data=idx[h>0],index=data.index,name=str(p)+'_highs_idx')
			hi_index = hi_index.fillna(method='ffill')
			data = data.join(hi_index)
			lo_index = pd.Series(data=idx[l>0],index=data.index,name=str(p)+'_lows_idx')
			lo_index = lo_index.fillna(method='ffill')
			data = data.join(lo_index)				
			l = l.ffill()
			data = data.join(l)
			h = h.ffill()
			data = data.join(h)
		if p > 5:					
			hi = pd.Series(data.ix[idx[].rolling(center=False,window=p+10).max(),name=str(p)+'_highs')
			hi = hi.fillna(method='ffill')
			lo = pd.Series(data['SMA_5_Lows'].rolling(center=False,window=p+10).min(),name=str(p)+'_lows')
			lo = lo.fillna(method='ffill')
			peak = pd.Series(data=(np.where((h==hi),hi,np.nan)), index=data.index, name=str(degree_dict[p])+'th_Degree_High')
			peak = peak.fillna(method='ffill')
			peak_idx = pd.Series(data=(np.where((h==hi),hi_index,np.datetime64('nat'))), index=data.index, name=str(degree_dict[p])+'th_Degree_High_IDX')
			peak_idx = peak_idx.fillna(method='ffill')
			trough = pd.Series(data=(np.where((l==lo),lo,np.nan)), index=data.index, name=str(degree_dict[p])+'th_Degree_Low')
			trough = trough.fillna(method='ffill')
			trough_idx = pd.Series(data=(np.where((l==lo),lo_index,np.datetime64('nat'))), index=data.index, name=str(degree_dict[p])+'th_Degree_Low_IDX')
			trough_idx = trough_idx.fillna(method='ffill')
			data = data.join(hi)
			data = data.join(peak)
			data = data.join(peak_idx)
			data = data.join(lo)
			data = data.join(trough)
			data = data.join(trough_idx)
