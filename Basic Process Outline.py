
μτ

1. θ
2. φ
3. μ

Π:

1. 1-30D

δια:

1. καθε ε*Σε→
  1. ΣΜ →
    1. καθε ΣΜ→
      1. ψ Ο π
        1. καθε π
          1. θ(π)
          2. φ(π)
          3. α(π)

διθ(ΣΜθφαΜ) κατα κνς → εκθ

θ=(π.τελ.τυπ - π.προ.τυπ)/μηκ(π)
θ(χ) = →

  καθε π ψ θ
  επανα (μετ(θ>0)/μηκ(χ))/(μετ(θ<0)/μηκ(χ))
  ή 
  επανα αθρ(Ο θ)

φ = θ
φ(χ) = →

  καθε π ψ φ
  επανα απλ(Οφ)/μηκ(χ)

α = (μγ(π)-λγ(π))/π.προ.τυπ
α(χ) = →

  καθε π ψ α
  επανα μεσ(Οα)

καθε ε*Σε

  π.χ. 1ε=κ → κ = διθ3(%κ)
  καθε 2ε
    ε1-χ = τωπ(100/(1-20))

διθ(μετοχ.,[κ,2ε1,2ε2,…,2εχ]) = ΣΜ

π = σειρ[1-30]

καθε ημ εν ΣΜ

  στ = επελκ ημ+29
  καθε π εν π
    πδ = επελκ ημ→π
    θ(πδ)
    φ(πδ)
    α(πδ)
    εκθ.προθ(θ,φ,α)
    


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
import sys
import gc
import pickle
import math


########################################################################LISTS & COLUMNS############################################################################################

lcrcolumns = ['Stock_Symbol','Option','Sort','Form_#','Level','Strategy','Strategy_Formula','Buy/Sell','D2Op/D1Cl','#_of_Transactions','#_in_Dataset','D2_Investible_Volume','Calc_Profit%','Win%','Net_Profit/Loss',
			'Highest_Hi','Ratio_Net/Highest_Hi','Hist_Profit/Loss_per_Tx','#_TX>0','%_of_TX>0', '#_OF_"FALSE"_IN_WIN_CALC',
			'Fail%','CALC._PROF/LOSS_ON_Fd_TXS','#_of_Exit_Attempts','%_Successful_Exits','Biggest_Loser','Max_Drawdown(Acct_Min)','Max_%_Drawdown','%_of_TXs_w/_LOSS>50%',
			'Catastrophic_Fail_%_(-80%)','AMT_AT_RISK','Buy_Target_%','Sell_Target_%','Win_Period_10th/90th','Win_Period_20th/80th','Win_Period_40th/60th','Win_Period_50th/50th','Win_Period_60th/40th']

			
optcolumns = ['quote_date','underlying_symbol', 'root', 'expiration', 'strike', 'option_type',
				 'open', 'high', 'low', 'close', 'trade_volume', 'bid_size_1545',
				 'bid_1545', 'ask_size_1545', 'ask_1545', 'underlying_bid_1545',
				 'underlying_ask_1545', 'implied_underlying_price_1545',
				 'active_underlying_price_1545', 'implied_volatility_1545', 'delta_1545',
				 'gamma_1545', 'theta_1545', 'vega_1545', 'rho_1545', 'bid_size_eod',
				 'bid_eod', 'ask_size_eod', 'ask_eod', 'underlying_bid_eod',
				 'underlying_ask_eod', 'vwap', 'open_interest', 'delivery_code']
				 
tnpsummarycolumns = ['#_of_Contracts','D2Op/D1Cl','D2Op','D2Hi','D2Lo','D2Cl','D2Vol','Entry_Target','Sell_Target','Exit_Price','Escape_Price','Entry?','Win?','Successful_Exit?','Escape?',              
					'Profit/Loss', 'Day_Calc_Sell_Reached','Trade_Hi_Over_D2Op_(%)','Day_25%_Acheived_Over_D2Op','Day_50%_Acheived_Over_D2Op','Day_75%_Acheived_Over_D2Op','Day_100%_Acheived_Over_D2Op','Day_25%_Acheived_Over_Calc_Buy','Day_50%_Acheived_Over_Calc_Buy','Day_75%_Acheived_Over_Calc_Buy','Day_100%_Acheived_Over_Calc_Buy','#_of_Transactions','#_in_Dataset','D2_Investible_Volume',
					'10%_Profit/Loss','20%_Profit/Loss','25%_Profit/Loss','30%_Profit/Loss','40%_Profit/Loss','50%_Profit/Loss']
				 
cluster_list = []
tnpcolumns = ['Trade_#','Log_Date','Trade_Date','Option_Symbol','Expiration']

def load_pickle(name):
	output = open(name, 'rb')
	# disable garbage collector
	gc.disable()
	mydict = pickle.load(output)
	# enable garbage collector again
	gc.enable()
	output.close()
	return mydict

def day_2_ratios(data):
	D2Op = pd.Series(data['Open'].shift(-1),name='D2Op')
	d2op_d1cl = pd.Series(D2Op/data['Close'],name='D2Op/D1Cl')
	data = data.join(d2op_d1cl)
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
	
def forw_targ(data): #Look for 5/10d forward highs and lows rather than forward value
	targ = (5,10,20)
	vals = (.02,.03,.04)
	TP = (data['High'] + data['Low'] + data['Close']) / 3
	for i,t in enumerate(targ):		
		l = pd.Series(data['Low'].shift((t)*-1).rolling(t,min_periods=1).min()/data['Open'].shift(-1), index=data.index, name=str(t) + '_Min_Low')
		h = pd.Series(data['High'].shift((t)*-1).rolling(t,min_periods=1).max()/data['Open'].shift(-1), index=data.index, name=str(t) + '_Max_Hi')
		targ_ratio = pd.Series(data=(TP.shift((t)*-1)/data['Open'].shift(-1)),index=data.index,name=str(t)+'_day_forward_typ_ratio')
		#data = data.join(targ_ratio)
		up_mask = (h-1)>=vals[i]
		down_mask = (1-l)>=vals[i]
		channel_mask = (~down_mask)&(~up_mask)
		volatility_mask = (down_mask)&(up_mask)
		remainder_mask = (~down_mask)&(~up_mask)
		bool_targ = pd.Series(data=np.ones(len(data))*3,index=data.index,name='Bool_'+str(t)+'_day_forward_Over_d2_ratio')
		bool_targ[up_mask] = 1
		bool_targ[down_mask] = -1
		bool_targ[channel_mask] = 0
		bool_targ[volatility_mask] = 2
		data = data.join(bool_targ)		
	return data

def stock_updater(symb,data):
	d = dt.date.today().date()
	stockcolumns = ['Open','High','Low','Close','Volume','Adj Close']
	stk = Share(symb)
	stock = pd.Series({'Open':stk.get_open(),'High': stk.get_days_high(),'Low': stk.get_days_low(),'Close': stk.get_price(),'Volume': stk.get_volume(), 'Adj Close':stk.get_price()}, name=d)
	stock = pd.to_numeric(stock)
	data = data.append(stock)
	data = data[~data.index.duplicated(keep='first')]
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
	signal = pd.Series(pd.ewma(macd, span = sign, min_periods = sign - 1)/macd, name = 'Signal_EMA_' + str(sign) + '_Normalized')
	data = data.join(signal)
	return data
	

def anchored_divergence(data,col,r,diff):#r = window for finding zeros, diff = value margin for finding zeros, cutoff = 
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
 #r = window for finding zeros, diff = value margin for finding zeros, cutoff = minimum swing to be classified as anchor #cutoff of 10%
def anchored_divergence_bool(data,col,r,diff,cutoff,offset):
	low_l = data[col].mean()-(((data[col].max()-data[col].min())/2)*cutoff)
	high_l = data[col].mean()+(((data[col].max()-data[col].min())/2)*cutoff)
	zzz = pd.Series()
	data[col] = pd.Series(data[col].rolling(window=5).mean(),data.index,name=col)
	anchors = pd.Series(index=data.index,name=col+'_Anchors')
	
	flipped = pd.Series(data=np.abs(data[col]-offset)-diff,index=data.index,name='Flipped')
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
	#data = data.join(anchors)
	anchor_indices = pd.Series(data=np.where((anchors == 1), data.index, data.index[0]),index=data.index,name=col+'_Anchor_Indices').astype(data.index.dtype)
	anchor_indices[anchor_indices == data.index[0]] = np.nan
	anchor_indices.fillna(method='ffill', inplace=True)
	anchor_indices.fillna(method='bfill',inplace=True)
	anchor_indices.sort_values(axis=0,inplace=True)
	data = data.join(anchor_indices)
	anchor_values = pd.Series(data=np.where((anchors == 1), data[col], np.nan),index=data.index,name=col+'_Anchor_Values')
	anchor_values.fillna(method='ffill', inplace=True)
	anchor_values.fillna(value=0,inplace=True)
	
	anchor_highs = pd.Series(data=np.where((anchors == 1), data['High'], np.nan),index=data.index,name=col+'_Anchor_Values')
	anchor_highs.fillna(method='ffill', inplace=True)
	anchor_highs.fillna(value=0,inplace=True)
	
	anchor_lows = pd.Series(data=np.where((anchors == 1), data['Low'], np.nan),index=data.index,name=col+'_Anchor_Values')
	anchor_lows.fillna(method='ffill', inplace=True)
	anchor_lows.fillna(value=0,inplace=True)
	
	
	high_divergence = pd.Series(data=(data['High']>anchor_highs) & (data[col]<=anchor_values),index=data.index,name=col+'_High_Divergence')
	data = data.join(high_divergence)
	
	high_div_count = pd.Series(data=high_divergence * (high_divergence.groupby((high_divergence != high_divergence.shift()).cumsum()).cumcount() + 1),index=data.index,name=col+'_High_Div_Count')
	data = data.join(high_div_count)
	  
	low_divergence = pd.Series(data=(data['Low']<anchor_lows) & (data[col]>=anchor_values),index=data.index,name=col+'_Low_Divergence')
	data = data.join(low_divergence)
	
	low_div_count = pd.Series(data=low_divergence * (low_divergence.groupby((low_divergence != low_divergence.shift()).cumsum()).cumcount() + 1),index=data.index,name=col+'_Low_Div_Count')
	data = data.join(low_div_count)
	return data
	
	

def mad(series):
    return np.mean(np.absolute(series - series[0]))

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
	k = pd.Series(100 * (data['Close'] - l) / (h - l), name = '%K_STO')
	data = data.join(k)
	data = data.join(l)
	data = data.join(h)
	data = data.fillna(value=0)
	return data
	
def SLSTOCH(data, ndays=3):
	d = pd.Series(data['%K_STO'].rolling(center=False,window=ndays).mean(), name='%D_STO')
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
	d = pd.Series(data['%K_STO'] - data['%D_STO'], name='D1_DIR_STO')
	dir = pd.Series(data=(data['%K_STO'] - data['%D_STO'])>0, name='D1_DIR__STO_Bool')	
	data = data.join(d)
	data = data.join(dir)
	data = data.fillna(value=0)
	return data

#D1_FAUXSTO
def FAUXSTO(data):
	d = pd.Series((data['%K_STO'] - data['%K_STO'].shift(+1))*(data['%D_STO'] - data['%D_STO'].shift(+1)), name='D1_FAUXSTO')
	data = data.join(d)
	data = data.fillna(value=0)
	return data

#ABSOL_UP_D1_STO
def STOCHABSOLUP(data):
	d = pd.Series((data['D1_FAUXSTO'] > 0) & (data['D1_DIR_STO'] > 0), name='ABSOL_UP_D1_STO')
	data = data.join(d)
	data = data.fillna(value=0)
	return data

#FRST_UP_OR_DWN_D1_%K_3PER_SLP
def K_UP_DOWN(data):
	km = pd.Series(pd.rolling_apply(arg=data['%K_STO'],func=three_linest,window=3),name='%K_3PER_SLP_STO')
	data = data.join(km)
	up = pd.Series(((data['%K_3PER_SLP_STO'] > 0) & (data['%K_3PER_SLP_STO'].shift(+1) < 0)), name='FIRST_UP_D1_%K_3PER_SLP_STO')
	data = data.join(up)
	down = pd.Series(((km < 0) & (km.shift(+1) > 0)), name='FIRST_DOWN_D1_%K_3PER_SLP_STO')
	data = data.join(down)
	return data


#CCI
def CCI(data, ndays): 
	TP = (data['High'] + data['Low'] + data['Close']) / 3 
	CCI = pd.Series(((TP - TP.rolling(center=False,window=ndays).mean())/ (0.015 * TP.rolling(center=False,window=ndays).apply(mad))),index=data.index, name='CCI_'+str(ndays))
	
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
	B1_slope = pd.Series(data=B1.rolling(window=3).apply(three_linest),index=data.index,name='BBAND_Upper_3p_Slope')
	data = data.join(B1)
	data = data.join(B1_slope) 	
	b2 = MA - (nstdev * SD)
	B2 = pd.Series(b2, name = 'Lower_BollingerBand') 
	B2_slope = pd.Series(data=B2.rolling(window=3).apply(three_linest),index=data.index,name='BBAND_Lower_3p_Slope')
	data = data.join(B2)
	data = data.join(B2_slope) 	
	#data = data.fillna(value=0)
	return data
	
#Bollinger Band Rank
def BBAND_RANK(data):
	rank = pd.Series(data=((data['Upper_BollingerBand']-data['Typical'])/(data['Upper_BollingerBand']-data['Lower_BollingerBand'])),index=data.index,name='BBAND_Rank')
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
	l = pd.Series(pd.rolling_min(data['Low'],ndays), name='LOWEST_Low_in_' + str(ndays))
	data = data.join(l)
	data = data.fillna(value=0)
	return data
	
def HLOWEST(data, ndays):
	l = pd.Series(pd.rolling_min(data['High'],ndays), name = 'LOWEST_Hi_in_' + str(ndays))
	data = data.join(l)
	data = data.fillna(value=0)
	return data
	
def BLOWEST(data, ndays): #not named, must be explicitly assigned to column ['BLOWEST_' + ndays]
	bl = pd.Series((data['LOWEST_Hi_in_' + str(ndays)] == data['High']) & (data['Low'] == data['LOWEST_Low_in_' + str(ndays)]), name = 'BLOWEST_' + str(ndays))
	data = data.join(bl)
	data = data.fillna(value=0)
	return data
	
#D1_H_LO_LOWEST_IN_X
def HLLOWESTIN(data, ndays):
	loin = pd.rolling_min(data['Low'], ndays)
	hloin = pd.rolling_min(data['High'], ndays)
	hlin = pd.Series((hloin == data['High']) & (data['Low'] == loin), name = 'D1_HI_LO_LOWEST_IN_' + str(ndays))
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

def count_consec(data,col):
	count = pd.Series(data=data[col] * (data[col].groupby((data[col] != data[col].shift()).cumsum()).cumcount() + 1),index=data.index,name='COUNT_'+col)
	data = data.join(count)
	return data
	
#ANN_LO_BY_DEFLAT_LOWEST
def ANNLOW(data, ndays=250):
	l = pd.Series(pd.rolling_min(data['Low'],ndays, min_periods=1), name='Annual_Low_250_Days')
	data = data.join(l)
	return data

def ANNLOWDEFLAT(data, ndays=250):
	ald = pd.Series((data['Annual_Low_250_Days'].shift(+1) > data['Annual_Low_250_Days']) & (data['Annual_Low_250_Days'] == data['Low']), name='ANN_LO_BY_DEFLAT_LOWEST')
	data = data.join(ald)
	return data	

def ANNLOWATTRIT(data, ndays=250):
	alt = pd.Series((data['Annual_Low_250_Days'].shift(+1) < data['Annual_Low_250_Days']) & (data['Annual_Low_250_Days'] != data['Low']), name='ANN_LO_BY_ATTR_LOWEST')
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

####Need to add in normalized volume indicator
				  
#Measures	
def directionality(data,ndays):
	nrange = (np.arange(int(ndays/5))+1)*5
	for n in nrange:
		high = pd.Series(data=data['High'].shift(int(n)*-1).rolling(window=n).max(),index=data.index)
		low = pd.Series(data=data['Low'].shift(int(n)*-1).rolling(window=n).min(),index=data.index)
		high_rat = pd.Series(data=((high-data['Typical'])/data['Typical']),index=data.index)
		low_rat = pd.Series(data=((data['Typical']-low)/data['Typical']),index=data.index)	
		dirat = pd.Series(data=(high_rat-low_rat)*100,index=data.index,name=str(n)+'_Day_Directionality(%Up-%Down)')
		data = data.join(dirat)
	return data

def momentum_peak(data,ndays): #ndays must be multiple of 5
	nrange = (np.arange(int(ndays/5))+1)*5
	for n in nrange:
		high = pd.Series(data=(((data['High'].shift(n*-1).rolling(window=n).max()-data['Typical'])/data['Typical'])/n), index=data.index)
		low = pd.Series(data=(((data['Low'].shift(n*-1).rolling(window=n).min()-data['Typical'])/data['Typical'])/n), index=data.index)
		peak = pd.Series(data=(np.where(high>abs(low),high,low)*100),index=data.index,name=str(n)+'_Day_Peak_Momentum(%/Day)')#reformat for positive bias
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
	new_data = pd.DataFrame(index=data.index)			 
	base = data['Close'].shift(1)
	op = pd.Series(data=data['Open']/base,index=data.index,name='Candle_Open')
	new_data = new_data.join(op)			 
	hi = pd.Series(data=data['High']/base,index=data.index,name='Candle_High')
	new_data = new_data.join(hi)				 
	lo = pd.Series(data=data['Low']/base,index=data.index,name='Candle_Low')
	new_data = new_data.join(lo)				 
	cl = pd.Series(data=data['Close']/base,index=data.index,name='Candle_Close')
	new_data = new_data.join(cl)	
	return new_data
	
def normalizer_float(data):
	new_data = pd.DataFrame(index=data.index)			 
	for col in data.columns:		 
		mx = data[col].max()
		mn = data[col].min()
		mx = mx + ((mx-mn)/100)
		mn = mn - ((mx-mn)/100)
		normalized = pd.Series(data=((data[col]-mn)/(mx-mn)),index=data.index, name=col+'_Norm')
		new_data = new_data.join(normalized)
	return new_data			       

def normalizer_bool(data):
	new_data = pd.DataFrame(index=data.index)			 
	for col in data.columns:
		normalized = pd.Series(data=np.where(data[col]==True,.99,.01),index=data.index, name=col+'_Norm')
		new_data = new_data.join(normalized)		       
	return new_data

def normalizer_centered(data):
	new_data = pd.DataFrame(data=data/2)
	return new_data
				       
def blob_trimmer(data):
	diff_mx = np.abs(data.iloc[-1] - data.ix[:-1])
	array = diff_mx.sum(axis=1).sort_values()
	powers = np.arange(6)+4			       
	for p in powers:
		mn = array.min()
		bin_width = (array.max() - array.min())/(2**p)		       
		bin_count = np.arange(2**p)
		bins =  np.array(mn + (bin_width * np.arange(2**p)))  #np.array([ mn + (bin_width * c) for c in bin_count]) # mn + (bin_width * np.arange(2**p))
		binplace = np.digitize(array, bins)		       
		bin_pop = np.array([len(array[binplace == i]) for i in range(1, len(bins))])
		bins = bins[1:]
		retained_bins_max = bins[(bin_pop>=bin_pop.mean()) & (bins>(bins.max()/2))].max()	#bin_pop>=bin_pop.quantile(dropout)	       
		trimmed_array = array[array<=retained_bins_max].sort()
		mean_diff = np.diff(trimmed_array).mean()
		if (array.max() - trimmed_array.max()) > mean_diff:
			array = trimmed_array
			continue
		else: 
			break
	return trimmed_array
	
def simple_cut(data,p):
	diff_mx = np.abs(data.iloc[-1] - data)
	array = diff_mx.sum(axis=1).sort_values()	
	if len(array[array<(p*.1)]) >= 10:
		array = array[array<(p*.1)]
	else:
		array = array[:10]
	return array
	
def clusterer(data, p, m):
    diff_mx = np.abs(data.iloc[-1] - data.ix[:-1])
    day_s = diff_mx.sum(axis=1).sort()
    eps = day_s.min() * (1+p)
    eps_slope = days_s.rolling(window=3,center=True).apply(three_linest)
    cluster = data.index[(day_s<eps) & (eps_slope<m)]
    return cluster 

##########SORT FUNCS#######

def next_monthlies_back(d): # must test len of option price history to ensure 1: that purchase is possible on d2 and 2: that there is at least 12 days of data
	f = d + dt.timedelta(days=35)
	g = d + dt.timedelta(days=60)
	g = dt.date(g.year, g.month, 15)
	f = dt.date(f.year, f.month, 15)
	g = (g + dt.timedelta(days=(calendar.FRIDAY - g.weekday()) % 7))
	f = (f + dt.timedelta(days=(calendar.FRIDAY - f.weekday()) % 7))
	e = f - dt.timedelta(weeks=2)
	return e,f,g

def next_monthly(d): # must test len of option price history to ensure 1: that purchase is possible on d2 and 2: that there is at least 12 days of data
	f = d + dt.timedelta(days=35)
	f = dt.date(f.year, f.month, 15)
	f = (f + dt.timedelta(days=(calendar.FRIDAY - f.weekday()) % 7))
	f = pd.to_datetime(f,infer_datetime_format=True)
	return f
	
def date_diff(dates1,dates2):
	diffs = ((dates2-dates1)/np.timedelta64(1, 'D')).astype(int)
	return diffs

def find_strike(array,value,movement):
	array = array.unique()
	value = value + movement
	array = np.sort(array, axis=0, kind='quicksort', order=None)
	idx = np.abs(array-value).argmin()
	stx = array[idx]	
	return stx
	
def find_delta(array,value):
	array = np.sort(array, axis=0, kind='quicksort', order=None)
	idx = np.abs(array-value).argmin()
	delta = array[idx]
	return delta
	


	
def call_trade_analyzer(data,sellpc,buypc,exstr,exitpc,mmgmt,strategyformula,acct,prices):
	trpl = ''.join([strategyformula,'_Trade_Profit/Loss'])
	profit = 1 + (sellpc-buypc)
	entries = pd.Series(data=(data['D2Lo/D2Op'] <= buypc),index=data.index,name=''.join([strategyformula,'_Got_In?']))
	if exstr == 'EX4':
		wins = pd.Series(data=((data['Per1-2Hi/D2Op'] >= sellpc) & (entries == True)),index=data.index,name=''.join([strategyformula,'_Wins?']))
	else:
		wins = pd.Series(data=((data['Per1Hi/D2Op'] >= sellpc) & (entries == True)),index=data.index,name=''.join([strategyformula,'_Wins?']))
	
	if exstr == 'EX1':
		successfulexits = pd.Series(data=((wins == False) & (entries == True) & (data['Per2Hi/D2Op'] >= exitpc)),index=data.index,name=''.join([strategyformula,'_Successful_Exits']))
	elif exstr == 'EX2':
		successfulexits = pd.Series(data=(wins == False) & (entries == True) & (data['Per2Hi/D2Op'] >= (buypc*.95)),index=data.index,name=''.join([strategyformula,'_Successful_Exits']))
	elif exstr == 'EX3':
		successfulexits = pd.Series(data=False,index=data.index,name=''.join([strategyformula,'_Successful_Exits']))
	elif exstr == 'EX4':
		successfulexits = pd.Series(data=((wins == False) & (entries == True) & (data['Per3-4Hi/D2Op'] >= exitpc)),index=data.index,name=''.join([strategyformula,'_Successful_Exits'])) # per4 needs to be added
	if exstr == 'EX3':
		escape = pd.Series(data=np.where(((wins == False) & (entries == True) & (successfulexits == False)),exstr, np.nan),index=data.index,name=''.join([strategyformula,'_Escapes?']))
	elif exstr == 'EX1':
		escape = pd.Series(data=(np.where((wins == False) & (entries == True) & (successfulexits == False),exstr, np.nan)),index=data.index,name=''.join([strategyformula,'_Escapes?']))
	elif exstr == 'EX2':
		escape = pd.Series(data=(np.where((wins == False) & (entries == True) & (successfulexits == False),exstr, np.nan)),index=data.index,name=''.join([strategyformula,'_Escapes?']))
	elif exstr == 'EX4':
		escape = pd.Series(data=(np.where((wins == False) & (entries == True) & (successfulexits == False),exstr, np.nan)),index=data.index,name=''.join([strategyformula,'_Escapes?']))
	tradepl = pd.Series(data=(np.where(wins == True,(acct * ((sellpc-buypc)*10)),
								np.where((successfulexits == True),((acct * (exitpc-buypc)*10)),
								np.where(escape == 'EX3',((acct * prices[6])-acct),
								np.where(((escape == 'EX2') & (successfulexits == False)),(np.maximum(((acct * prices[10])-acct),(acct*-1))),
								np.where(((escape == 'EX1') & (successfulexits == False)),(np.maximum(((acct * prices[10])-acct),(acct*-1))),
								np.where((escape == 'EX4'),(np.maximum(((acct * prices[15])-acct),(acct*-1))),np.nan))))))),index=data.index,name=trpl)
	ev = ''.join([strategyformula,'_Evaluation'])
	eval = pd.Series(data=(np.where(((wins == False) | ((wins == True) & (tradepl == (acct * ((sellpc-buypc)*10))))),True,False)),index=data.index, name =''.join([strategyformula,'_Win_Calc_Evaluation'])) #Discuss this calc with Dad

	datah = pd.concat([entries,wins,successfulexits,escape,tradepl,eval],axis=1)
	return datah

def put_trade_analyzer(data,sellpc,buypc,exstr,exitpc,mmgmt,strategyformula,acct,prices):
	trpl = ''.join([strategyformula,'_Trade_Profit/Loss'])
	profit = 1 + (buypc-sellpc)
	entries = pd.Series(data=(data['D2Hi/D2Op'] >= buypc),index=data.index,name=''.join([strategyformula,'_Got_In?']))
	if exstr == 'EX4':
		wins = pd.Series(data=((data['Per1-2Lo/D2Op'] <= sellpc) & (entries == True)),index=data.index,name=''.join([strategyformula,'_Wins?']))
	else:
		wins = pd.Series(data=((data['Per1Lo/D2Op'] <= sellpc) & (entries == True)),index=data.index,name=''.join([strategyformula,'_Wins?']))	
	if exstr == 'EX1':
		successfulexits = pd.Series(data=((wins == False) & (entries == True) & (data['Per2Lo/D2Op'] <= exitpc)),index=data.index,name=''.join([strategyformula,'_Successful_Exits']))
	elif exstr == 'EX2':
		successfulexits = pd.Series(data=(wins == False) & (entries == True) & (data['Per2Lo/D2Op'] <= (buypc*.95)),index=data.index,name=''.join([strategyformula,'_Successful_Exits']))
	elif exstr == 'EX3':
		successfulexits = pd.Series(data=False,index=data.index,name=''.join([strategyformula,'_Successful_Exits']))
	elif exstr == 'EX4':
		successfulexits = pd.Series(data=((wins == False) & (entries == True) & (data['Per3-4Lo/D2Op'] <= exitpc)),index=data.index,name=''.join([strategyformula,'_Successful_Exits'])) # per4 needs to be added
	if exstr == 'EX3':
		escape = pd.Series(data=np.where(((wins == False) & (entries == True) & (successfulexits == False)),exstr, np.nan),index=data.index,name=''.join([strategyformula,'_Escapes?']))
	elif exstr == 'EX1':
		escape = pd.Series(data=(np.where((wins == False) & (entries == True) & (successfulexits == False),exstr, np.nan)),index=data.index,name=''.join([strategyformula,'_Escapes?']))
	elif exstr == 'EX2':
		escape = pd.Series(data=(np.where((wins == False) & (entries == True) & (successfulexits == False),exstr, np.nan)),index=data.index,name=''.join([strategyformula,'_Escapes?']))
	elif exstr == 'EX4':
		escape = pd.Series(data=(np.where((wins == False) & (entries == True) & (successfulexits == False),exstr, np.nan)),index=data.index,name=''.join([strategyformula,'_Escapes?']))
	tradepl = pd.Series(data=(np.where(wins == True,(acct * ((buypc-sellpc)*10)),
								np.where(successfulexits == True,(acct * ((buypc-exitpc)*10)),
								np.where(escape == 'EX3',((acct * prices[6])-acct),
								np.where(((escape == 'EX2') & (successfulexits == False)),(np.maximum(((acct * prices[10])-acct),(acct*-1))),
								np.where(((escape == 'EX1') & (successfulexits == False)),(np.maximum(((acct * prices[10])-acct),(acct*-1))),
								np.where((escape == 'EX4'),(np.maximum(((acct * prices[15])-acct),(acct*-1))),np.nan))))))),index=data.index,name=trpl)
	ev = ''.join([strategyformula,'_Evaluation'])
	eval = pd.Series(data=(np.where(((wins == False) | ((wins == True) & (tradepl == (acct * ((buypc-sellpc)*10))))),True,False)),index=data.index, name =''.join([strategyformula,'_Win_Calc_Evaluation'])) #Discuss this calc with Dad
	datah = pd.concat([entries,wins,successfulexits,escape,tradepl,eval],axis=1)
	return datah

################################################################################################################	
	
def opt_call_trade_analyzer(data,sellpc,buypc,exstr,exitpc,mmgmt,strategyformula,acct,prices):
	trpl = ''.join([strategyformula,'_Trade_Profit/Loss'])
	profit = 1 + (sellpc-buypc)
	entries = pd.Series(data=(data['D2Lo/D2Op'] <= buypc),index=data.index,name=''.join([strategyformula,'_Got_In?']))
	if exstr == 'EX4':
		wins = pd.Series(data=((prices[3,4,5,6,7,8,9,10].max() >= sellpc) & (entries == True)),index=data.index,name=''.join([strategyformula,'_Wins?']))
	else:
		wins = pd.Series(data=((prices[3,4,5,6].max() >= sellpc) & (entries == True)),index=data.index,name=''.join([strategyformula,'_Wins?']))
	
	if exstr == 'EX1':
		successfulexits = pd.Series(data=((wins == False) & (entries == True) & (prices[7,8,9,10].max() >= exitpc)),index=data.index,name=''.join([strategyformula,'_Successful_Exits']))
	elif exstr == 'EX2':
		successfulexits = pd.Series(data=(wins == False) & (entries == True) & (prices[7,8,9,10].max() >= (buypc*.95)),index=data.index,name=''.join([strategyformula,'_Successful_Exits']))
	elif exstr == 'EX3':
		successfulexits = pd.Series(data=False,index=data.index,name=''.join([strategyformula,'_Successful_Exits']))
	elif exstr == 'EX4':
		successfulexits = pd.Series(data=((wins == False) & (entries == True) & (prices[11,12,13,14,15].max() >= exitpc)),index=data.index,name=''.join([strategyformula,'_Successful_Exits'])) # per4 needs to be added
	if exstr == 'EX3':
		escape = pd.Series(data=np.where(((wins == False) & (entries == True) & (successfulexits == False)),exstr, np.nan),index=data.index,name=''.join([strategyformula,'_Escapes?']))
	elif exstr == 'EX1':
		escape = pd.Series(data=(np.where((wins == False) & (entries == True) & (successfulexits == False),exstr, np.nan)),index=data.index,name=''.join([strategyformula,'_Escapes?']))
	elif exstr == 'EX2':
		escape = pd.Series(data=(np.where((wins == False) & (entries == True) & (successfulexits == False),exstr, np.nan)),index=data.index,name=''.join([strategyformula,'_Escapes?']))
	elif exstr == 'EX4':
		escape = pd.Series(data=(np.where((wins == False) & (entries == True) & (successfulexits == False),exstr, np.nan)),index=data.index,name=''.join([strategyformula,'_Escapes?']))
	tradepl = pd.Series(data=(np.where(wins == True,(acct * ((sellpc-buypc)*10)),
								np.where((successfulexits == True),((acct * (exitpc-buypc)*10)),
								np.where(escape == 'EX3',((acct * prices[6])-acct),
								np.where(((escape == 'EX2') & (successfulexits == False)),(np.maximum(((acct * prices[10])-acct),(acct*-1))),
								np.where(((escape == 'EX1') & (successfulexits == False)),(np.maximum(((acct * prices[10])-acct),(acct*-1))),
								np.where((escape == 'EX4'),(np.maximum(((acct * prices[15])-acct),(acct*-1))),np.nan))))))),index=data.index,name=trpl)
	ev = ''.join([strategyformula,'_Evaluation'])
	eval = pd.Series(data=(np.where(((wins == False) | ((wins == True) & (tradepl == (acct * ((sellpc-buypc)*10))))),True,False)),index=data.index, name =''.join([strategyformula,'_Win_Calc_Evaluation'])) #Discuss this calc with Dad

	datah = pd.concat([entries,wins,successfulexits,escape,tradepl,eval],axis=1)
	return datah

def opt_put_trade_analyzer(data,sellpc,buypc,exstr,exitpc,mmgmt,strategyformula,acct,prices):
	trpl = ''.join([strategyformula,'_Trade_Profit/Loss'])
	profit = 1 + (buypc-sellpc)
	entries = pd.Series(data=(data['D2Hi/D2Op'] >= buypc),index=data.index,name=''.join([strategyformula,'_Got_In?']))
	if exstr == 'EX4':
		wins = pd.Series(data=((prices[3,4,5,6,7,8,9,10].max() <= sellpc) & (entries == True)),index=data.index,name=''.join([strategyformula,'_Wins?']))
	else:
		wins = pd.Series(data=((prices[3,4,5,6].max() <= sellpc) & (entries == True)),index=data.index,name=''.join([strategyformula,'_Wins?']))	
	if exstr == 'EX1':
		successfulexits = pd.Series(data=((wins == False) & (entries == True) & (prices[7,8,9,10].max() <= exitpc)),index=data.index,name=''.join([strategyformula,'_Successful_Exits']))
	elif exstr == 'EX2':
		successfulexits = pd.Series(data=(wins == False) & (entries == True) & (prices[7,8,9,10].max() <= (buypc*.95)),index=data.index,name=''.join([strategyformula,'_Successful_Exits']))
	elif exstr == 'EX3':
		successfulexits = pd.Series(data=False,index=data.index,name=''.join([strategyformula,'_Successful_Exits']))
	elif exstr == 'EX4':
		successfulexits = pd.Series(data=((wins == False) & (entries == True) & (prices[11,12,13,14,15].max() <= exitpc)),index=data.index,name=''.join([strategyformula,'_Successful_Exits'])) # per4 needs to be added
	if exstr == 'EX3':
		escape = pd.Series(data=np.where(((wins == False) & (entries == True) & (successfulexits == False)),exstr, np.nan),index=data.index,name=''.join([strategyformula,'_Escapes?']))
	elif exstr == 'EX1':
		escape = pd.Series(data=(np.where((wins == False) & (entries == True) & (successfulexits == False),exstr, np.nan)),index=data.index,name=''.join([strategyformula,'_Escapes?']))
	elif exstr == 'EX2':
		escape = pd.Series(data=(np.where((wins == False) & (entries == True) & (successfulexits == False),exstr, np.nan)),index=data.index,name=''.join([strategyformula,'_Escapes?']))
	elif exstr == 'EX4':
		escape = pd.Series(data=(np.where((wins == False) & (entries == True) & (successfulexits == False),exstr, np.nan)),index=data.index,name=''.join([strategyformula,'_Escapes?']))
	tradepl = pd.Series(data=(np.where(wins == True,(acct * ((buypc-sellpc)*10)),
								np.where(successfulexits == True,(acct * ((buypc-exitpc)*10)),
								np.where(escape == 'EX3',((acct * prices[6])-acct),
								np.where(((escape == 'EX2') & (successfulexits == False)),(np.maximum(((acct * prices[10])-acct),(acct*-1))),
								np.where(((escape == 'EX1') & (successfulexits == False)),(np.maximum(((acct * prices[10])-acct),(acct*-1))),
								np.where((escape == 'EX4'),(np.maximum(((acct * prices[15])-acct),(acct*-1))),np.nan))))))),index=data.index,name=trpl)
	ev = ''.join([strategyformula,'_Evaluation'])
	eval = pd.Series(data=(np.where(((wins == False) | ((wins == True) & (tradepl == (acct * ((buypc-sellpc)*10))))),True,False)),index=data.index, name =''.join([strategyformula,'_Win_Calc_Evaluation'])) #Discuss this calc with Dad
	datah = pd.concat([entries,wins,successfulexits,escape,tradepl,eval],axis=1)
	return datah

##############################TNP FUNC######################################

def TnPLogger(data,d):	
	tnplogger = pd.DataFrame(columns = tnpcolumns)		
	stocks = data['Stock_Symbol'].unique()
	i = -1
	for symb in stocks:
		#stk = web.DataReader(symb, 'google', start, d)
		#center = stk.ix[d]['open']
		trades = data[data['Stock_Symbol']==symb]
		calls = trades[trades['Option']=='call']
		puts = trades[trades['Option']=='put']
		cstx = calls['Strike'].unique()
		pstx = puts['Strike'].unique()		
		if len(cstx) > 0:
			for x in cstx: #Filter by Buy/Sell Calcs
				i = i + 1
				optc = calls[calls['Strike']==x]				
				#min_exp,exp,max_exp = next_monthlies(d)
				#opt = option_finder(symb=symb,center=center,t='c',min_exp=min_exp,exp=exp,max_exp=max_exp,m=x)
				tnplogger.at[i,'Trade_Date'] =	d
				tnplogger.at[i,'Option_Type'] = 'Call'
				tnplogger.at[i,'Strike_Position(stx)'] = x
				tnplogger.at[i,'Underlying_Symbol'] = symb
				optc.sort_values(by='Win%',ascending=True,axis=0,inplace=True)
				strtgs = optc['Strategy_Formula'].unique()
				strtgs = strtgs[:20]
				#tnplogger['Suggested_Option_Symbol'] = opt
				#tnplogger['Expiration'] = opt.split(',',3)[0]
				z = -1
				for s in strtgs:
					z = z + 1				
					col = ''.join(['Strategy_',str(z+1)])
					tnplogger.at[i,col] = s
					
		else: 
			pass
		if len(pstx) > 0:
			for x in pstx:
				i = i + 1
				optp = puts[puts['Strike']==x]				
				#min_exp,exp,max_exp = next_monthlies(d)
				#opt = option_finder(symb=symb,center=center,t='p',min_exp=min_exp,exp=exp,max_exp=max_exp,m=x)
				tnplogger.at[i,'Trade_Date'] =	d
				tnplogger.at[i,'Option_Type'] = 'Put'
				tnplogger.at[i,'Strike_Position(stx)'] = x
				tnplogger.at[i,'Underlying_Symbol'] = symb
				optp.sort_values(by='Win%',ascending=True,axis=0,inplace=True)
				strtgs = optp['Strategy_Formula'].unique()
				strtgs = strtgs[:20]
				#tnplogger['Option_Symbol'] = opt
				#tnplogger['Expiration'] = opt.split(',',3)[0]
				z = -1
				for s in strtgs:
					z = z + 1				
					col = ''.join(['Strategy_',str(z+1)])
					tnplogger.at[i,col] = s
		else: 
			pass
	tnplogger['Trade_#'] = tnplogger.index + 1
	tnplogger['Log_Date'] = dt.date.today()
	return tnplogger
	
def tnp_single_trade_logger(data,d):
	tnplogger = pd.DataFrame(columns=tnpcolumns)
	lcr = pd.DataFrame(columns=lcrcolumns)
	stocks = data['Stock_Symbol'].unique()
	i = -1
	if len(data) > 1:
		for symb in stocks:
			trades = data[data['Stock_Symbol']==symb]
			calls = trades[trades['Option']=='call']
			puts = trades[trades['Option']=='put']
			call = calls[calls['Hist_Profit/Loss_per_Tx'] == calls['Hist_Profit/Loss_per_Tx'].max()]
			call = call[call['Max_Drawdown(Acct_Min)'] == call['Max_Drawdown(Acct_Min)'].min()]
			put = puts[puts['Hist_Profit/Loss_per_Tx'] == puts['Hist_Profit/Loss_per_Tx'].max()]
			put = put[put['Max_Drawdown(Acct_Min)'] == put['Max_Drawdown(Acct_Min)'].min()]
			call.reset_index(drop=False,inplace=True)
			put.reset_index(drop=False,inplace=True)
			if len(call) > 0:
				i = i + 1
				lcr.at[i] = call.ix[0]
				tnplogger.at[i,'Trade_#'] = lcr.index[i]+1
				tnplogger.at[i,'Trade_Date'] =	d		
			if len(put) > 0:
				i = i + 1
				lcr.at[i] = put.ix[0]
				tnplogger.at[i,'Trade_#'] = lcr.index[i]+1
				tnplogger.at[i,'Trade_Date'] =	d
	tnplogger = pd.concat([tnplogger,lcr],axis=1)
	return tnplogger
	
def nearest_exp(series,dat,i):
	aray = series.reset_index(drop=True)
	idx = np.abs(aray-dat).argmin()
	try:
		exp = aray[idx+i]
	except:
		try:
			exp = aray[idx-i]
		except:
			exp = aray[idx]
	exp = dt.date(exp.year, exp.month, exp.day)
	return exp
	
def option_recaller(opt,symb,strike,start,exp): # open opt_key outside of func in begining of loop and pass df as argument
	opt = opt[opt['quote_date'] >= start]
	opt = opt[opt['strike'] == strike]	
	opt = opt[opt['expiration'] == exp]	
	opt.drop_duplicates(['expiration', 'strike','quote_date'],inplace=True)	
	if (opt.ix[opt.index[1]]['open'] == 0):
		opt.ix[opt.index[1],'open'] = (opt.ix[opt.index[1]]['bid_1545'] + opt.ix[opt.index[1]]['ask_1545'])/2
	
	#opt.set_index('quote_date',drop=True,inplace=True)
	opt = opt[['underlying_symbol', 'root', 'expiration', 'strike', 'option_type',
				 'open', 'high', 'low', 'close', 'trade_volume', 'bid_size_1545',
				 'bid_1545', 'ask_size_1545', 'ask_1545', 'underlying_bid_1545',
				 'underlying_ask_1545', 'implied_underlying_price_1545',
				 'active_underlying_price_1545', 'implied_volatility_1545', 'delta_1545',
				 'gamma_1545', 'theta_1545', 'vega_1545', 'rho_1545', 'bid_size_eod',
				 'bid_eod', 'ask_size_eod', 'ask_eod', 'underlying_bid_eod',
				 'underlying_ask_eod', 'vwap', 'open_interest', 'delivery_code']]	
	return opt

def option_finder(opt,symb,strike,start,opt_type,min_exp,exp,max_exp,m,stxx): # open opt_key outside of func in begining of loop and pass df as argument
	if opt_type == 'call':
		min = 1
		max = 1.05
	elif opt_type == 'put':
		min = .95
		max = 1
	opt = opt[opt['quote_date'] >= start]
	opt = opt[opt['expiration'] <= max_exp]
	opt = opt[opt['expiration'] >= min_exp]	
	opte = opt[opt['quote_date']==start]	
	if len(opte) > 1:
		if stxx == 'strike':
			stx = find_strike(opte['strike'],strike,m)
			opt = opt[opt['strike'] == stx]
			opte = opte[opte['strike'] == stx]
			exps = pd.Series(opte['expiration'].unique()).sort_values()
			for i in np.arange(len(exps)):
				x = nearest_exp(exps,exp,i)
				opte = opt[opt['expiration'] == x].reset_index(drop=True)
				if opte.ix[1]['open'] > 0:
					opt = opte
					break
				elif (opte.ix[opte.index[1]]['bid_1545'] > 0) & (opte.ix[opte.index[1]]['ask_1545'] > 0):
						opte.ix[opte.index[1],'open'] = (opte.ix[opte.index[1]]['bid_1545'] + opte.ix[opte.index[1]]['ask_1545'])/2
						opt = opte
						break
				elif i == len(exps)-1:
					print(opte.ix[:10])
					opt = opt[opt['expiration'] == exp]					
					continue
		elif stxx == 'delta':
			opts = opt[opt['quote_date'] == start]
			opts = opts[(opts['strike'] >= opts['underlying_bid_1545']*min) & (opts['strike'] <= opts['underlying_bid_1545']*max)]
			x = nearest_exp(exps,exp,i)
			opts = opts[opts['expiration'] == x]
			opts['delta_ratio'] = np.abs((opts['delta_1545']/opts['close'])/(1/(((opts['underlying_bid_1545']+opts['underlying_ask_1545'])/2)/100)))
			opts['delta_ratio'][np.isinf(opts['delta_ratio'])] = 0
			stx = find_delta(opts['delta_ratio'],strike)
			opts = opts[opts['delta_ratio'] == stx]	
		opt.drop_duplicates(['expiration', 'strike','quote_date'],inplace=True)	
		
		opt.set_index('quote_date',drop=True,inplace=True)
		print (opt,x,stx,min_exp,exp,max_exp,exps)
		if len(opt)>0:
			option = ''.join([str(opt.loc[opt.index[0]]['expiration'].date()),',',opt_type,',',str(stx),',',symb])
			opt = opt[['underlying_symbol', 'root', 'expiration', 'strike', 'option_type',
						 'open', 'high', 'low', 'close', 'trade_volume', 'bid_size_1545',
						 'bid_1545', 'ask_size_1545', 'ask_1545', 'underlying_bid_1545',
						 'underlying_ask_1545', 'implied_underlying_price_1545',
						 'active_underlying_price_1545', 'implied_volatility_1545', 'delta_1545',
						 'gamma_1545', 'theta_1545', 'vega_1545', 'rho_1545', 'bid_size_eod',
						 'bid_eod', 'ask_size_eod', 'ask_eod', 'underlying_bid_eod',
						 'underlying_ask_eod', 'vwap', 'open_interest', 'delivery_code']]	
		else:
			opt = 'NA'
			option = 'NA'
	elif len(opte) == 1:
		opt.reset_index(drop=True,inplace=True)
		opt = opt[(opt['expiration'] == opt.ix[0]['expiration'])&(opt['strike'] == opt.ix[0]['strike'])]
		option = ''.join([str(opt.loc[opt.index[0]]['expiration'].date()),',',opt_type,',',str(opt.loc[opt.index[0]]['strike']),',',symb])
	else:
		opt = 'NA'
		option = 'NA'
		print('No Option Found:..',min_exp,exp,max_exp)
	return opt, option
	
def tnp_analyzer(opt_data,entry_perc,sell_perc,exit_strg,acct):
	analysis = pd.DataFrame(index=np.arange(1),columns=tnpsummarycolumns)
	opt_data.reset_index(drop=False,inplace=True)
	print(opt_data)
	analysis['D2Op'] = opt_data.ix[1]['open']
	analysis['D2Hi'] = opt_data.ix[1]['high']
	analysis['D2Lo'] = opt_data.ix[1]['low']
	analysis['D2Cl'] = opt_data.ix[1]['close']
	analysis['D2Vol'] = opt_data.ix[1]['trade_volume']
	sell_targ = sell_perc * opt_data.ix[1]['open']
	multiplier = int(acct/(opt_data.ix[1]['open']*entry_perc)/100)
	mx = int(acct/(opt_data.ix[1]['open'])/100)
	analysis['Entry_Target'] = (opt_data.ix[1]['open']*entry_perc)
	analysis['Sell_Target'] = sell_targ
	analysis['#_of_Contracts'] = multiplier
	analysis['Trade_Hi_Over_D2Op_(%)'] = opt_data.ix[2:20]['high'].max()/opt_data.ix[1]['open']
	analysis['delta_1545'] = opt_data.ix[1]['delta_1545']
	buys = (opt_data.ix[1]['open'],(opt_data.ix[1]['open']*entry_perc))
	buy_points = ('_Over_D2Op', '_Over_Calc_Buy')
	for i in np.arange(2): 
		if opt_data.ix[2:]['high'].max() >= buys[i] * 2:
			analysis['Day_100%_Acheived'+buy_points[i]] = opt_data.index.get_loc(opt_data.ix[2:][opt_data.ix[2:]['high'] >= buys[i] * 2].index[0]) + 1
		if opt_data.ix[2:]['high'].max() >= buys[i] * 1.75:
			analysis['Day_75%_Acheived'+buy_points[i]] = opt_data.index.get_loc(opt_data.ix[2:][opt_data.ix[2:]['high'] >= buys[i] * 1.75].index[0]) + 1
		if opt_data.ix[2:]['high'].max() >= buys[i] * 1.5:
			analysis['Day_50%_Acheived'+buy_points[i]] = opt_data.index.get_loc(opt_data.ix[2:][opt_data.ix[2:]['high'] >= buys[i] * 1.5].index[0]) + 1
		if opt_data.ix[2:]['high'].max() >= buys[i] * 1.25:
			analysis['Day_25%_Acheived'+buy_points[i]] = opt_data.index.get_loc(opt_data.ix[2:][opt_data.ix[2:]['high'] >= buys[i] * 1.25].index[0]) + 1
		if opt_data.ix[2:]['high'].max() >= buys[i] * 1.2:
			analysis['Day_20%_Acheived'+buy_points[i]] = opt_data.index.get_loc(opt_data.ix[2:][opt_data.ix[2:]['high'] >= buys[i] * 1.2].index[0]) + 1
		if opt_data.ix[2:]['high'].max() >= buys[i] * 1.1:
			analysis['Day_10%_Acheived'+buy_points[i]] = opt_data.index.get_loc(opt_data.ix[2:][opt_data.ix[2:]['high'] >= buys[i] * 1.1].index[0]) + 1
			
	#Go for the Kill
	target_range = (10,20,25,30,40,50)
	for v in target_range:
		if opt_data.ix[2:14]['high'].max() >= (opt_data.ix[1]['open']*(1+(v/100))):
			analysis[str(v)+'%_Profit/Loss'] = ((opt_data.ix[1]['open']*(1+(v/100)))-opt_data.ix[1]['open'])*mx*100
		elif opt_data.ix[15:]['high'].max() >= (opt_data.ix[1]['open']*.75):
			analysis[str(v)+'%_Profit/Loss'] = ((opt_data.ix[1]['open']*.75)-opt_data.ix[1]['open'])*mx*100
		elif len(opt_data) > 19:
			analysis[str(v)+'%_Profit/Loss'] = ((opt_data.ix[19]['close']/opt_data.ix[1]['open'])-opt_data.ix[1]['open'])*mx*100
		else: 
			analysis[str(v)+'%_Profit/Loss'] = ((opt_data.iloc[-1]['close']/opt_data.ix[1]['open'])-opt_data.ix[1]['open'])*mx*100
			
	if opt_data.ix[1]['low'] <= (opt_data.ix[1]['open']*entry_perc):
		analysis['Entry?'] = True
	else:
		analysis['Entry?'] = False

	if (analysis.ix[0]['Entry?'] == True) & (exit_strg == 'EX4') & (opt_data.ix[2:10]['high'].max() >= sell_targ):
		analysis['Win?'] = True
		analysis['Day_Calc_Sell_Reached'] = opt_data.ix[2:][opt_data.ix[2:]['high'] >= sell_targ].index[0] + 1
	elif (analysis.ix[0]['Entry?'] == True) & (exit_strg != 'EX4') & (opt_data.ix[2:6]['high'].max() >= sell_targ):
		analysis['Win?'] = True
		analysis['Day_Calc_Sell_Reached'] = opt_data.ix[2:][opt_data.ix[2:]['high'] >= sell_targ].index[0] + 1
	elif (analysis.ix[0]['Entry?'] == True):
		analysis['Win?'] = False

	
	
	if (analysis.ix[0]['Win?'] == False) & (exit_strg == 'EX1') & (opt_data.ix[6:10]['high'].max() >= sell_targ):
		analysis['Successful_Exit?'] = True
		analysis['Exit_Price'] = (opt_data.ix[1]['open']*sell_perc) #Absolute Value
	elif (analysis.ix[0]['Win?'] == False) & (exit_strg == 'EX2') & (opt_data.ix[6:10]['high'].max() >= (opt_data.ix[1]['open']*entry_perc*.95)): 
		analysis['Successful_Exit?'] = True
		analysis['Exit_Price'] = (opt_data.ix[1]['open']*entry_perc*.95) #Absolute Value
	elif (analysis.ix[0]['Win?'] == False) & (exit_strg == 'EX3'):  
		analysis['Successful_Exit?'] = False
		analysis['Exit_Price'] = opt_data.ix[5]['close'] #Absolute Value
	elif (analysis.ix[0]['Win?'] == False) & (exit_strg == 'EX4') & (opt_data.ix[10:19]['high'].max() >= (sell_targ*.98)):
		analysis['Successful_Exit?'] = True
		analysis['Exit_Price'] = ((sell_targ*.98)) #Absolute Value
	elif (analysis.ix[0]['Win?'] == False): 
		analysis['Successful_Exit?'] = False
		analysis['Exit_Price'] = 0

	if analysis.ix[0]['Successful_Exit?'] == False:
		if (exit_strg == 'EX3'):
			analysis['Escape?'] =  True
			analysis['Escape_Price'] = (opt_data.ix[5]['close'])
		elif (exit_strg == 'EX4'):
			analysis['Escape?'] =  True
			if len(opt_data) >= 20:
				analysis['Escape_Price'] = (opt_data.ix[19]['close'])
			else:
				analysis['Escape_Price'] = (opt_data.iloc[-1]['close'])
		else:
			analysis['Escape?'] =  True
			analysis['Escape_Price'] = (opt_data.ix[9]['close'])
			

	if (analysis.ix[0]['Win?'] == True):
		analysis['Profit/Loss'] = ((opt_data.ix[1]['open']*sell_perc) - (opt_data.ix[1]['open']*entry_perc))*multiplier*100
	elif (analysis.ix[0]['Win?'] == False) & (analysis.ix[0]['Successful_Exit?'] == True):
		analysis['Profit/Loss'] = (analysis.ix[0]['Exit_Price'] - (opt_data.ix[1]['open']*entry_perc))*multiplier*100
	elif (analysis.ix[0]['Win?'] == False) & (analysis.ix[0]['Successful_Exit?'] == False):  
		analysis['Profit/Loss'] = (analysis['Escape_Price'] - (opt_data.ix[1]['open']*entry_perc))*multiplier*100
	elif (analysis.ix[0]['Entry?'] == False):
		analysis['Profit/Loss'] = 0

	return analysis
	
def decoder(strategy):
#"20/10.GLB,7-10.GLB,EX1,MGMT1"
	st = strategy.split(',',3)
	buysell = st[0].split('/',1)
	buyperc = int(buysell[0])
	buysell = buysell[1].split('.',1)
	sellperc = int(buysell[0])
	sellset = buysell[1]
	if len(st[1]) != 0:
		exit = st[1].split('.',1)
		exitperiod = exit[0]
		exitset = exit[1]
	else:
		exitperiod = np.nan
		exitset = np.nan		
	exitstrategy = st[2]
	mmgmt = st[3]	
	return buyperc,sellperc,sellset,exitperiod,exitset,exitstrategy,mmgmt
	
def reporter(data,strategy,account,option,prices,criteria,price_base):
	buyperc,sellperc,sellset,exitperiod,exitset,exstr,mmgmt = decoder(strategy)
	if price_base == 0:
		if option == 'call':			
			heads = ['D2Lo/D2Op','Trade_Hi','Per1Hi/D2Op','Per2Hi/D2Op','Per1-2Hi/D2Op','Per3-4Hi/D2Op']
			quants = [.1,.2,.4,.5,.6]
		elif option == 'put':			
			heads = ['D2Hi/D2Op','Trade_Lo','Per1Lo/D2Op','Per2Lo/D2Op','Per1-2Lo/D2Op','Per3-4Lo/D2Op']			
			buyperc = 100-buyperc
			sellperc = 100-sellperc
			quants = [.9,.8,.6,.5,.4]
	elif price_base == 1:
		quants = [.1,.2,.4,.5,.6]

			
			
	report = pd.DataFrame(index=np.arange(1),columns=criteria)
	report['D2Op/D1Cl'] = data['D2Op/D1Cl'].mean()
	
	report['Strategy_Formula'] = strategy
	report['#_in_Dataset'] = len(data)
	
	#Buy Price
	if price_base == 0:
		buypc = data[heads[0]].quantile(q=buyperc/100)
	if price_base == 1:
		buypc = prices[1].quantile(q=buyperc/100)
	acct = account * .5
	
	#SemiSet
	if option == 'call':
		semiset = data[data[heads[0]] <= buypc]
		if price_base == 1:
			buypc = ((buypc-1)*10)+1		#Semi-Restricted Data Set
	elif option == 'put':
		semiset = data[data[heads[0]] >= buypc]
		if price_base == 1:
			buypc = ((1-buypc)*10)+1
	report['Buy_Target_%'] = buypc
	
	#Quantiles
	if exstr == 'EX4':
		if price_base == 0:
			if option == 'call': 
				sellpc = buypc * 1.025
			elif option == 'put':
				sellpc = buypc * .975
			report['Win_Period_10th/90th'] = data[heads[4]].quantile(q=quants[0])
			report['Win_Period_20th/80th'] = data[heads[4]].quantile(q=quants[1])
			report['Win_Period_40th/60th'] = data[heads[4]].quantile(q=quants[2])
			report['Win_Period_50th/50th'] = data[heads[4]].quantile(q=quants[3])
			report['Win_Period_60th/40th'] = data[heads[4]].quantile(q=quants[4])
		elif price_base == 1:
			sellpc = buypc * 1.25
			report['Win_Period_10th/90th'] = prices[3,4,5,6,7,8,9,10].stack().quantile(q=quants[0])
			report['Win_Period_20th/80th'] = prices[3,4,5,6,7,8,9,10].stack().quantile(q=quants[1])
			report['Win_Period_40th/60th'] = prices[3,4,5,6,7,8,9,10].stack().quantile(q=quants[2])
			report['Win_Period_50th/50th'] = prices[3,4,5,6,7,8,9,10].stack().quantile(q=quants[3])
			report['Win_Period_60th/40th'] = prices[3,4,5,6,7,8,9,10].stack().quantile(q=quants[4])
	else: 
		if sellset == 'SEMI':
			if price_base == 0:
				sellpc = semiset[heads[2]].quantile(q=sellperc/100)
				report['Win_Period_10th/90th'] = semiset[heads[2]].quantile(q=quants[0])
				report['Win_Period_20th/80th'] = semiset[heads[2]].quantile(q=quants[1])
				report['Win_Period_40th/60th'] = semiset[heads[2]].quantile(q=quants[2])
				report['Win_Period_50th/50th'] = semiset[heads[2]].quantile(q=quants[3])
				report['Win_Period_60th/40th'] = semiset[heads[2]].quantile(q=quants[4])
			elif price_base == 1:
				sellpc = prices[3,4,5,6].stack().quantile(q=sellperc/100)#need to set the indexing
				report['Win_Period_10th/90th'] = prices[3,4,5,6].stack().quantile(q=quants[0])
				report['Win_Period_20th/80th'] = prices[3,4,5,6].stack().quantile(q=quants[1])
				report['Win_Period_40th/60th'] = prices[3,4,5,6].stack().quantile(q=quants[2])
				report['Win_Period_50th/50th'] = prices[3,4,5,6].stack().quantile(q=quants[3])
				report['Win_Period_60th/40th'] = prices[3,4,5,6].stack().quantile(q=quants[4])
		else:
			if price_base == 0:
				sellpc = data[heads[2]].quantile(q=sellperc/100)
				report['Win_Period_10th/90th'] = data[heads[2]].quantile(q=quants[0])
				report['Win_Period_20th/80th'] = data[heads[2]].quantile(q=quants[1])
				report['Win_Period_40th/60th'] = data[heads[2]].quantile(q=quants[2])
				report['Win_Period_50th/50th'] = data[heads[2]].quantile(q=quants[3])
				report['Win_Period_60th/40th'] = data[heads[2]].quantile(q=quants[4])
			elif price_base == 1:
				sellpc = prices[3,4,5,6].stack().quantile(q=sellperc/100)
				report['Win_Period_10th/90th'] = prices[3,4,5,6].stack().quantile(q=quants[0])
				report['Win_Period_20th/80th'] = prices[3,4,5,6].stack().quantile(q=quants[1])
				report['Win_Period_40th/60th'] = prices[3,4,5,6].stack().quantile(q=quants[2])
				report['Win_Period_50th/50th'] = prices[3,4,5,6].stack().quantile(q=quants[3])
				report['Win_Period_60th/40th'] = prices[3,4,5,6].stack().quantile(q=quants[4])
		
	restset = semiset[semiset[heads[2]] >= sellpc] #Restricted Data Set
	report['Sell_Target_%'] = sellpc
	#Exit Percentage Target Based On Exit Set Specification
	if exstr != 'EX4':
		if exitset == 'GLB':
			if price_base == 0:
				exitpc = data[heads[3]].quantile(q=sellperc/100)
			elif price_base == 1:
				exitpc = prices[7,8,9,10].stack().quantile(q=sellperc/100)
		elif exitset == 'SEMI':
			if price_base == 0:
				exitpc = semiset[heads[3]].quantile(q=sellperc/100)
			elif price_base == 1:
				exitpc = prices[7,8,9,10].stack().quantile(q=sellperc/100)
		elif exitset == 'REST':
			if price_base == 0:		
				exitpc = restset[heads[3]].quantile(q=sellperc/100)
			elif price_base == 1:
				exitpc = prices[7,8,9,10].stack().quantile(q=sellperc/100)
	elif exstr == 'EX4':
		if price_base == 0:	
			if option == 'call':
					exitpc = buypc * 1.05
			elif option == 'put':
					exitpc = buypc * .95
		elif price_base == 1:
			exitpc = buypc * 1.05
		
	#Generate Column Names
	wins = ''.join([strategyformula,'_Wins?'])
	trpl = ''.join([strategyformula,'_Trade_Profit/Loss'])
	ev = ''.join([strategyformula,'_Win_Calc_Evaluation'])
	biglosstx = ''.join([strategyformula,'_TXs_Loss>50%'])
	cumprofit = ''.join([strategyformula,'_Cumulative_Profit/Loss'])
	entries = ''.join([strategyformula,'_Got_In?'])
	successfulexits = ''.join([strategyformula,'_Successful_Exits'])
	
	#Analyze
	if option == 'call':
		report['Calc_Profit%'] = (sellpc-buypc)*10
		pft = 1 + ((sellpc-buypc)*10)
		if price_base == 0:
			datah = call_trade_analyzer(data,sellpc=sellpc,buypc=buypc,exstr=exstr,exitpc=exitpc,mmgmt=mmgmt,strategyformula=strategyformula,acct=acct,prices=prices)
		elif price_base == 1:
			datah = opt_call_trade_analyzer(data,sellpc=sellpc,buypc=buypc,exstr=exstr,exitpc=exitpc,mmgmt=mmgmt,strategyformula=strategyformula,acct=acct,prices=prices)
	elif option == 'put':
		report['Calc_Profit%'] = (buypc-sellpc)*10
		pft = 1 + ((buypc-sellpc)*10)
		if price_base == 0:
			datah = put_trade_analyzer(data,sellpc=sellpc,buypc=buypc,exstr=exstr,exitpc=exitpc,mmgmt=mmgmt,strategyformula=strategyformula,acct=acct,prices=prices)
		elif price_base == 1:
			
			datah = opt_put_trade_analyzer(data,sellpc=sellpc,buypc=buypc,exstr=exstr,exitpc=exitpc,mmgmt=mmgmt,strategyformula=strategyformula,acct=acct,prices=prices)
		
	#Final Calcs
	#report['D2_Investible_Volume'] = data.ix[-20:]['d2ivst'].mean()#test
	report['#_of_Transactions'] = np.sum(datah[entries])
	datah[biglosstx] = datah[trpl] <= (acct * -.5)
	report['#_OF_"FALSE"_IN_WIN_CALC'] = len(data) - np.sum(datah[ev]) #Investigate: Clarify the meaning of the Win Calc and its calculation
	report['%_of_TXs_w/_LOSS>50%'] = np.sum(datah[biglosstx])/len(semiset)
	datah[cumprofit] = datah[trpl].cumsum()
	report['Win%'] = np.sum(datah[wins])/np.sum(datah[entries])
	report['Fail%'] = 1-report['Win%']
	if option == 'call':
		report['CALC._PROF/LOSS_ON_Fd_TXS'] = exitpc-buypc
	elif option == 'put':
		report['CALC._PROF/LOSS_ON_Fd_TXS'] = buypc-exitpc
	report['Highest_Hi'] = datah[cumprofit].max()
	report['Biggest_Loser'] = datah[trpl].min()
	report['Max_Drawdown(Acct_Min)'] = datah[cumprofit].min()
	report['Max_%_Drawdown'] = datah[cumprofit].min()/acct
	net = datah[datah[entries] == True]
	report['Net_Profit/Loss'] = net.ix[-1][cumprofit]
	report['Ratio_Net/Highest_Hi'] = report['Net_Profit/Loss']/report['Highest_Hi']
	report['Hist_Profit/Loss_per_Tx'] = (report['Net_Profit/Loss']/report['#_of_Transactions'])
	report['#_TX>0'] = np.sum(datah[trpl]>0)
	report['%_of_TX>0'] = np.sum(datah[trpl]>0)/len(semiset)
	report['Catastrophic_Fail_%_(-80%)'] = np.sum(datah[trpl]<=(acct*(-.8)))/len(datah[trpl])
	report['AMT_AT_RISK'] = acct
	report['#_of_Exit_Attempts'] = np.sum(datah[entries]) - np.sum(datah[wins])
	report['%_Successful_Exits'] = np.sum(datah[successfulexits])/(np.sum(datah[entries]) - np.sum(datah[wins]))
	
	
	if (mmgmt == 'MGMT1') & (report.ix[0]['#_of_Transactions'] > 15) & (report.ix[0]['Ratio_Net/Highest_Hi'] >= .9499) & (report.ix[0]['Hist_Profit/Loss_per_Tx'] >= (acct * .099)) & (report.ix[0]['#_OF_"FALSE"_IN_WIN_CALC'] < 2) &  (report.ix[0]['%_of_TXs_w/_LOSS>50%'] < .06) & (report.ix[0]['Win%'] >= .8) & (report.ix[0]['Catastrophic_Fail_%_(-80%)'] == 0) & (report.ix[0]['Calc_Profit%'] > .1):
		report['Level'] = 2
	elif (mmgmt == 'MGMT1') & (report.ix[0]['#_of_Transactions'] > 15) & (report.ix[0]['Ratio_Net/Highest_Hi'] >= .9499) & (report.ix[0]['Hist_Profit/Loss_per_Tx'] >= (acct * .099)) & (report.ix[0]['#_OF_"FALSE"_IN_WIN_CALC'] < 2) &  (report.ix[0]['%_of_TXs_w/_LOSS>50%'] < .06) & (report.ix[0]['Win%'] >= .9) & (report.ix[0]['Catastrophic_Fail_%_(-80%)'] == 0) & (report.ix[0]['Calc_Profit%'] > .15): 
		report['Level'] = 1
	elif (mmgmt == 'MGMT1') & (report.ix[0]['#_of_Transactions'] > 15) & (report.ix[0]['Ratio_Net/Highest_Hi'] >= .9499) & (report.ix[0]['Hist_Profit/Loss_per_Tx'] >= (acct * .299)) & (report.ix[0]['#_OF_"FALSE"_IN_WIN_CALC'] < 2) &  (report.ix[0]['%_of_TXs_w/_LOSS>50%'] < .06) & (report.ix[0]['Win%'] >= .92) & (report.ix[0]['Catastrophic_Fail_%_(-80%)'] == 0) & (report.ix[0]['Calc_Profit%'] > .20): 
		report['Level'] = 0
	else:
		report['Level'] = 3
	return report, datah 

def buy_sell(data):
	#option = pd.Series((x.split(',',4)[4].split('.',1)[0] for x in data['Strategy_Formula']),index=data.index,name='Option')
	#strike = pd.Series((x.split(',',4)[4].split('.',1)[1] for x in data['Strategy_Formula']),index=data.index,name='Strike')
	buy_sell = pd.Series((x.split(',',4)[0].split('.',1)[0] for x in data['Strategy_Formula']),index=data.index,name='Buy/Sell')
	data = pd.concat([data,buy_sell],axis=1)
	return data	


def price_matrixer(symbs,types,root):
	for symb in symbs:
		dfs = os.listdir(root+symb+'\\')
		price_matrices = root+'Price_Matrices\\'
		os.makedirs(price_matrices,exist_ok=True)
		for type in types:
			datae = [df for df in dfs if type in df]
			data = pd.DataFrame()
			for df in datae:		
				dat = pd.read_csv(root+symb+'\\'+df,'rb',delimiter=',',parse_dates=['expiration','quote_date'],infer_datetime_format=True)
				data = pd.concat([data,dat],axis=0)
			underlying = pd.Series(data=np.round(((data['underlying_bid_eod']+data['underlying_ask_eod'])/2),decimals=0),index=data.index,name='underlying..')
			stx = pd.Series(data=(data['strike'] - underlying),index=data.index,name='stx')
			close = pd.Series(data=((data['bid_eod']+data['ask_eod'])/2),index=data.index,name='close..')
			d2x = pd.Series(data=((data['expiration']-data['quote_date'])/np.timedelta64(1, 'D')).astype(int), index=data.index,name='d2x')
			data = pd.concat([data,underlying,stx,d2x,close],axis=1)
			price_matrix = pd.DataFrame()
			for strike in np.arange(37)-18:
				price_curve = pd.Series(index=np.arange(30)+1,name=str(strike))
				for days in np.arange(31):
					day_price = data['close..'][(d2x==days)&(stx==strike.astype('float64'))].mean()
					price_curve.set_value(days,day_price)
				price_matrix = pd.concat([price_matrix,price_curve],axis=1)
			x = price_matrix.fillna(method='backfill',axis=0)
			y = price_matrix.fillna(method='ffill',axis=0)
			price_matrix = (x+y)/2
			#price_matrix.to_csv(root+'Data\\Price_Matrices\\'+symb+'_'+type+'_price_matrix.csv',mode='w',index=True)	
	return price_matrix
	
def closest(value, myList):
	myList = map(float,myList)
	idx = min(myList, key=lambda x:np.abs(x-value))
	return idx
	
class dicts:
	def __init__(self,direct):
		self.direct = direct
		
	def stk_dict(self,symb):
		dict = load_pickle(self.direct.dict_dir + 'stock_dictionary.pkl')
		stk = dict[symb]
		return stk
		
	def all_dict(self):
		dict = load_pickle(self.direct.dict_dir + 'stock_dictionary.pkl')
		return dict
		
	def opt_dict(self):
		dict = load_pickle(self.direct.dict_dir + 'option_dictionary.pkl')
		return dict
	
	def dnc_dict(self):
		dict = load_pickle(self.direct.dict_dir + 'dnc_dictionary.pkl')
		return dict
	
	def dtree_dicts(self,symb):
		master_tree = load_pickle(self.direct.forest_dir + self.symb + '\\' + 'master_crit_'+self.symb+'_dictionary.pkl')
		master_imp = load_pickle(self.direct.forest_dir + self.symb + '\\' + 'master_imp_'+self.symb+'_dictionary.pkl')
		master_val = load_pickle(self.direct.forest_dir + self.symb + '\\' + 'master_val_'+self.symb+'_dictionary.pkl')
		return master_tree, master_imp, master_val
		
	
############################################### CLASSES ##################################################
class dirs:
	'''Directories for all MT objects'''
	def __init__(self,root,user,initiation):
		self.user = user
		#Mt Auto Reporting Dirs
		
		self.dc_dir = root + 'Data_&_Calcs\\'
		self.sort_par_dir = root + 'Sorts_Reports\\'
		self.presort_dir = self.sort_par_dir + 'Presorts\\'
		self.sort_dir = self.sort_par_dir + 'Sorts\\'
		self.lcr_dir = root + 'Level_Class_Reports\\'
		self.opt_dir = root + 'Option_Data\\'
		self.log_dir = root + 'Logs\\'
		self.tnp_dir = root + 'Trades_&_Plays\\'
		self.data_dir = root + 'Data\\'
		self.opt_data_dir = self.data_dir + 'Option_Data\\'
		self.pm_dir = self.data_dir + 'Price_Matrices\\'
		if initiation == 1:			
			os.makedirs(self.dc_dir, exist_ok=True)			
			os.makedirs(self.sort_par_dir, exist_ok=True)			
			os.makedirs(self.presort_dir, exist_ok=True)			
			os.makedirs(self.sort_dir, exist_ok=True)			
			os.makedirs(self.lcr_dir, exist_ok=True)			
			os.makedirs(self.opt_dir, exist_ok=True)			
			os.makedirs(self.log_dir, exist_ok=True)			
			os.makedirs(self.tnp_dir, exist_ok=True)			
			os.makedirs(self.data_dir, exist_ok=True)			
			os.makedirs(self.opt_data_dir, exist_ok=True)			
			os.makedirs(self.pm_dir, exist_ok=True)
		#Data Base Dirs
		self.db_root = 'C:\\Users\\' + user + '\\Dropbox (marketrader)\\'
		self.cboe_dir = self.db_root + 'CBOE\\'
		os.makedirs(self.cboe_dir, exist_ok=True)
		self.dict_dir = self.db_root + 'Data\\Dictionaries\\'
		os.makedirs(self.dict_dir, exist_ok=True)
		self.forest_dir = self.dict_dir + 'Forest_Dicts\\'
		os.makedirs(self.forest_dir, exist_ok=True)		
		self.db_quant = self.db_root + 'Quant\\'
		os.makedirs(self.db_quant, exist_ok=True)
		self.db_dir = self.db_quant + 'Database\\'
		os.makedirs(self.db_dir, exist_ok=True)
		self.db_log_dir = self.db_quant + 'Logs\\'
		os.makedirs(self.db_log_dir, exist_ok=True)
		
		
	def paths(self,symb,d):
		presort_path = self.presort_dir + d.strftime('%Y-%m-%d') + '\\' + symb + '\\'
		os.makedirs(presort_path, exist_ok=True)
		sorttarget_path = self.sort_dir + d.strftime('%Y-%m-%d') + '\\' + symb +'\\Sorts\\'
		os.makedirs(sorttarget_path, exist_ok=True)
		datatarget_path = self.sort_dir + d.strftime('%Y-%m-%d') + '\\' + symb + '\\Data\\'
		os.makedirs(datatarget_path, exist_ok=True)
		return presort_path, datatarget_path, sorttarget_path
	def tree_paths(self,symb):
		presort_path = self.presort_dir + 'TreeSorts\\' + symb + '\\'
		os.makedirs(presort_path, exist_ok=True)
		sorttarget_path = self.sort_dir + 'TreeSorts\\' + symb +'\\Sorts\\'
		os.makedirs(sorttarget_path, exist_ok=True)
		datatarget_path = self.sort_dir + 'TreeSorts\\' + symb + '\\Data\\'
		os.makedirs(datatarget_path, exist_ok=True)
		return presort_path, datatarget_path, sorttarget_path
		
class logger:


	def __init__(self,direct):
		self.direct = direct
	
	def lcr_logger(self,d,data,name):
		lcr_dir = self.direct.lcr_dir
		if os.isfile(lcr_dir+name):
			data.to_csv(lcr_dir+name,mode='a',index=False)
		else:
			data.to_csv(lcr_dir+name,mode='w',index=False,headers=lcrcolumns)
		return self
		
class price_matrix:
	''' Price Matrices '''
	types = ('Puts','Calls')
	def __init__(self,symbs,direct):
		self.symbs = symbs
		self.direct = direct
		
	def _opt_p_matrixer(self):
		for symb in self.symbs:
			dfs = os.listdir(self.direct.data_dir+symb+'\\')
			stk = dicts(self.direct).stk_dict(symb)
			pm_dir = self.direct.pm_dir
			stk_idx = self.stk_idx_dict(stk.ix[:dt.datetime.today()-dt.timedelta(days=10)])
			for type in self.types:
				datae = [df for df in dfs if type in df]
				data = pd.DataFrame()
				
				for df in datae:		
					dat = pd.read_csv(self.direct.data_dir+symb+'\\'+df,'rb',delimiter=',',parse_dates=['expiration','quote_date'],infer_datetime_format=True)
					data = pd.concat([data,dat],axis=0)
					data = data[data['expiration']<(dt.datetime.today()-dt.timedelta(days=20))]
					print(df)
				underlying = pd.Series(data=np.round(((data['underlying_bid_eod']+data['underlying_ask_eod'])/2),decimals=0),index=data.index,name='underlying..')
				stx = pd.Series(data=(data['strike'] - underlying),index=data.index,name='stx')
				close = pd.Series(data=((data['bid_eod']+data['ask_eod'])/2),index=data.index,name='close..')
				print('Calculating Differences..')
				d2x = pd.Series(data=((data['expiration']-data['quote_date'])/np.timedelta64(1, 'D')).astype(int),index=data.index,name='d2x')#pd.Series(data=self.date_differs(data['expiration'],data['quote_date'],stk_idx), index=data.index,name='d2x') #((-)/np.timedelta64(1, 'D')).astype(int)
				print(list(d2x.unique()))
				data = pd.concat([data,underlying,stx,d2x,close],axis=1)
				price_matrix = pd.DataFrame()
				print('constructing matrix...')
				for strike in np.arange(37)-18:
					price_curve = pd.Series(index=np.arange(30)+1,name=str(strike))
					for days in np.arange(31):
						day_price = data['close..'][(data['d2x']==days)&(data['stx']==strike.astype('float64'))].mean()
						price_curve.set_value(days,day_price)
					price_matrix = pd.concat([price_matrix,price_curve],axis=1)
				x = price_matrix.fillna(method='backfill',axis=0)
				y = price_matrix.fillna(method='ffill',axis=0)
				price_matrix = (x+y)/2
				price_matrix.to_csv(self.direct.pm_dir+symb+'_'+type+'_price_matrix.csv',mode='w',index=True)
		return self
		
	def _bs_price_matricer(self):
		for symb in self.symbs:
			stk = dicts(self.direct).stk_dict(symb)
			v = np.std(stk.iloc[-252:]['Close'])/np.mean(stk.iloc[-252:]['Close'])
			s = stk.iloc[-1]['Close']
			rf = .03
			div = 0 
			for typ in self.types:
				if typ == 'Calls':
					cp = 1
				elif typ == 'Puts':
					cp = -1
				price_matrix = pd.DataFrame()
				for strike in np.arange(37)-18:
					k = ((100+strike)/100)*s
					price_curve = pd.Series(index=np.arange(30)+1,name=str(strike))
					for days in np.arange(51):
						day_price = self.black_scholes(s,k,days/365,v,rf,div,cp)
						price_curve.set_value(days,day_price)
					price_matrix = pd.concat([price_matrix,price_curve],axis=1)
				price_matrix.to_csv(self.direct.pm_dir+symb+'_'+typ+'_price_matrix.csv',mode='w',index=True)
		return self
	
	def black_scholes(self,s,k,t,v,rf,div,cp):
		''' Price an option using the Black-Scholes model.		
		s : initial stock price
		k : strike price
		t : expiration time in years
		v : volatility
		rf : risk-free rate
		div : dividend
		cp : +1/-1  for call/put
		'''
		d1 = (math.log(s/k)+(rf-div+0.5*math.pow(v,2))*t)/(v*math.sqrt(t))
		d2 = d1 - v*math.sqrt(t)
		optprice = cp*s*math.exp(-div*t)*stats.norm.cdf(cp*d1) - \
			cp*k*math.exp(-rf*t)*stats.norm.cdf(cp*d2)
		if optprice == np.nan:
			print('v=',v,' sqrt(t)=', math.sqrt(t))
		return optprice
		
	def pms(self,symb):				
		pms = {}
		for type in price_matrix.types:
			pms[symb+'_'+type] = pd.read_csv(self.direct.pm_dir+symb+'_'+type+'_price_matrix.csv','rb',delimiter=',').set_index(keys=['d2x'],drop=True)
		return pms
		
		
	
		
	def date_differ(self,series1,series2,idx):
		exp = pd.Series(index=series1.index)
		qqq = pd.Series(index=series1.index)
		for d in series1.unique():
			d = pd.to_datetime(d,infer_datetime_format=True)
			exp[series1 == d] = idx.get_loc(key=d,method='bfill')
		for d in series2.unique():
			d = pd.to_datetime(d,infer_datetime_format=True)
			qqq[series2 == d] = idx.get_loc(key=d,method='bfill')
		diffs = pd.Series(data=(exp-qqq),index=series1.index,name='diffs')
		return diffs

	def date_differs(self,series1,series2,stk_idx):
		exp = pd.Series(index=series1.index)
		qqq = pd.Series(index=series1.index)
		for d in series1.unique():
			d = pd.to_datetime(d,infer_datetime_format=True)
			exp[series1 == d] = stk_idx[d]
		for d in series2.unique():
			d = pd.to_datetime(d,infer_datetime_format=True)
			qqq[series2 == d] = stk_idx[d]
		diffs = pd.Series(data=(exp-qqq),index=series1.index,name='diffs')
		return diffs
		
		
	def stk_idx_dict(self,stk):
		idx_dict = {stk.index[i]:i for i in np.arange(len(stk))}		
		sats = self.all_sats(stk.index[0])
		for sat in sats:
			try:
				idx_dict[sat] = idx_dict[sat-dt.timedelta(days=1)]
			except:
				try:
					idx_dict[sat] = idx_dict[sat-dt.timedelta(days=2)]
				except:
					continue
		return idx_dict
		  
	def all_sats(self,date):
		length = ((dt.datetime.today()-dt.timedelta(days=20)) - date).days
		d_range = pd.date_range(date, periods=length, freq='D')
		sats = pd.Series(data=[d.weekday() for d in d_range])
		sats = d_range[sats==5]
		return sats
		
		
class option_pricer:
	types = ('Puts','Calls')
	def __init__(self,symb,direct):
		if symb == '^VIX':
			symb == 'VXX'
		self.symb = symb		
		self.direct = direct
		self.pmx = price_matrix((self.symb,),self.direct).pms(symb)
		self.stk_dict = load_pickle(direct.dict_dir + 'stock_dictionary.pkl')
		try:
			self.stk = self.stk_dict[self.symb] #pd.read_csv(self.direct.db_dir+self.symb+'\\Stock\\'+self.symb+'.csv','rb',delimiter=',',parse_dates=['Date'],infer_datetime_format=True).set_index(keys='Date')
		except:			
			self.stk = web.DataReader(self.symb,'google',pd.to_datetime('1980-01-02', infer_datetime_format=True),dt.datetime.today())
			
	def single_option_price_estimator(self,i,d2,d2x,typ,idx):
		#print(d2,d2x)
		stk = self.stk[d2:d2+dt.timedelta(days=40)]
		closes = pd.Series(data=np.round(((stk['Close']/stk.ix[d2]['Close'])-1)*100,decimals=0),index=stk.index).reset_index(drop=True)
		#print(closes)#.reset_index(drop=True)
		#diffs = pd.Series(data=(((stk.index[j]-stk.index[0])/np.timedelta64(1, 'D')).astype(int) for j in idx),index=array(idx))
		pr = pd.Series(index=idx)
		if typ == 'Calls':
			t = 2
		elif typ == 'Puts':
			t = -2
		pms = self.pmx[self.symb+'_'+typ]
		d2_price = pms.ix[d2x][str(closest(t,pms.columns))]
		
		for j in idx:		
			pr.ix[j] = pms.ix[(d2x+2) - j][str(closest(t - closes[j-2],pms.columns))]/d2_price
			#print(t - closes[j-2],d2_price)
			#print(pr.ix[j])
			price_ratios = pr.T		
		return price_ratios

	def option_price_estimator(self,data):
		idx = [2,3,4,5,6,7,8,9,10,15]
		#print(self.stk.index[:5])
		exps = pd.Series(data=(map(next_monthly,data.index)),index=data.index)
		d2s = pd.Series(data=[self.stk.index[pd.Index(self.stk.index).get_loc(str(i.date()))+1] for i in data.index],index=data.index,name='d2s')
		#d2s = d2s[list(data.index)]
		d2xs = pd.Series(data=date_diff(d2s,exps),index=data.index,name='d2xs')
		dedo = pd.concat([d2s,d2xs],axis=1)
		price_data = pd.DataFrame(index=data.index,columns=idx)
		prices = {}
		for typ in self.types:
			for i in data.index:
				price_data.ix[i] = option_pricer.single_option_price_estimator(self=self,i=i,d2=d2s.ix[i],d2x=d2xs.ix[i],typ=typ,idx=idx)
			prices[typ] = price_data   #map(option_pricer.single_option_price_estimator,d2s,d2xs,typ)
		calls = pd.DataFrame(prices['Calls'],index=data.index)
		puts = pd.DataFrame(prices['Puts'],index=data.index)
		return calls, puts
		
	def bs_opt_pricer(self,data,stk):
		exps = pd.Series(data=(map(next_monthly,data.index)),index=data.index)
		d2s = pd.Series(data=[stk.index[pd.Index(stk.index).get_loc(str(i.date()))+1] for i in data.index],index=data.index,name='d2s')				
		prices = {}
		for typ in types:
			price_data = pd.DataFrame(index=data.index,columns=np.arange(19)+2)
			for i in data.index:
				price_data.ix[i] = single_bs_price_calc(stk=stk,d=i,d2=d2s.ix[i],exp=exps.ix[i],typ=typ)
			prices[typ] = price_data   #map(option_pricer.single_option_price_estimator,d2s,d2xs,typ)
		calls = pd.DataFrame(prices['Calls'],index=data.index)
		puts = pd.DataFrame(prices['Puts'],index=data.index)		
		return calls, puts
		
	def single_bs_price_calc(stk,d,d2,exp,typ): #Need to determine why it's producing NAN
		st = stk[stk.index[stk.index.get_loc(str(d2))-252]:d2]
		v = (np.std(st['Close'])/np.mean(st['Close']))*2	 
		if typ == 'Calls':
			buy = stk.loc[d2]['Low']
			data = stk.ix[d2:stk.index[stk.index.get_loc(d2)+18]]['High']
			cp = 1
			strike = np.round(stk.loc[d2]['Open']*1.02,decimals=0)
		elif typ == 'Puts':
			buy = stk.loc[d2]['High']
			data = stk.ix[d2:stk.index[stk.index.get_loc(d2)+18]]['Low']
			cp = -1	
			strike = np.round(stk.loc[d2]['Open']*.98,decimals=0)
		rf = .03
		div = 0 
		prices = pd.Series(index=np.arange(20)+1,name=d)
		d2op = black_scholes(buy,strike,(exp-d2).days/365,v,rf,div,cp)
		prices.iloc[1] = (black_scholes(s,strike,t,v,rf,div,cp))/d2op
		for i in data.index:
			s = data[i]
			t = (exp-i).days/365
			if t <= 0:
				break
			j = len(data.ix[:i])-1
			
			prices.iloc[j] = black_scholes(s,strike,t,v,rf,div,cp)
		prices = prices/d2op
		return prices
		
class stock:
	'''All data and methods for MT-Auto w/ input of stock OHLCV data	
	Features:
		Stock Data (OHLCV)
	'''
	

	def __init__(self, symb,direct):
		self.symb = symb
		self.direct = direct
		
	def load_opt_data(self, years):
		types = ['Calls','Puts']
		opt_data = pd.DataFrame()
		for typ in types:
			for year in years:
				datae = [df for df in dfs if year in df]
				data = pd.read_csv(opt_data_dir+self.symbol+'_'+str(year)+typ+'.csv','rb',delimiter=',')
				opt_data = pd.concat([opt_data,data],axis=0)
		return opt_data
		

	@staticmethod	
	def load_pmx(symb):
		pm_root = root + 'marketrader Team Folder\\Data\\'
		types = ('Puts','Calls')				
		pmx = {}
		for type in types:
			for symb in symbs:
				pmx[symb+'_'+type] = pd.read_csv(pm_root+'Price_Matrices\\'+symb+'_'+type+'_price_matrix.csv','rb',delimiter=',')
		return pmx
			
	def stck(self):
		try:
			stk = dicts(self.direct).stk_dict(self.symb)
		except:
			strt = pd.to_datetime('1980-01-02', infer_datetime_format=True)
			stk = web.DataReader(self.symb, 'google', strt, dt.datetime.today())
		return stk
		
	def dnc(self,after_market_update):

		stk = self.stck()
		if after_market_update == True:
			stk = stock_updater(self.symb,stk)
		stk = forw_targ(stk)
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
		stk = count_consec(stk,'BLOWEST_20')
		stk = LLOWEST(stk, 5)
		stk = HLOWEST(stk, 5)
		stk = BLOWEST(stk, 5)
		stk = HLLOWESTIN(stk, 5)
		stk = count_consec(stk,'BLOWEST_5')
		stk = ANNLOW(stk)
		stk = ANNLOWDEFLAT(stk)
		stk = CCI(stk, 14)
		stk = CCI(stk, 40)
		stk = CCI(stk, 89)
		#stk = CCI_DIVERG_T(stk,14)
		#stk = CCI_DIVERG_F(stk,14)
		#stk = CCI_DIVERG_T(stk,40)
		#stk = CCI_DIVERG_F(stk,40)
		#stk = CCI_DIVERG_S(stk,40)
		stk = anchored_divergence_bool(data=stk,col='CCI_40',r=30,diff=40,cutoff=.2,offset=0)
		stk = anchored_divergence_bool(data=stk,col='CCI_14',r=30,diff=10,cutoff=.2,offset=0)
		stk = RSI(stk,20)
		stk = RSI(stk,40)
		stk = RSI(stk,89)
		stk = anchored_divergence_bool(data=stk,col='RSI_40',r=30,diff=40,cutoff=.2,offset=50)
		stk = anchored_divergence_bool(data=stk,col='RSI_20',r=30,diff=40,cutoff=.2,offset=50)
		stk = SMAs(stk,'Typical')
		stk = EMAs(stk,'Typical')
		stk = MACD(data=stk,nday1=13,nday2=27,sign=8)
		stk = anchored_divergence_bool(data=stk,col='MACD_13-27',r=20,diff=.1,cutoff=.2,offset=0)
		stk = SMA_slope(data=stk,sma='SMA_5',ndays=3)
		stk = direx(stk,[5,10,20,50,100])
		stk.to_csv(self.direct.dc_dir + self.symb + '_Data_&_Calcs.csv',mode='w')
		return stk	
			
	def stk_normalizer(self, data):
		stk_norm = pd.DataFrame(index=data.index)	
		stk_types = data.columns.to_series().groupby(data.dtypes).groups
		stk_types = {k.name: v for k, v in stk_types.items()}
		for key in stk_types.keys():
			stk_type = stk_set[stk_types[key]]
			if key == 'bool':
				stk_type = normalizer_bool(stk_type)					
			elif (key == 'float64') or (key == 'int64'):
				stk_type = normalizer_float(stk_type)
				p = len(stk_type.columns)
			else:
				stk_type = pd.DataFrame()
			stk_norm = stk_norm.join(stk_type)		
		return stk_norm
		
	def ptree_sorter(self,data):
		final_tree = self.tree_orderer()
		ptree_sorts = {}
		for s in final_tree.keys():
			criteria = final_tree[s]
			print(criteria)
			sort = data
			for c in criteria.keys():
				print(criteria[c])
				if criteria[c][1] == 1:
					sort = sort[sort[criteria[c][0]] > criteria[c][2]]
				elif criteria[c][1] == -1:
					sort = sort[sort[criteria[c][0]] <= criteria[c][2]]
			ptree_sorts[s] = sort
		return ptree_sorts
	
	def remove_duplicates_dict(self,dict):
		seen = []
		new_dict = {}
		for k,val in dict.items():
			if not val in seen:
				seen.append(val)
				new_dict[k] = val				
		return dict
		
	def tree_orderer(self):
		master_tree = load_pickle(self.direct.forest_dir + self.symb + '\\' + 'master_crit_'+self.symb+'_dictionary.pkl')
		master_imp = load_pickle(self.direct.forest_dir + self.symb + '\\' + 'master_imp_'+self.symb+'_dictionary.pkl')
		master_val = load_pickle(self.direct.forest_dir + self.symb + '\\' + 'master_val_'+self.symb+'_dictionary.pkl')
		final_tree = {}
		#master_tree = self.remove_duplicates_dict(master_tree)
		#imp_list = sorted(master_im, key=master_imp.get, reverse=False)# order master_imp from lowest to highest set ordered keys as imp_list
		for i in master_tree.keys():
			#print(master_val[t][0])
			if (master_val[i][0].max() != master_val[i][0][1]):
				final_tree[i] = master_tree[i]
		return final_tree
				
		
	def _presorter(self,symb,data,d,sorts):

		presort_path,a,b = self.direct.paths(symb,d)
		#if data.index[250] > d:
			#continue
		stock = data[:d]
		#Sort 1
		if 'sort_1' in sorts:
			if stock.ix[-1]['D1_HI_LO_LOWEST_IN_20'] == True:
				sort1 = stock[stock['D1_HI_LO_LOWEST_IN_20'] == True]
			else:
				sort1 = []
		#Sort 2
		if 'sort_2' in sorts:
			if stock.ix[-1]['ANN_LO_BY_DEFLAT_LOWEST'] == True:
				sort2 = stock[stock['ANN_LO_BY_DEFLAT_LOWEST'] == True]
			else:
				sort2 = []
		#Sort 3 - I want to try this sort with slightly different criteria - i.e. to take only values that are within a .5 range regardless of their proximity to the edges. 
		if 'sort_3' in sorts:
			if (stock.ix[-1]['%K_STO'] <= 95) & (stock.ix[-1]['%K_STO'] >= 5):
				sort3a = stock[((stock['%K_STO'] < (stock.ix[-1]['%K_STO'] + 5)) & (stock['%K_STO'] > (stock.ix[-1]['%K_STO'] - 5)))]
			elif stock.ix[-1]['%K_STO'] < 5:
				sort3a = stock[stock['%K_STO'] < 10]
			elif stock.ix[-1]['%K_STO'] > 95:
				sort3a = stock[stock['%K_STO'] > 90]
			if 	stock.ix[-1]['D1_DIR_STO'] >= 0:
				sort3b = sort3a[sort3a['D1_DIR_STO'] >= 0]
			elif stock.ix[-1]['D1_DIR_STO'] < 0:
				sort3b = sort3a[sort3a['D1_DIR_STO'] < 0]
			if 	stock.ix[-1]['D1_FAUXSTO'] >= 0:
				sort3 = sort3b[sort3b['D1_FAUXSTO'] >= 0]
			elif stock.ix[-1]['D1_FAUXSTO'] < 0:
				sort3 = sort3b[sort3b['D1_FAUXSTO'] < 0]
			else:
				sort3 = []

		#Sort 4 - 'ABSOL_UP_D1_STO'
		if 'sort_4' in sorts:
			if stock.ix[-1]['ABSOL_UP_D1_STO'] == True:
				sort4a = stock[stock['ABSOL_UP_D1_STO'] == True]
				if (stock.ix[-1]['%K_STO'] <= 95) & (stock.ix[-1]['%K_STO'] >= 5):
					sort4 = sort4a[((sort4a['%K_STO'] < (stock.ix[-1]['%K_STO'] + 5)) & (sort4a['%K_STO'] > (stock.ix[-1]['%K_STO'] - 5)))]
				elif stock.ix[-1]['%K_STO'] < 5:
					sort4 = sort4a[sort4a['%K_STO'] < 10]
				elif stock.ix[-1]['%K_STO'] > 95:
					sort4 = sort4a[sort4a['%K_STO'] > 90]
			else:
				sort4 = []
		#Sort 5 Duplicate
	
		#Sort 6 Uses D2 values
		#Sort 7 Uses D2 values
		
		
		#Sort 8
		if 'sort_8' in sorts:
			if stock.ix[-1]['FIRST_UP_D1_%K_3PER_SLP_STO'] == True:
				sort8a = stock[stock['FIRST_UP_D1_%K_3PER_SLP_STO'] == True]
				if (stock.ix[-1]['%K_STO'] <= 95) & (stock.ix[-1]['%K_STO'] >= 5):
					sort8 = sort8a[((sort8a['%K_STO'] < (stock.ix[-1]['%K_STO'] + 5)) & (sort8a['%K_STO'] > (stock.ix[-1]['%K_STO'] - 5)))]
				elif stock.ix[-1]['%K_STO'] < 5:
					sort8 = sort8a[sort8a['%K_STO'] < 10]
				elif stock.ix[-1]['%K_STO'] > 95:
					sort8 = sort8a[sort8a['%K_STO'] > 90]
			else:
				sort8 = []
		#Sort 9
		if 'sort_9' in sorts:
			if stock.ix[-1]['FIRST_DOWN_D1_%K_3PER_SLP_STO'] == True:
				sort9a = stock[stock['FIRST_DOWN_D1_%K_3PER_SLP_STO'] == True]
				if (stock.ix[-1]['%K_STO'] <= 95) & (stock.ix[-1]['%K_STO'] >= 5):
					sort9 = sort9a[((sort9a['%K_STO'] < (stock.ix[-1]['%K_STO'] + 5)) & (sort9a['%K_STO'] > (stock.ix[-1]['%K_STO'] - 5)))]
				elif stock.ix[-1]['%K_STO'] < 5:
					sort9 = sort9a[sort9a['%K_STO'] < 10]
				elif stock.ix[-1]['%K_STO'] > 95:
					sort9 = sort9a[sort9a['%K_STO'] > 90]
			else:
				sort9 = []
		#Sort 10 ? D1_H_LO_LOWEST_IN_20

		#Sort 14 #needs explanation long formula
		#Sort 15 #needs explanation long formula
		#Sort 16 - Generate SMA Tests #Needs ufunc to be defined SMA_5 > than all SMA_10-50 and D-1_SMA_5 
		#Sort 17 #needs explanation long formula
		#Sort 18 - # of days with trend up in GH (D&C)
		#Sort 19 #needs explanation long formula
		#Sort 20
		if 'sort_20' in sorts:
			if (stock.ix[-1]['COUNT_BLOWEST_5'] > 0) & (stock.ix[-1]['D1_HI_LO_LOWEST_IN_5'] == True):
				sort20 = stock[(stock['COUNT_BLOWEST_5'] <= stock.ix[-1]['COUNT_BLOWEST_5'] + 1) & (stock['COUNT_BLOWEST_5'] >= stock.ix[-1]['COUNT_BLOWEST_5'] - 1)]
			else: 
				sort20 = []
		#Sort 21
		#Sort 22
		if 'sort_22' in sorts:
			if (stock.ix[-1]['COUNT_BLOWEST_20'] > 0)  & (stock.ix[-1]['D1_HI_LO_LOWEST_IN_20'] == True):
				sort22 = stock[(stock['COUNT_BLOWEST_20'] <= stock.ix[-1]['COUNT_BLOWEST_20'] + 1) & (stock['COUNT_BLOWEST_20'] >= stock.ix[-1]['COUNT_BLOWEST_20'] - 1)]
			else: 
				sort22 = []
		#Sort 23
		#Sort 24 # What is MS? 
		#Sort 25 # BBPinch & Expand with Sort 3
		if 'sort_25' in sorts:
			if stock.ix[-1]['BBANDS_PINCH'] == True:
				sort25a = stock[stock['BBANDS_PINCH'] == True]
				if (stock.ix[-1]['%K_STO'] <= 95) & (stock.ix[-1]['%K_STO'] >= 5):
					sort25a = stock[((stock['%K_STO'] < (stock.ix[-1]['%K_STO'] + 5)) & (stock['%K_STO'] > (stock.ix[-1]['%K_STO'] - 5)))]
				elif stock.ix[-1]['%K_STO'] < 5:
					sort25a = stock[stock['%K_STO'] < 10]
				elif stock.ix[-1]['%K_STO'] > 95:
					sort25a = stock[stock['%K_STO'] > 90]
				if 	int(stock.ix[-1]['D1_DIR_STO']) >= 0:
					sort25b = sort25a[sort25a['D1_DIR_STO'] >= 0]
				elif int(stock.ix[-1]['D1_DIR_STO']) < 0:
					sort25b = sort25a[sort25a['D1_DIR_STO'] < 0]
				if 	int(stock.ix[-1]['D1_FAUXSTO']) >= 0:
					sort25 = sort25b[sort25b['D1_FAUXSTO'] >= 0]
				elif int(stock.ix[-1]['D1_FAUXSTO']) < 0:
					sort25 = sort25b[sort25b['D1_FAUXSTO'] < 0]
			else:
				sort25 = []




		if len(sort1) > 0:
			sort1.to_csv(presort_path + symb + '_sort-01_' + d.strftime('%Y-%m-%d') + '.csv')
		if len(sort2) > 0:
			sort2.to_csv(presort_path + symb + '_sort-02_' + d.strftime('%Y-%m-%d') + '.csv')
		if len(sort3) > 0:
			sort3.to_csv(presort_path + symb + '_sort-03_' + d.strftime('%Y-%m-%d') + '.csv')
		if len(sort4) > 0:
			sort4.to_csv(presort_path + symb + '_sort-04_' + d.strftime('%Y-%m-%d') + '.csv')
		if len(sort8) > 0:
			sort8.to_csv(presort_path + symb + '_sort-08_' + d.strftime('%Y-%m-%d') + '.csv')
		if len(sort9) > 0:
			sort9.to_csv(presort_path + symb + '_sort-09_' + d.strftime('%Y-%m-%d') + '.csv')
		if len(sort20) > 0:
			sort20.to_csv(presort_path + symb + '_sort-20_' + d.strftime('%Y-%m-%d') + '.csv')
		if len(sort22) > 0:
			sort22.to_csv(presort_path + symb + '_sort-22_' + d.strftime('%Y-%m-%d') + '.csv')
		if len(sort25) > 0:
			sort25.to_csv(presort_path + symb + '_sort-25_' + d.strftime('%Y-%m-%d') + '.csv')

		return data

	def clusterer(self,data,d):			
		div_count_cols = [col for col in data.columns if 'Div_Count' in col]
		for col in div_count_cols:			
			data[col][data[col]<0] = 0
		indicators = ['CCI','RSI','MACD','SMA','EMA','BBAND','STO','Candle','LOWEST']
		indic_dict = {}
		for indic in indicators:
			indic_cols = [col for col in stk_dnc_dict[symb].columns if indic in col]
			indic_dict[indic] = indic_cols
		for symb in symbs:
			for d in dets:
				stk_data = stk_dnc_dict[symb][stk_dnc_dict[symb].index[50]:d]				
				for k in indic_dict.keys():
					if 'BBAND' in k:
						stk_set = stk_data[indic_dict[k]]
					else:
						stk_set = stk_data[indic_dict[k] + indic_dict['BBAND']]
					stk_norm = pd.DataFrame(index=stk_set.index)	
					stk_types = stk_set.columns.to_series().groupby(stk_set.dtypes).groups
					stk_types = {k.name: v for k, v in stk_types.items()}
					for key in stk_types.keys():
						stk_type = stk_set[stk_types[key]]
						if key == 'bool':
							stk_type = normalizer_bool(stk_type)					
						elif (key == 'float64') or (key == 'int64'):
							stk_type = normalizer_float(stk_type)
							p = len(stk_type.columns)
						else:
							stk_type = pd.DataFrame()
						stk_norm = stk_norm.join(stk_type)		
					cluster = simple_cut(stk_norm,p)			
					stk_norm = stk_data[stk_data.index.isin(cluster.index)]
					stk_norm = stk_norm.join(pd.Series(data=cluster, index=stk_norm.index,name='Diffs'))
		return stk_norm
	
	@classmethod
	def _get_mstd(cls, data, column, windows):
		""" get moving standard deviation
		:param df: data
		:param column: column to calculate
		:param windows: collection of window of moving standard deviation
		:return: None
		"""
		window = cls.get_only_one_positive_int(windows)
		column_name = '{}_{}_m_std'.format(column, window)
		data[column_name] = data[column].rolling(min_periods=1, window=window,
						     center=False).std()
		return data

	@classmethod
	def _get_mvar(cls, data, column, windows):
		""" get moving variance
		:param df: data
		:param column: column to calculate
		:param windows: collection of window of moving variance
		:return: None
		"""
		window = cls.get_only_one_positive_int(windows)
		column_name = '{}_{}_m_var'.format(column, window)
		data[column_name] = data[column].rolling(
			min_periods=1, window=window, center=False).var()
		return data
	
	@classmethod
	def _get_mdm(cls, data, windows):
		""" -DM, negative directional moving accumulation
		If window is not 1, return the SMA of -DM.
		:param df: data
		:param windows: range
		:return:
		"""
		window = cls.get_only_one_positive_int(windows)
		column_name = 'mdm_{}'.format(window)
		um, dm = data['um'], data['dm']
		data['mdm'] = np.where(dm > um, dm, 0)
		if window > 1:
		mdm = data['mdm_{}_ema'.format(window)]
		else:
		mdm = data['mdm']
		data[column_name] = mdm
		return data

	@classmethod
	def _get_pdi(cls, data, windows):
		""" +DI, positive directional moving index
		:param df: data
		:param windows: range
		:return:
		"""
		window = cls.get_only_one_positive_int(windows)
		pdm_column = 'pdm_{}'.format(window)
		tr_column = 'atr_{}'.format(window)
		pdi_column = 'pdi_{}'.format(window)
		data[pdi_column] = data[pdm_column] / data[tr_column] * 100
		return data[pdi_column]

	@classmethod
	def _get_mdi(cls, data, windows):
		window = cls.get_only_one_positive_int(windows)
		mdm_column = 'mdm_{}'.format(window)
		tr_column = 'atr_{}'.format(window)
		mdi_column = 'mdi_{}'.format(window)
		data[mdi_column] = data[mdm_column] / data[tr_column] * 100
		return data[mdi_column]

	@classmethod
	def _get_dx(cls, df, windows):
		window = cls.get_only_one_positive_int(windows)
		dx_column = 'dx_{}'.format(window)
		mdi_column = 'mdi_{}'.format(window)
		pdi_column = 'pdi_{}'.format(window)
		mdi, pdi = data[mdi_column], data[pdi_column]
		data[dx_column] = abs(pdi - mdi) / (pdi + mdi) * 100
		return data[dx_column]
	
	@classmethod
	def _get_tr(cls, data):
		""" True Range of the trading
		tr = max[(high - low), abs(high - close_prev), abs(low - close_prev)]
		:param df: data
		:return: None
		"""
		prev_close = data['close_-1_s']
		high = data['high']
		low = data['low']
		c1 = high - low
		c2 = np.abs(high - prev_close)
		c3 = np.abs(low - prev_close)
		data['tr'] = np.max((c1, c2, c3), axis=0)

	@classmethod
	def _get_atr(cls, data, window=None):
		""" Average True Range
		The average true range is an N-day smoothed moving average (SMMA) of
		the true range values.  Default to 14 days.
		https://en.wikipedia.org/wiki/Average_true_range
		:param df: data
		:return: None
		"""
		if window is None:
			window = 14
			column_name = 'atr'
		else:
			window = int(window)
			column_name = 'atr_{}'.format(window)
		tr_smma_column = 'tr_{}_smma'.format(window)
		data[column_name] = data[tr_smma_column]
		del data[tr_smma_column]

	@classmethod
	def _get_dma(cls, data):
		""" Different of Moving Average
		default to 10 and 50.
		:param df: data
		:return: None
		"""
		data['dma'] = data['close_10_sma'] - data['close_50_sma']

	@classmethod
	def _get_dmi(cls, data):
		""" get the default setting for DMI
		including:
		+DI: 14 days SMMA of +DM,
		-DI: 14 days SMMA of -DM,
		DX: based on +DI and -DI
		ADX: 6 days SMMA of DX
		:param df: data
		:return:
		"""
		data['pdi'] = cls._get_pdi(data, 14)
		data['mdi'] = cls._get_mdi(data, 14)
		data['dx'] = cls._get_dx(data, 14)
		data['adx'] = data['dx_6_ema']
		data['adxr'] = data['adx_6_ema']
		return data
	

	@classmethod
	def _get_um_dm(cls, data):
		""" Up move and down move
		initialize up move and down move
		:param df: data
		"""
		hd = data['High'] - data['Open']
		data['um'] = (hd + hd.abs()) / 2
		ld = -data['Low'] - data['Open']
		data['dm'] = (ld + ld.abs()) / 2

###########################SORTING###############################

	


	
class sort:

	'''Sorting on stock object class'''
	criteria = ['Form_#','Level','Strategy_Formula','#_of_Transactions','#_in_Dataset','D2_Investible_Volume','Calc_Profit%','Win%','Net_Profit/Loss',
			'Highest_Hi','Ratio_Net/Highest_Hi','Hist_Profit/Loss_per_Tx','#_TX>0','%_of_TX>0', '#_OF_"FALSE"_IN_WIN_CALC',
			'Fail%','CALC._PROF/LOSS_ON_Fd_TXS','#_of_Exit_Attempts','%_Successful_Exits','Biggest_Loser','Max_Drawdown(Acct_Min)','Max_%_Drawdown','%_of_TXs_w/_LOSS>50%',
			'Catastrophic_Fail_%_(-80%)','AMT_AT_RISK','Buy_Target_%','Sell_Target_%','Win_Period_10th/90th','Win_Period_20th/80th','Win_Period_40th/60th','Win_Period_50th/50th','Win_Period_60th/40th']	

	strategies = ["20/10.GLB,7-10.GLB,EX1,MGMT1",
			"20/10.GLB,7-10.GLB,EX1,MGMT2",
			"20/10.GLB,7-10.GLB,EX2,MGMT1",
			"20/10.GLB,7-10.GLB,EX2,MGMT2",
			"20/10.GLB,7-10.GLB,EX3,MGMT1",
			"20/10.GLB,7-10.GLB,EX3,MGMT2",
			"20/10.GLB,7-10.SEMI,EX1,MGMT1",
			"20/10.GLB,7-10.SEMI,EX1,MGMT2",
			"20/10.GLB,7-10.REST,EX1,MGMT1",
			"20/10.GLB,7-10.REST,EX1,MGMT2",
			"20/10.SEMI,7-10.GLB,EX1,MGMT1",
			"20/10.SEMI,7-10.GLB,EX1,MGMT2",
			"20/10.SEMI,7-10.GLB,EX2,MGMT1",
			"20/10.SEMI,7-10.GLB,EX2,MGMT2",
			"20/10.SEMI,7-10.GLB,EX3,MGMT1",
			"20/10.SEMI,7-10.GLB,EX3,MGMT2",
			"20/10.SEMI,7-10.SEMI,EX1,MGMT1",
			"20/10.SEMI,7-10.SEMI,EX1,MGMT2",
			"20/10.SEMI,7-10.REST,EX1,MGMT1",
			"20/10.SEMI,7-10.REST,EX1,MGMT2",
			"20/5.GLB,7-10.GLB,EX1,MGMT1",
			"20/5.GLB,7-10.GLB,EX1,MGMT2",
			"20/5.GLB,7-10.GLB,EX2,MGMT1",
			"20/5.GLB,7-10.GLB,EX2,MGMT2",
			"20/5.GLB,7-10.GLB,EX3,MGMT1",
			"20/5.GLB,7-10.GLB,EX3,MGMT2",
			"20/5.GLB,7-10.SEMI,EX1,MGMT1",
			"20/5.GLB,7-10.SEMI,EX1,MGMT2",
			"20/5.GLB,7-10.REST,EX1,MGMT1",
			"20/5.GLB,7-10.REST,EX1,MGMT2",
			"20/5.SEMI,7-10.GLB,EX1,MGMT1",
			"20/5.SEMI,7-10.GLB,EX1,MGMT2",
			"20/5.SEMI,7-10.GLB,EX2,MGMT1",
			"20/5.SEMI,7-10.GLB,EX2,MGMT2",
			"20/5.SEMI,7-10.GLB,EX3,MGMT1",
			"20/5.SEMI,7-10.GLB,EX3,MGMT2",
			"20/5.SEMI,7-10.SEMI,EX1,MGMT1",
			"20/5.SEMI,7-10.SEMI,EX1,MGMT2",
			"20/5.SEMI,7-10.REST,EX1,MGMT1",
			"20/5.SEMI,7-10.REST,EX1,MGMT2",
			"40/10.SEMI,7-10.GLB,EX1,MGMT1",
			"40/10.SEMI,7-10.GLB,EX1,MGMT2",
			"40/10.SEMI,7-10.GLB,EX2,MGMT1",
			"40/10.SEMI,7-10.GLB,EX2,MGMT2",
			"40/10.SEMI,7-10.GLB,EX3,MGMT1",
			"40/10.SEMI,7-10.GLB,EX3,MGMT2",
			"40/10.SEMI,7-10.SEMI,EX1,MGMT1",
			"40/10.SEMI,7-10.SEMI,EX1,MGMT2",
			"40/10.SEMI,7-10.REST,EX1,MGMT1",
			"40/10.SEMI,7-10.REST,EX1,MGMT2",
			"40/5.GLB,7-10.GLB,EX1,MGMT1",
			"40/5.GLB,7-10.GLB,EX1,MGMT2",
			"40/5.GLB,7-10.GLB,EX2,MGMT1",
			"40/5.GLB,7-10.GLB,EX2,MGMT2",
			"40/5.GLB,7-10.GLB,EX3,MGMT1",
			"40/5.GLB,7-10.GLB,EX3,MGMT2",
			"40/5.GLB,7-10.SEMI,EX1,MGMT1",
			"40/5.GLB,7-10.SEMI,EX1,MGMT2",
			"40/5.GLB,7-10.REST,EX1,MGMT1",
			"40/5.GLB,7-10.REST,EX1,MGMT2",
			"40/5.SEMI,7-10.GLB,EX1,MGMT1",
			"40/5.SEMI,7-10.GLB,EX1,MGMT2",
			"40/5.SEMI,7-10.GLB,EX2,MGMT1",
			"40/5.SEMI,7-10.GLB,EX2,MGMT2",
			"40/5.SEMI,7-10.GLB,EX3,MGMT1",
			"40/5.SEMI,7-10.GLB,EX3,MGMT2",
			"40/5.SEMI,7-10.SEMI,EX1,MGMT1",
			"40/5.SEMI,7-10.SEMI,EX1,MGMT2",
			"40/5.SEMI,7-10.REST,EX1,MGMT1",
			"40/5.SEMI,7-10.REST,EX1,MGMT2",
			"20/10.GLB,11-20.GLB,EX4,MGMT1",
			"20/10.SEMI,11-20.GLB,EX4,MGMT1",
			"20/10.GLB,11-20.SEMI,EX4,MGMT1",
			"20/10.SEMI,11-20.SEMI,EX4,MGMT1"]
	account = 20000
	def __init__(self,symbs,d,direct,price_base):
		self.symbs = symbs
		self.direct = direct
		self.d = d
		self.price_base = price_base
		
	def _sorter(self,sorts):
		symbs = os.listdir(self.direct.presort_dir + self.d.strftime('%Y-%m-%d') + '\\')
		for symb in self.symbs:
			print(symb)
			presort_path, datatarget_path, sorttarget_path = self.direct.paths(symb,self.d)
			try:
				sortdirs = os.listdir(presort_path) #list of all presorts for a given symbol on a given day, File level
			except:
				continue #gives list of each sort's summary report folder
			
			for sor in sortdirs: #looks at each folder in summary report folder for each sort folder in list 'sortdirs'
				#sorts = os.listdir(''.join([sortdir,dir])) #gives list of all files in each sort report summary folder
				print(sor)
				#s = sor.split('_',2)
				#option = s[3].replace('.csv','')
				srt = sor.split('_',2)[1]
				if srt in sorts:
					sortsummary = pd.read_csv(''.join([presort_path,sor]),'rb',delimiter=',',parse_dates=['Date'],infer_datetime_format=True).set_index(['Date'])
					#sortsummary.dropna(axis=0,subset=['D2Op'],inplace=True)

					#sortsummary = sortsummary[['Option_Symbol','D2Op',	'd2hi',	'd2lo',	'd2cl',	'd2ivst',	'per1op',	'per1hi',	'per1lo',	'per1cl',	'per1ivst',	'per2op',	'per2hi',	'per2lo',	'per2cl',	'per2ivst',	'per3op',	'per3hi',	'per3lo',	'per3cl',	'per3ivst',	'Avg_Investible_Volume',	'Trade_Hi',	'Trade_Lo',	'per1hi_percent',	'per2hi_percent',	'per3hi_percent',	'per1lo_percent',	'per2lo_percent',	'per3lo_percent',	'per1cl_percent',	'per2cl_percent',	'per3cl_percent',	'D2Hi/D2Op',	'D3Hi/D2Op',	'D4Hi/D2Op',	'D5Hi/D2Op',	'D6Hi/D2Op',	'D7Hi/D2Op',	'D8Hi/D2Op',	'D9Hi/D2Op',	'D10Hi/D2Op',	'D11Hi/D2Op',	'D12Hi/D2Op',	'D13Hi/D2Op',	'D14Hi/D2Op',	'D15Hi/D2Op',	'D16Hi/D2Op',	'D17Hi/D2Op',	'D18Hi/D2Op',	'D19Hi/D2Op',	'D20Hi/D2Op',	'D2Lo/D2Op',	'D3Lo/D2Op',	'D4Lo/D2Op',	'D5Lo/D2Op',	'D6Lo/D2Op',	'D7Lo/D2Op',	'D8Lo/D2Op',	'D9Lo/D2Op',	'D10Lo/D2Op',	'D11Lo/D2Op',	'D12Lo/D2Op',	'D13Lo/D2Op',	'D14Lo/D2Op',	'D15Lo/D2Op',	'D16Lo/D2Op',	'D17Lo/D2Op',	'D18Lo/D2Op',	'D19Lo/D2Op',	'D20Lo/D2Op']]
					data = sortsummary.ix[:dt.datetime.today()-dt.timedelta(days=40)]				
					op = option_pricer(symb,self.direct)
					calls,puts = op.option_price_estimator(data)
					options = ['call','put']
					for option in options:
						if option == 'call':
							datae = pd.DataFrame(data=calls,index=data.index)
						elif option == 'put':
							datae = pd.DataFrame(data=puts,index=data.index)
						sort_report = pd.DataFrame()
						rprtpath = ''.join([sorttarget_path,symb,'_Sort_Report_',srt,'_',option,'_',self.d.strftime('%Y-%m-%d'),'.csv'])
						datapath = ''.join([datatarget_path,symb,'_Sort_Report_Data_',srt,'_',option,'_',self.d.strftime('%Y-%m-%d'),'.csv'])
						frm = 0
						for strategy in self.strategies:
							frm = frm + 1			
							if strategy.split(',',3)[3] == 'MGMT2':
								continue
							else:
								if option == 'call':
									prices = calls
								elif option == 'put':
									prices = puts
								data_report, datah = reporter(data=data,strategy=strategy,account=self.account,option=option,prices=prices,criteria=self.criteria,price_base=self.price_base)
								datae = pd.concat([datae,datah],axis=1)
								if 'Unnamed: 0' in datae.columns:
									datae.drop(['Unnamed: 0'], axis=1,inplace=True)					
								data_report['Form_#'] = str(frm)
								sort_report = pd.concat([sort_report,data_report],axis=0)
						datae.to_csv(datapath,mode='w',header=datae.columns)
						sort_report = sort_report[['Form_#','Level','Strategy_Formula','D2Op/D1Cl','#_of_Transactions','#_in_Dataset','D2_Investible_Volume','Calc_Profit%','Win%','Net_Profit/Loss',
													'Highest_Hi','Ratio_Net/Highest_Hi','Hist_Profit/Loss_per_Tx','#_TX>0','%_of_TX>0','#_OF_"FALSE"_IN_WIN_CALC',
													'Fail%','CALC._PROF/LOSS_ON_Fd_TXS','#_of_Exit_Attempts','%_Successful_Exits','Biggest_Loser','Max_Drawdown(Acct_Min)','Max_%_Drawdown','%_of_TXs_w/_LOSS>50%',
													'Catastrophic_Fail_%_(-80%)','AMT_AT_RISK','Buy_Target_%','Sell_Target_%','Win_Period_10th/90th','Win_Period_20th/80th','Win_Period_40th/60th','Win_Period_50th/50th','Win_Period_60th/40th']]
						sort_report.to_csv(rprtpath,mode='w',index=False,header=sort_report.columns)
					else:
						continue
		return self
		
	def tree_sorter(self,sorts):
		symb = self.symbs[0]
		presort_path, datatarget_path, sorttarget_path = self.direct.tree_paths(symb)
		for sor in sorts.keys(): #looks at each folder in summary report folder for each sort folder in list 'sortdirs'
			
			print(sor)
			sortsummary = sorts[sor]
			data = sortsummary.ix[:dt.datetime.today()-dt.timedelta(days=40)]				
			op = option_pricer(symb,self.direct)
			calls,puts = op.option_price_estimator(data)
			options = ['call','put']
			for option in options:
				if option == 'call':
					datae = pd.DataFrame(data=calls,index=data.index)
				elif option == 'put':
					datae = pd.DataFrame(data=puts,index=data.index)
				sort_report = pd.DataFrame()
				rprtpath = ''.join([sorttarget_path,symb,'_Tree_Sort_Report_',str(sor),'_',option,'.csv'])
				datapath = ''.join([datatarget_path,symb,'_Tree_Sort_Report_Data_',str(sor),'_',option,'.csv'])
				frm = 0
				for strategy in self.strategies:
					frm = frm + 1			
					if strategy.split(',',3)[3] == 'MGMT2':
						continue
					else:
						if option == 'call':
							prices = calls
						elif option == 'put':
							prices = puts
						data_report, datah = reporter(data=data,strategy=strategy,account=self.account,option=option,prices=prices,criteria=self.criteria,price_base=self.price_base)
						datae = pd.concat([datae,datah],axis=1)
						if 'Unnamed: 0' in datae.columns:
							datae.drop(['Unnamed: 0'], axis=1,inplace=True)					
						data_report['Form_#'] = str(frm)
						sort_report = pd.concat([sort_report,data_report],axis=0)
				datae.to_csv(datapath,mode='w',header=datae.columns)
				sort_report = sort_report[['Form_#','Level','Strategy_Formula','D2Op/D1Cl','#_of_Transactions','#_in_Dataset','D2_Investible_Volume','Calc_Profit%','Win%','Net_Profit/Loss',
											'Highest_Hi','Ratio_Net/Highest_Hi','Hist_Profit/Loss_per_Tx','#_TX>0','%_of_TX>0','#_OF_"FALSE"_IN_WIN_CALC',
											'Fail%','CALC._PROF/LOSS_ON_Fd_TXS','#_of_Exit_Attempts','%_Successful_Exits','Biggest_Loser','Max_Drawdown(Acct_Min)','Max_%_Drawdown','%_of_TXs_w/_LOSS>50%',
											'Catastrophic_Fail_%_(-80%)','AMT_AT_RISK','Buy_Target_%','Sell_Target_%','Win_Period_10th/90th','Win_Period_20th/80th','Win_Period_40th/60th','Win_Period_50th/50th','Win_Period_60th/40th']]
				sort_report.to_csv(rprtpath,mode='w',index=False,header=sort_report.columns)
			else:
				continue
		return self
				
				

	
class lcr:
	''' LC_Reporter '''
	
	
	def __init__(self,d,direct):
		self.d = d
		self.direct = direct
		
	def _lcreporter(self,d):
		lcr_dir = self.direct.lcr_dir
		sort_dir = self.direct.sort_dir
		lcr = pd.DataFrame(columns=lcrcolumns)
		symbs = os.listdir(self.direct.sort_dir+d.strftime('%Y-%m-%d'))
		for symb in symbs:		
			targetdir = self.direct.lcr_dir + d.strftime('%Y-%m-%d')+ '\\'		
			sortsdir = self.direct.sort_dir + d.strftime('%Y-%m-%d') + '\\' + symb + '\\Sorts\\'	
			path = ''.join([targetdir,'Level_Class_Report_',d.strftime('%Y-%m-%d'),'.csv'])
			os.makedirs(targetdir, exist_ok=True)
			sorts = os.listdir(sortsdir)
			lcr_report = pd.DataFrame(columns=lcrcolumns)
			del lcr_report['Buy/Sell']
			lcr = lcr_report(sorts,sortsdir,targetdir,to_print=0)
		lcr = lcr[['Stock_Symbol','Option','Sort','Form_#','Level','Strategy','Strategy_Formula','Buy/Sell','D2Op/D1Cl','#_of_Transactions','#_in_Dataset','D2_Investible_Volume','Calc_Profit%','Win%','Net_Profit/Loss',
			'Highest_Hi','Ratio_Net/Highest_Hi','Hist_Profit/Loss_per_Tx','#_TX>0','%_of_TX>0', '#_OF_"FALSE"_IN_WIN_CALC',
			'Fail%','CALC._PROF/LOSS_ON_Fd_TXS','#_of_Exit_Attempts','%_Successful_Exits','Biggest_Loser','Max_Drawdown(Acct_Min)','Max_%_Drawdown','%_of_TXs_w/_LOSS>50%',
			'Catastrophic_Fail_%_(-80%)','AMT_AT_RISK','Buy_Target_%','Sell_Target_%','Win_Period_10th/90th','Win_Period_20th/80th','Win_Period_40th/60th','Win_Period_50th/50th','Win_Period_60th/40th']]
		lcr.to_csv(path,mode='w',header=lcr.columns)
		return self
			
	def lcr_report(self,sorts,sortsdir,targetdir,to_print):			
		for sort in sorts:
			if any(cluster in sort for cluster in cluster_list):
				source = 'Cluster'
			elif '20k' in sort:
				source = '20k'
			elif 'sort' in sort:
				source = 'MT-Auto'
			elif 'dtree' in sort:
				source = 'D_Tree'
			lcreport = pd.read_csv(''.join([sortsdir,sort]),'rb',delimiter=',')
			if 'Unnamed: 0' in lcreport.columns:
				del lcreport['Unnamed: 0']				
			if len(lcreport) > 0:
				#lcreport['Level'] = pd.Series(data=np.where(((lcreport['#_of_Transactions'] > 15) & (lcreport['Ratio_Net/Highest_Hi'] >= .9499) & 
														#(lcreport['Hist_Profit/Loss_per_Tx'] >= (acct * .199)) & (lcreport['#_OF_"FALSE"_IN_WIN_CALC'] < 2) &  
														#(lcreport['%_of_TXs_w/_LOSS>50%'] < .06) & (lcreport['Win%'] >= .9) & (lcreport['Catastrophic_Fail_%_(-80%)'] == 0)),1,
														#np.where(((lcreport['#_of_Transactions'] > 15) & (lcreport['Ratio_Net/Highest_Hi'] >= .9499) & 
														#(lcreport['Hist_Profit/Loss_per_Tx'] >= (acct * .099)) & (lcreport['#_OF_"FALSE"_IN_WIN_CALC'] < 2) &  
														#(lcreport['%_of_TXs_w/_LOSS>50%'] < .06) & (lcreport['Win%'] >= .8) & (lcreport['Catastrophic_Fail_%_(-80%)'] == 0)),2,0)),index=lcreport.index,name='Level')
				lcreport['Strategy'] = source
				level0 = lcreport[lcreport['Level']==0]
				level1 = lcreport[lcreport['Level']==1]
				level2 = lcreport[lcreport['Level']==2]				
				lcreport = pd.concat([level0,level1,level2],axis=0)
				s = sort.split('_',5)
				lcreport['Stock_Symbol'] = s[0]
				lcreport['Sort'] = s[3]
				lcreport['Option'] = s[4]

				lcr_report = pd.concat([lcr_report,lcreport],axis=0)
			else:
				continue

		lcr_report = buy_sell(lcr_report)
		lcr = pd.concat([lcr,lcr_report],axis=0)
		if to_print == 1:
			path = ''.join([targetdir,'Level_Class_Report_',d.strftime('%Y-%m-%d'),'.csv'])
			lcr = lcr[['Stock_Symbol','Option','Sort','Form_#','Level','Strategy','Strategy_Formula','Buy/Sell','D2Op/D1Cl','#_of_Transactions','#_in_Dataset','D2_Investible_Volume','Calc_Profit%','Win%','Net_Profit/Loss',
						'Highest_Hi','Ratio_Net/Highest_Hi','Hist_Profit/Loss_per_Tx','#_TX>0','%_of_TX>0', '#_OF_"FALSE"_IN_WIN_CALC',
						'Fail%','CALC._PROF/LOSS_ON_Fd_TXS','#_of_Exit_Attempts','%_Successful_Exits','Biggest_Loser','Max_Drawdown(Acct_Min)','Max_%_Drawdown','%_of_TXs_w/_LOSS>50%',
						'Catastrophic_Fail_%_(-80%)','AMT_AT_RISK','Buy_Target_%','Sell_Target_%','Win_Period_10th/90th','Win_Period_20th/80th','Win_Period_40th/60th','Win_Period_50th/50th','Win_Period_60th/40th']]
			lcr.to_csv(path,mode='w',header=lcr.columns)
		return lcr
		

	
class tnp_log:
	''' Trades & Plays Log'''
	acct = 10000
	#tnplogpath = dirs.log_dir + 'Trades_&_Plays_Log.csv'
	def __init__(self,direct,initialize):
		self.last_record = [] #tnplog['Trade_Date'].max()
		self.direct = direct
		if initialize == 1:
			self.tnplog = self.tnplogger()
		else:
			self.tnplog = pd.read_csv(self.direct.log_dir + 'Trades_&_Plays_Log.csv','rb',delimiter=',',parse_dates=['Trade_Date'],infer_datetime_format=True).sort_values(by='Trade_Date',ascending=False,axis=0)
	
	def tnplogger(self):
		tnplogpath = self.direct.log_dir + 'Trades_&_Plays_Log.csv'
		if os.path.isfile(tnplogpath):
			tnplog = pd.read_csv(tnplogpath,'rb',delimiter=',',parse_dates=['Trade_Date'],infer_datetime_format=True).sort_values(by='Trade_Date',ascending=False,axis=0)
			lcrs = os.listdir(self.direct.lcr_dir)
			dates = [dt.datetime.strptime(i, '%Y-%m-%d').date() for i in lcrs]
			trades = [i.astype('M8[D]').astype('O') for i in tnplog['Trade_Date'].unique()]
			remaining = list(set(dates) - set(trades))
			tx = 1
		else:
			lcrs = os.listdir(self.direct.lcr_dir)
			remaining = [dt.datetime.strptime(i, '%Y-%m-%d').date() for i in lcrs]
			tnplog = pd.DataFrame(columns = tnpcolumns)
			tx = 0
			
		for d in remaining:
			lcrpath = ''.join([self.direct.lcr_dir,d.strftime('%Y-%m-%d'),'\\Level_Class_Report_',d.strftime('%Y-%m-%d'),'.csv'])	
			lcr = pd.read_csv(lcrpath,'rb',delimiter=',')
			log = tnp_single_trade_logger(lcr,d)
			log.reset_index(drop=True, inplace=True)
			if len(log) > 0:
				if not tx == 0:		
					log['Trade_#'] = log.index + len(tnplog)+1
				else:
					log['Trade_#'] = log.index + 1
				tx = 1
				tnplog = pd.concat([tnplog,log],axis=0)
			#tnplog.sort(columns='Log_Date', axis=0, ascending=True, inplace=True) 
			#tnplog.reset_index(drop=True,inplace=True)
		tnplog = tnplog[['Trade_#','Trade_Date','Option_Symbol','Expiration','Option','Stock_Symbol','Sort','Form_#','Level','Strategy','Strategy_Formula','D2Op/D1Cl','#_of_Transactions','#_in_Dataset','D2_Investible_Volume',
						'Calc_Profit%','Win%','Net_Profit/Loss','Highest_Hi','Ratio_Net/Highest_Hi','Hist_Profit/Loss_per_Tx','#_TX>0','%_of_TX>0', '#_OF_"FALSE"_IN_WIN_CALC',
						'Fail%','CALC._PROF/LOSS_ON_Fd_TXS','#_of_Exit_Attempts','%_Successful_Exits','Biggest_Loser','Max_Drawdown(Acct_Min)','Max_%_Drawdown','%_of_TXs_w/_LOSS>50%',
						'Catastrophic_Fail_%_(-80%)','AMT_AT_RISK','Buy_Target_%','Sell_Target_%','Win_Period_10th/90th','Win_Period_20th/80th','Win_Period_40th/60th','Win_Period_50th/50th','Win_Period_60th/40th']]
		
		tnplog.to_csv(tnplogpath,mode='w',sep=',',index=False)
		return tnplog
	
	def optionizer(self):
		#tnplog = tnplog.replace(' ',np.nan, regex=True)
		self.tnplog.set_index(['Trade_#'])
		tnplogpath = self.direct.log_dir + 'Trades_&_Plays_Log.csv'
		self.tnplog.set_index(keys=['Trade_#'], drop=True, inplace=True)
		to_get_option_symbols = self.tnplog[self.tnplog['Option_Symbol'].isnull()]
		self.tnplog['Option_Symbol'] = self.tnplog['Option_Symbol'].astype('str')
		self.tnplog['Expiration'] = self.tnplog['Expiration'].astype('str')
		dix = dicts(self.direct)
		if os.path.isfile(self.direct.dict_dir + 'option_dictionary.pkl'):
			opt_dict = dix.opt_dict()
		else:
			opt_dict = {}
		for i in to_get_option_symbols.index:
			print(i)
			if i in opt_dict.keys():
				continue
			dat = self.tnplog.loc[i]['Trade_Date'] #datetime.strptime(self.tnplog.loc[x]['Trade_Date'], '%m/%d/%Y')
			symb = self.tnplog.loc[i]['Stock_Symbol']
			print(symb)
			#dirstk = self.direct.db_dir + symb + '\\Stock\\' + symb + '.csv'
			stk = dix.stk_dict(symb) #pd.read_csv(dirstk,'r',',',parse_dates=['Date'],infer_datetime_format=True).set_index(['Date'])
			y = pd.Index(stk.index).get_loc(dat)
			if y == len(stk)-1:
				continue	
			symb = self.tnplog.loc[i]['Stock_Symbol']			
			if symb == '^VIX':
				symb = 'VXX'	
			targetpath = self.direct.opt_data_dir + symb + '\\'
			dir = self.direct.db_dir + symb + '\\Options\\'	
			os.makedirs(targetpath, exist_ok=True)
			if self.tnplog.loc[i]['Option'] == 'call':
				m = 2
			elif self.tnplog.loc[i]['Option'] == 'put':	
				m = -2


			
			d = stk.index[y+1]
			opt_type = self.tnplog.loc[i]['Option']
			min_exp, exp, max_exp = next_monthlies_back(d=dat)
			strike = np.round(stk.loc[d][0],decimals=0)
			diropt = ''.join([dir,symb,'_',str(exp.year),'_',opt_type,'s.csv'])
			try: 
				optdata = pd.read_csv(diropt,'rb',delimiter=',',parse_dates=['expiration','quote_date'],infer_datetime_format=True).sort_values(by='quote_date')						
			except:
				optdata = 'NA'
			opt,option_symbol = option_finder(opt=optdata,symb=symb,strike=strike,start=dat,opt_type=opt_type,min_exp=min_exp,exp=exp,max_exp=max_exp,m=m,stxx='strike')

			print(option_symbol)
			if len(option_symbol) > 5:	
				self.tnplog.at[i,'Option_Symbol'] = option_symbol
				self.tnplog.at[i,'Expiration'] =  option_symbol.split(',',1)[0]
				self.tnplog.at[i,'Log_Date'] = dt.date.today()
				optpath = ''.join([targetpath,option_symbol,'.csv'])
				if i in opt_dict.keys():
					opts = pd.read_csv(optpath,usecols=optcolumns)
					opts = opts[optcolumns] 
					opt = pd.concat([opts,opt],axis=0)
					opt.drop_duplicates(['expiration', 'strike','quote_date'],inplace=True)
					#opt.to_csv(optpath,mode='w',header=opt.columns)
					opt_dict[i] = opt
				else:
					#opt.to_csv(optpath,mode='w',header=opt.columns)
					opt_dict[i] = opt
			else:
				continue
		self.tnplog.to_csv(tnplogpath,mode='w',index=True,header=self.tnplog.columns)
		
		self.tnplog['Expiration'] = pd.to_datetime(self.tnplog['Expiration'],infer_datetime_format=True)
		to_update_option = self.tnplog[self.tnplog['Expiration']>=(dt.datetime.today()+dt.timedelta(weeks=1))]
		for i in to_update_option.index:
			print(i)
			dat = self.tnplog.loc[i]['Trade_Date']
			symb = self.tnplog.loc[i]['Stock_Symbol']
			opt_type = self.tnplog.loc[i]['Option']
			targetpath = self.direct.opt_data_dir + symb + '\\'
			if symb == '^VIX':
				symb = 'VXX'
			option_symbol = self.tnplog.loc[i]['Option_Symbol']
			strike = int(float(option_symbol.split(',',3)[2]))
			exp = pd.to_datetime(option_symbol.split(',',3)[0], infer_datetime_format=True)
			dir = self.direct.db_dir + symb + '\\Options\\'	
			diropt = ''.join([dir,symb,'_',str(exp.year),'_',opt_type,'s.csv'])
			try: 
				optdata = pd.read_csv(diropt,'rb',delimiter=',',parse_dates=['expiration','quote_date'],infer_datetime_format=True).sort_values(by='quote_date')						
			except:
				optdata = 'NA'		
			if len(optdata) > 2:			
				opt = option_recaller(opt=optdata,symb=symb,strike=strike,start=dat,exp=exp)
			else:
				continue
			optpath = ''.join([targetpath,option_symbol,'.csv'])
			if i in opt_dict.keys():
				opts = pd.read_csv(optpath)
				opts = opts[optcolumns]
				print (opts.columns)
				opt = pd.concat([opts,opt],axis=0)
				opt.drop_duplicates(['expiration', 'strike','quote_date'],inplace=True)
				opt = opt[optcolumns]
				#opt.to_csv(optpath,mode='w',index=False,header=opt.columns)
				opt_dict[i] = opt
			else:
				#opt.to_csv(optpath,mode='w',index=False,header=opt.columns)
				opt_dict[i] = opt
			pickle.dump(opt_dict,open(self.direct.dict_dir + 'option_dictionary.pkl', 'wb', pickle.HIGHEST_PROTOCOL))
		return self
		
	def print_options(self):
		dix = dicts(self.direct)
		opt_dict = dix.opt_dict()
		for i in opt_dict.keys():
			opt = opt_dict[i]
			option_symbol = ''.join([str(opt.loc[opt.index[0]]['expiration'].date()),',',opt.loc[opt.index[0]]['option_type'],',',str(opt.loc[opt.index[0]]['strike']),',',opt.loc[opt.index[0]]['underlying_symbol']])
			symb = opt.loc[opt.index[0]]['underlying_symbol']
			optpath = ''.join([self.direct.opt_data_dir, symb,'\\',option_symbol,'.csv'])
			opt.to_csv(optpath,mode='w',index=False,header=opt.columns)
		return self
		
	def tnp_summarizer(self):
		tnpsummarycolumns = ['#_of_Contracts','D2Op/D1Cl','D2Op','D2Hi','D2Lo','D2Cl','D2Vol','Entry_Target','Sell_Target','Exit_Price','Escape_Price','Entry?','Win?','Successful_Exit?','Escape?',              
							'Profit/Loss', 'Day_Calc_Sell_Reached','Trade_Hi_Over_D2Op_(%)','Day_25%_Acheived_Over_D2Op','Day_50%_Acheived_Over_D2Op','Day_75%_Acheived_Over_D2Op','Day_100%_Acheived_Over_D2Op','Day_25%_Acheived_Over_Calc_Buy','Day_50%_Acheived_Over_Calc_Buy','Day_75%_Acheived_Over_Calc_Buy','Day_100%_Acheived_Over_Calc_Buy','#_of_Transactions','#_in_Dataset','D2_Investible_Volume',
							'10%_Profit/Loss','20%_Profit/Loss','25%_Profit/Loss','30%_Profit/Loss','40%_Profit/Loss','50%_Profit/Loss']
							
		acct = 10000
		tnpsumpath = ''.join([self.direct.tnp_dir,'Trades_&_Plays_Summary.csv'])
		tnpsumrawpath = ''.join([self.direct.tnp_dir,'Trades_&_Plays_Summary_Raw.csv'])
		#os.makedirs(tnp_dir, exist_ok=True)
		#self.tnplog = pd.read_csv(tnplogpath,'rb',delimiter=',',parse_dates=['Trade_Date'],infer_datetime_format=True,index_col='Trade_#') 
		if os.path.isfile(tnpsumpath):
			tnpsummary = pd.read_csv(tnpsumpath,'rb',delimiter=',')
			tnpraw = pd.read_csv(tnpsumrawpath,'rb',delimiter=',')
			new_tx = list(set(self.tnplog.index) - set(tnpsummary['Trade_#']))
		else:
			tnpsummary = pd.DataFrame()
			tnpraw =  pd.DataFrame()
			new_tx = self.tnplog.index
			
		dix = dicts(self.direct)
		opt_dict = dix.opt_dict()	
		
		start = pd.to_datetime('2017-01-02', infer_datetime_format=True)
		end =  dt.datetime.today()
		symbs = self.tnplog['Stock_Symbol'].unique()
		#stk_dict = {s: web.DataReader(s, 'google', start, end) for s in symbs}
		self.tnplog = self.tnplog[self.tnplog.index.isin(new_tx)].sort_values(by=['Stock_Symbol','Trade_Date'],axis=0, ascending=True)
		symb1 = ''
		exp1 = pd.to_datetime('2020-10-01', infer_datetime_format=True)	
		t1 = ''
		for tx in new_tx:
			if ((dt.datetime.today() - pd.to_datetime(self.tnplog.loc[tx]['Trade_Date'],infer_datetime_format=True))/np.timedelta64(1, 'D')).astype(int) >= 20:
				#20/10.GLB,11-20.GLB,EX4,MGMT1,call.0
				print(tx)
				strg = self.tnplog.loc[tx]['Strategy_Formula']
				exit_strg = strg.split(',',3)[2]
				t = self.tnplog.loc[tx]['Option'] 
				if t == 'call':
					entry_perc=((self.tnplog.loc[tx]['Buy_Target_%']-1)*10)+1
					sell_perc=((self.tnplog.loc[tx]['Sell_Target_%']-1)*10)+1
					opt_tag = '_Calls.csv'
				if t == 'put':
					entry_perc=((1-self.tnplog.loc[tx]['Buy_Target_%'])*10)+1
					sell_perc=((1-self.tnplog.loc[tx]['Sell_Target_%'])*10)+1	
					opt_tag = '_Puts.csv'
				start = self.tnplog.ix[tx]['Trade_Date'] 
				end = start + dt.timedelta(days=5)
				symb = self.tnplog.loc[tx]['Stock_Symbol']
				if symb == '^VIX':
					symb = 'VXX'		
				option_symbol = self.tnplog.loc[tx]['Option_Symbol']
				print(option_symbol)
				try:
					if tx in opt_dict.keys():
						opt_data = opt_dict[tx]
					else:
						opt_data = 'NA'
					#opt_path = self.direct.opt_data_dir + symb + '\\'
					#opt_data = pd.read_csv(opt_path + option_symbol + '.csv','rb',delimiter=',',parse_dates=['expiration','quote_date'],infer_datetime_format=True)
				except:
					strike = int(float(option_symbol.split(',',3)[2]))
					exp =  pd.to_datetime(option_symbol.split(',',3)[0], infer_datetime_format=True)
					dir = self.direct.db_dir + symb + '\\Options\\'
					diropt = ''.join([dir,symb,'_',str(exp.year),opt_tag])
					if (symb1 != symb) or (exp1.year != exp.year) or (t1 != t):
						try: 
							optdata = pd.read_csv(diropt,'rb',delimiter=',',parse_dates=['expiration','quote_date'],infer_datetime_format=True).sort_index(axis=0)						
						except:
							optdata = 'NA'	
					if len(optdata) > 5:
						opt_data = option_recaller(opt=optdata,symb=symb,strike=strike,start=start,exp=exp)
					t1 = t
					exp1 = exp	
					symb1 = symb
				if (len(opt_data)>15):
					if (opt_data.ix[1]['open']>0):
						analysis_report = tnp_analyzer(opt_data=opt_data,entry_perc=entry_perc,sell_perc=sell_perc,exit_strg=exit_strg,acct=acct) #self.tnplog.loc[tx]['Buy_Target_%']+.05
					opt_data['Trade_#'] = tx
					opt_data['Option_Symbol'] = option_symbol
					tnpraw = pd.concat([tnpraw,opt_data],axis=0)
				else:
					analysis_report = pd.DataFrame(index=np.arange(1),columns=tnpsummarycolumns)
					analysis_report['Trade_#'] = tx
					analysis_report['Option_Symbol'] = option_symbol
				analysis_report['Trade_#'] = tx
				analysis_report['Option_Symbol'] = option_symbol
				#analysis_report = pd.concat([analysis_report,self.tnplog.loc[tx]],axis=1)
				tnpsummary = pd.concat([tnpsummary, analysis_report],axis=0)

			else:
				continue
		tnpsummary.set_index('Trade_#',inplace=True)	
		tnpsummary = pd.concat([tnpsummary, self.tnplog],axis=1)
		tnpsummary = tnpsummary[['Trade_Date','Option_Symbol','delta_1545','Stock_Symbol','Strategy_Formula','Sort','Form_#','Level','#_of_Contracts','D2Op','D2Hi','D2Lo','D2Cl','D2Vol','Entry_Target','Sell_Target','Exit_Price','Escape_Price','Entry?','Win?','Successful_Exit?','Escape?',              
								'Profit/Loss', 'Day_Calc_Sell_Reached','Trade_Hi_Over_D2Op_(%)','Day_25%_Acheived_Over_D2Op','Day_50%_Acheived_Over_D2Op','Day_75%_Acheived_Over_D2Op','Day_100%_Acheived_Over_D2Op','Day_25%_Acheived_Over_Calc_Buy','Day_50%_Acheived_Over_Calc_Buy','Day_75%_Acheived_Over_Calc_Buy','Day_100%_Acheived_Over_Calc_Buy','#_of_Transactions','#_in_Dataset','D2_Investible_Volume',
								'10%_Profit/Loss','20%_Profit/Loss','25%_Profit/Loss','30%_Profit/Loss','40%_Profit/Loss','50%_Profit/Loss',
								'Calc_Profit%','Win%','Net_Profit/Loss','Highest_Hi','Ratio_Net/Highest_Hi','Hist_Profit/Loss_per_Tx','#_TX>0','%_of_TX>0', '#_OF_"FALSE"_IN_WIN_CALC',
								'Fail%','CALC._PROF/LOSS_ON_Fd_TXS','#_of_Exit_Attempts','%_Successful_Exits','Biggest_Loser','Max_Drawdown(Acct_Min)','Max_%_Drawdown','%_of_TXs_w/_LOSS>50%',
								'Catastrophic_Fail_%_(-80%)','AMT_AT_RISK','Buy_Target_%','Sell_Target_%','Win_Period_10th/90th','Win_Period_20th/80th','Win_Period_40th/60th','Win_Period_50th/50th','Win_Period_60th/40th']]
		tnpraw = tnpraw[['Trade_#','quote_date','underlying_symbol', 'root', 'expiration', 'strike', 'option_type',
						 'open', 'high', 'low', 'close', 'trade_volume', 'bid_size_1545',
						 'bid_1545', 'ask_size_1545', 'ask_1545', 'underlying_bid_1545',
						 'underlying_ask_1545', 'implied_underlying_price_1545',
						 'active_underlying_price_1545', 'implied_volatility_1545', 'delta_1545',
						 'gamma_1545', 'theta_1545', 'vega_1545', 'rho_1545', 'bid_size_eod',
						 'bid_eod', 'ask_size_eod', 'ask_eod', 'underlying_bid_eod',
						 'underlying_ask_eod', 'vwap', 'open_interest', 'delivery_code']]


		tnpsummary.to_csv(tnpsumpath, mode='w', index=True, header=tnpsummary.columns)
		tnpraw.to_csv(tnpsumrawpath, mode='w', index=False, header=tnpraw.columns)
		return tnpsummary, tnpraw


    

