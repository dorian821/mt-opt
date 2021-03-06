def date_diff(dates1,dates2):
	diffs = ((dates2-dates1)/np.timedelta64(1, 'D')).astype(int)
	return diffs

def closest(value, myList):
	myList = map(int,myList)
	idx = min(myList, key=lambda x:abs(x-value))
	return idx

class option_pricer:
	types = ('Puts','Calls')
	def __init__(self,symb):
		self.symb = symb
		self.stk = web.data_reader(self.symb,'yahoo',
					   
	def single_option_price_estimator(self,d2,d2x,typ):	
		stk = self.stk[d2:d2+dt.timedelta(days=35)]
		closes = pd.Series(data=np.round((((stk['Close'] - stk.ix[d2]['Close'])/stk.ix[d2]['Close'])-1)*100,decimals=1),index=stk.index).reset_index(drop=True)
		diffs = pd.Series(data=(((stk.index[j]-stk.index[0])/np.timedelta64(1, 'D')).astype(int) for j in (6,10,15,20)),index=(6,10,15,20))
		price_ratios = {}
		if typ == 'Calls':
			t = 1
		elif typ == 'Puts':
			t = -1
		for i in diffs.index:
			d2_price = self.pmx[symb+'_'+typ].ix[d2x][str(closest(tpmx.columns))]
			price_ratios[i] = self.pmx[symb+'_'+typ].ix[i][str(int(closest(t-closes[i],pmx.columns)))]/d2_price
			#price_ratios = pd.Series(data=sig.savgol_filter(price_ratios, 5, 3),index=np.arange(len(data))+2)	
		return price_ratios
	
	def option_price_estimator(self,data):
		exps = pd.Series(data=(map(next_monthly,data.index)),index=data.index)
		d2s = pd.Series(data=[self.stk.index[i+1] for i in np.arange(len(self.stk)-1)],index=self.stk.index[:-1],name='d2s')
		d2s = d2s[d2s.index.isin(data.index)]
		d2xs = pd.Series(data=date_diff(d2s,exps),index=data.index,name='d2xs')
		dedo = pd.concat([d2s,d2xs],axis=1)
		prices = {}
		for typ in self.types:
			prices[typ] = map(single_option_price_estimator,zip(d2s,d2xs,typ))
		calls = pd.DataFrame(prices['Calls'],index=data.index)
		puts = pd.DataFrame(prices['Puts'],index=data.index)
		return calls, puts

		for d in data.index:		
			put_p, call_p = single_option_price_estimator(dedo)
			puts.ix[d] = put_p
			calls.ix[d] = call_p
		return puts, calls
	
def call_trade_analyzer(data,sellpc,buypc,exstr,exitpc,mmgmt,strategyformula,acct,calls):
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
								np.where(escape == 'EX3',(acct * prices.ix[6]),
								np.where(((escape == 'EX2') & (successfulexits == False)),(np.maximum((acct * prices.ix[10]),(acct*-1))),
								np.where(((escape == 'EX1') & (successfulexits == False)),(np.maximum((acct * prices.ix[10]),(acct*-1))),
								np.where((escape == 'EX4'),(np.maximum((acct * prices.ix[20]),(acct*-1))),np.nan))))))),index=data.index,name=trpl)
	ev = ''.join([strategyformula,'_Evaluation'])
	eval = pd.Series(data=(np.where(((wins == False) | ((wins == True) & (tradepl == (acct * ((sellpc-buypc)*10))))),True,False)),index=data.index, name =''.join([strategyformula,'_Win_Calc_Evaluation'])) #Discuss this calc with Dad

	datah = pd.concat([entries,wins,successfulexits,escape,tradepl,eval],axis=1)
	return datah

def put_trade_analyzer(data,sellpc,buypc,exstr,exitpc,mmgmt,strategyformula,acct,puts):
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
								np.where(escape == 'EX3',(acct * prices.ix[6]),
								np.where(((escape == 'EX2') & (successfulexits == False)),(np.maximum((acct * prices.ix[10]),(acct*-1))),
								np.where(((escape == 'EX1') & (successfulexits == False)),(np.maximum((acct * prices.ix[10]),(acct*-1))),
								np.where((escape == 'EX4'),(np.maximum((acct * prices.ix[20]),(acct*-1))),np.nan))))))),index=data.index,name=trpl)
	ev = ''.join([strategyformula,'_Evaluation'])
	eval = pd.Series(data=(np.where(((wins == False) | ((wins == True) & (tradepl == (acct * ((buypc-sellpc)*10))))),True,False)),index=data.index, name =''.join([strategyformula,'_Win_Calc_Evaluation'])) #Discuss this calc with Dad
	datah = pd.concat([entries,wins,successfulexits,escape,tradepl,eval],axis=1)
	return datah
  
def reporter(data,strategy,account,option,prices,criteria):
	buyperc,sellperc,sellset,exitperiod,exitset,exstr,mmgmt = decoder(strategy)
	if option == 'call':		
		heads = ['D2Lo/D2Op','Trade_Hi','Per1Hi/D2Op','Per2Hi/D2Op','Per1-2Hi/D2Op','Per3-4Hi/D2Op']
		quants = [.1,.2,.4,.5,.6]
	elif option == 'put':		
		heads = ['D2Hi/D2Op','Trade_Lo','Per1Lo/D2Op','Per2Lo/D2Op','Per1-2Lo/D2Op','Per3-4Lo/D2Op']
		buyperc = 100-buyperc
		sellperc = 100-sellperc
		quants = [.9,.8,.6,.5,.4]
	
	report = pd.DataFrame(index=np.arange(1),columns=criteria)
	strategyformula = strategy
	report['Strategy_Formula'] = strategyformula
	report['#_in_Dataset'] = len(data)
	buypc = data[heads[0]].quantile(q=buyperc/100)
	report['Buy_Target_%'] = buypc
	acct = account * .5
	if option == 'call':
		semiset = data[data[heads[0]] <= buypc] #Semi-Restricted Data Set
	elif option == 'put':
		semiset = data[data[heads[0]] >= buypc]
		
	
	
	if exstr == 'EX4':
		if option == 'call': 
			sellpc = buypc * 1.025
		elif option == 'put':
			sellpc = buypc * .975
		report['Win_Period_10th'] = data[heads[4]].quantile(q=quants[0])
		report['Win_Period_20th'] = data[heads[4]].quantile(q=quants[1])
		report['Win_Period_40th'] = data[heads[4]].quantile(q=quants[2])
		report['Win_Period_50th'] = data[heads[4]].quantile(q=quants[3])
		report['Win_Period_60th'] = data[heads[4]].quantile(q=quants[4])
	else: 
		if sellset == 'SEMI':
			sellpc = semiset[heads[2]].quantile(q=sellperc/100)
			report['Win_Period_10th'] = semiset[heads[2]].quantile(q=quants[0])
			report['Win_Period_20th'] = semiset[heads[2]].quantile(q=quants[1])
			report['Win_Period_40th'] = semiset[heads[2]].quantile(q=quants[2])
			report['Win_Period_50th'] = semiset[heads[2]].quantile(q=quants[3])
			report['Win_Period_60th'] = semiset[heads[2]].quantile(q=quants[4])
		else: 
			sellpc = data[heads[2]].quantile(q=sellperc/100)
			report['Win_Period_10th'] = data[heads[2]].quantile(q=quants[0])
			report['Win_Period_20th'] = data[heads[2]].quantile(q=quants[1])
			report['Win_Period_40th'] = data[heads[2]].quantile(q=quants[2])
			report['Win_Period_50th'] = data[heads[2]].quantile(q=quants[3])
			report['Win_Period_60th'] = data[heads[2]].quantile(q=quants[4])
		
	restset = semiset[semiset[heads[2]] >= sellpc] #Restricted Data Set
	report['Sell_Target_%'] = sellpc
	#Exit Percentage Target Based On Exit Set Specification
	if exstr != 'EX4':
		if exitset == 'GLB':
			exitpc = data[heads[3]].quantile(q=sellperc/100)
		elif exitset == 'SEMI':
			exitpc = semiset[heads[3]].quantile(q=sellperc/100)
		elif exitset == 'REST':
			exitpc = restset[heads[3]].quantile(q=sellperc/100)
	elif exstr == 'EX4':
		if option == 'call': 
			exitpc = buypc * 1.05
		elif option == 'put':
			exitpc = buypc * .95
	
		
	#Generate Column Names
	wins = ''.join([strategyformula,'_Wins?'])
	trpl = ''.join([strategyformula,'_Trade_Profit/Loss'])
	ev = ''.join([strategyformula,'_Win_Calc_Evaluation'])
	biglosstx = ''.join([strategyformula,'_TXs_Loss>50%'])
	cumprofit = ''.join([strategyformula,'_Cumulative_Profit/Loss'])
	entries = ''.join([strategyformula,'_Got_In?'])
	successfulexits = ''.join([strategyformula,'_Successful_Exits'])
	
	
	if option == 'call':
		report['Calc_Profit%'] = (sellpc-buypc)*10
		pft = 1 + ((sellpc-buypc)*10)
		datah = call_trade_analyzer(data,sellpc=sellpc,buypc=buypc,exstr=exstr,exitpc=exitpc,mmgmt=mmgmt,strategyformula=strategyformula,acct=acct,prices=prices)
	elif option == 'put':
		report['Calc_Profit%'] = (buypc-sellpc)*10
		pft = 1 + ((buypc-sellpc)*10)
		datah = put_trade_analyzer(data,sellpc=sellpc,buypc=buypc,exstr=exstr,exitpc=exitpc,mmgmt=mmgmt,strategyformula=strategyformula,acct=acct,prices=prices)
	#Final Calcs
	#report['D2_Investible_Volume'] = data.ix[-20:]['d2ivst'].mean()#test
	report['#_of_Transactions'] = np.sum(datah[entries])
	datah[biglosstx] = datah[trpl] <= (acct * -.5)
	report['#_OF_"FALSE"_IN_WIN_CALC'] = len(data) - np.sum(datah[ev]) #Investigate: Clarify the meaning of the Win Calc and its calculation
	report['%_of_TXs_w/_LOSS>50%'] = np.sum(datah[biglosstx])/len(semiset)
	datah[cumprofit] = datah[trpl].cumsum()
	report['Win%'] = np.sum(datah[wins])/np.sum(datah[entries])
	report['Fail%'] = 1-report['Win%']
	report['CALC._PROF/LOSS_ON_Fd_TXS'] = exitpc-buypc
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
	report['Catastrophic_Fail_%_(-80%)'] = np.sum(datah[trpl]<=(acct*(-.8)))
	report['AMT_AT_RISK'] = acct
	report['#_of_Exit_Attempts'] = np.sum(datah[entries]) - np.sum(datah[wins])
	report['%_Successful_Exits'] = np.sum(datah[successfulexits])/(np.sum(datah[entries]) - np.sum(datah[wins]))
	
	
	if (mmgmt == 'MGMT1') & (report.ix[0]['#_of_Transactions'] > 15) & (report.ix[0]['Ratio_Net/Highest_Hi'] >= .9499) & (report.ix[0]['Hist_Profit/Loss_per_Tx'] >= (acct * .099)) & (report.ix[0]['#_OF_"FALSE"_IN_WIN_CALC'] < 2) &  (report.ix[0]['%_of_TXs_w/_LOSS>50%'] < .06) & (report.ix[0]['Win%'] >= .8) & (report.ix[0]['Catastrophic_Fail_%_(-80%)'] == 0):
		report['Level'] = 2
	elif (mmgmt == 'MGMT1') & (report.ix[0]['#_of_Transactions'] > 15) & (report.ix[0]['Ratio_Net/Highest_Hi'] >= .9499) & (report.ix[0]['Hist_Profit/Loss_per_Tx'] >= (acct * .199)) & (report.ix[0]['#_OF_"FALSE"_IN_WIN_CALC'] < 2) &  (report.ix[0]['%_of_TXs_w/_LOSS>50%'] < .06) & (report.ix[0]['Win%'] >= .9) & (report.ix[0]['Catastrophic_Fail_%_(-80%)'] == 0): 
		report['Level'] = 1
	elif (mmgmt == 'MGMT1') & (report.ix[0]['#_of_Transactions'] > 15) & (report.ix[0]['Ratio_Net/Highest_Hi'] >= .9499) & (report.ix[0]['Hist_Profit/Loss_per_Tx'] >= (acct * .299)) & (report.ix[0]['#_OF_"FALSE"_IN_WIN_CALC'] < 2) &  (report.ix[0]['%_of_TXs_w/_LOSS>50%'] < .06) & (report.ix[0]['Win%'] >= .92) & (report.ix[0]['Catastrophic_Fail_%_(-80%)'] == 0): 
		report['Level'] = 0
	else:
		report['Level'] = 3
	return report, datah 

for symb in symbs:
	dfs = os.listdir(direct.data_dir+symb+'\\')
	price_matrices = direct.pm_dir
	os.makedirs(price_matrices,exist_ok=True)
	
	for typ in types:
		datae = [df for df in dfs if typ in df]
		data = pd.DataFrame()
		for df in datae:		
			dat = pd.read_csv(direct.data_dir+symb+'\\'+df,'rb',delimiter=',',parse_dates=['expiration','quote_date'],infer_datetime_format=True)
			data = pd.concat([data,dat],axis=0)
		underlying = pd.Series(data=((data['underlying_bid_eod']+data['underlying_ask_eod'])/2),index=data.index,name='underlying..')
		stx = pd.Series(data=np.round(((data['strike'] - underlying)/underlying)-1)*100,decimals=1),index=data.index,name='stx')
		close = pd.Series(data=((data['bid_eod']+data['ask_eod'])/2),index=data.index,name='close..')
		d2x = pd.Series(data=((data['expiration']-data['quote_date'])/np.timedelta64(1, 'D')).astype(int), index=data.index,name='d2x')
		pm_data = pd.concat([stx,d2x,close],axis=1)
		pm_data = pm_data[pm_data['d2x']<=40]
		grouped = pm_data.groupby(['stx','d2x'])['close..'].mean()
		price_matrix = grouped.pivot_table(values='close..',index=['d2x'],columns=['stx']
		price_matrix.to_csv(direct.pm_dir+symb+'_'+typ+'_price_matrix.csv',mode='w',index=True)				   
		return price_matrix
						   
						   
		price_matrix = pd.DataFrame()
		for strike in np.arange(37)-18:
			price_curve = pd.Series(index=np.arange(30)+1,name=str(strike))
			for days in np.arange(31):
				day_price = pm_data['close..'][(d2x==days)&(stx==strike.astype('float64'))].mean()
				price_curve.set_value(days,day_price)
			price_matrix = pd.concat([price_matrix,price_curve],axis=1)
		x = price_matrix.fillna(method='backfill',axis=0)
		y = price_matrix.fillna(method='ffill',axis=0)
		price_matrix = (x+y)/2
		price_matrix.to_csv(direct.pm_dir+symb+'_'+type+'_price_matrix.csv',mode='w',index=True)
