



Filter to only 1.6-2stdev otm

Set calls

Set puts

With each

sxl_s matrix of strike diffs

sxl_p matrix of price diffs 3:45pm ask to bid

sxl_r matrix of ratios p/s

Then make df and order by profit, delta, gamma, theta, impvol/40dsma, impvol/40dstdev_norm

Need to make matrices of diffs of theta, delta and impvol.

Need to determine the priority of these factors

The back test with final outcome

Report features:

Max drawdown and final outcome

optdata = pd.read_csv(diropt,'rb',delimiter=',',parse_dates=['expiration','quote_date'],infer_datetime_format=True).sort_index(axis=0)
start = pd.to_datetime('2017-01-02', infer_datetime_format=True)
end = pd.to_datetime('2017-01-02', infer_datetime_format=True)

def days_2_exp(data):
	d2x = pd.Series(data=(data['expiration'] - data['quote_date']),name='days_to_expiration')
	data = data.join(d2x)
	return data

def OTM(data,t):
	if t == 'c'
		otm = pd.Series(data=(data['strike'] - (((data['underlying_bid_eod']+data['underlying_ask_eod'])/2)),name='out_of_the_money')
	if t == 'p'
		otm = pd.Series(data=(data['strike'] - (((data['underlying_bid_eod']+data['underlying_ask_eod'])/2)),name='out_of_the_money')
	data = data.join(otm)
	return data
	
def build_matrix(func,args):
	return func(*args)
def subtract(A,B):
	return A-B


#Filtering 1:	
optdata = optdata[(optdata['days_to_expiration']>=30) & (optdata['days_to_expiration']<=45)]	
stdev = pd.Series(data=(stk['Close'].rolling(window=(40)).apply(np.std)),index=stk.index)
for q in optdata['quote_date'].unique():
	optdata['40_day_standard_dev.'][optdata['quote_date']==q] = stdev.ix[q]
optdata = optdata[(optdata['out_of_the_money'] >= (optdata['40_day_standard_dev.'] * 1.6))]
optdata.reset_index(drop=True,inplace=True)

X = numpy.zeros((len(optdata),len(optdata)))
for x in optdata['expiration'].unique():
	opt = optdata[optdata['expiration'] == x]
	sxl_s = build_matrix(subtract,(opt['strike'],opt['strike']))#pd.DataFrame([[s-l for s in opt['strike']] for l in opt['strike']])
	sxl_p = build_matrix(subtract,(opt['bid_1545'],opt['ask_1545']))#pd.DataFrame([[s-l for s in opt['bid_1545']] for l in opt['ask_1545']])
	sxp_r = sxl_p/sxl_s
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	