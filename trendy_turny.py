stackoverflow.com/questions/8587047/support-resistance-algorithm-technical-analysis+&cd=3&hl=en&ct=clnk&gl=gr
def turningpoints(lst):
    dx = np.diff(lst)
    turns = list(np.sum(dx[1:] * dx[:-1] < 0))
    turns = turns + ([]*np.arange(len(lst)-len(turns))
    return turns


#trend or turn
def trend_turn(data,highband,lowband):
  time_frame = (5,10,20)
  for t in time_frame:
    sma = data['Typical'].rolling(window=t).apply(mean).fillna(method='bfill')
    data[str(t) + 'd_Turning_Points'] = turningpoints(sma)
    slope = sma.rolling(window=3).apply(three_linest)
    data[str(t) + 'd_Trend/Turn'][slope > highband] = 'Up_Trend'
    data[str(t) + 'd_Trend/Turn'][slope < lowband] = 'Down_Trend'
    data[str(t) + 'd_Trend/Turn'][(data[str(t) + 'd_Trend/Turn'].shift(-1) == 'Down_Trend') & (data[str(t) + 'd_Trend/Turn'].isnull())] = 'Down_Turn'
    data[str(t) + 'd_Trend/Turn'][(data[str(t) + 'd_Trend/Turn'].shift(-1) == 'Up_Trend') & (data[str(t) + 'd_Trend/Turn'].isnull())] = 'Up_Turn'
    data[str(t) + 'd_Trend/Turn'].fillna(method='bfill')
  return data
    
def opt_volume_peak(data):
    peak_moneyness = {}
    for d in data['quote_date'].unique():
        data['strike'][(data['quote_date'] == d) & (data['trade_volume'] == data['quote_date'].max())]
                    
