
find mins/maxs in time_frame
draw linspace between

def get_bottom_idx():
def get_top_idx():
  
def find_lsq(candidates, data): TL anchor is always the first candidate, candidates is list of tuples of (max/min, index), data is the OHLC of the stock
  anchor = candidates[0]
  c_rank = {}
  i = 1
  for c in candidates[1:]:
    line = pd.Series(data=np.linspace(anchor[0], c[0], num=len(data[anchor[1]:c[1]]), endpoint=True, retstep=False, dtype=None),index=data[anchor[1]:c[1]].index)
    diffs = abs(data[OHLC] - line)/data['Close']
    test = diffs.min(axis=1) < .005
    c_rank[i] = (test.sum(),c)
    i += 1
  best_fit = c_rank[max(c_rank.items(), key= (lambda key: key[1][0]))][1]
  return np.linspace(anchor[0], best_fit[0], num=len(data[anchor[1]:best_fit[1]]), endpoint=True, retstep=False, dtype=None)
    

  def extremes(data,col,threshold,direction):
    series = data[col]
    dominion = measure_turn(series,direction)
    anchor_idx = [series.index[dominion > threshold]]
    anchors = [series[anchor_idx]]
    extremities = {}
    for i in anchor_idx:
      period = series[anchor_idx[i]:anchor_idx[i+1]]
      middle  = period.mean()      
      if col == 'High':
         cut = pd.Series(data=np.linspace(series[anchor_idx[i]]-middle, series[anchor_idx[i+1]]-middle, num=len(period),endpoint=True, retstep=False, dtype=None),index=period.index)
         extreme = period[(period-cut) >= 0].max()
      elif col == 'Low':
         cut = pd.Series(data=np.linspace(series[anchor_idx[i]]+middle, series[anchor_idx[i+1]]+middle, num=len(period),endpoint=True, retstep=False, dtype=None),index=period.index)
         extremes = period[(period-cut) <= 0].min()
      extremes = extremes.groupby((extremes != extremes.shift()).cumsum()).idxmax()
      extremities[i]
        
      
def extremities(data,col,n):
  series = data[col]
  extremitys = {}  
  for i in np.arange(n):
    if i == 0:
      middle  = series.mean()  
      m = np.polyfit(x=np.arange(len(series.iloc[:-100])),y=series.iloc[:-100],deg=1)[0]
      mid_idx = int(len(series)/2)
      remainder = len(series) - mid_idx
      cut = pd.Series(data=np.linspace(0, m*mid_idx, num=mid_idx,endpoint=True, retstep=False, dtype=None),index=series.index[:mid_idx])
      cut1 = pd.Series(data=np.linspace(m*mid_idx, m*mid_idx+(m*remainder), num=remainder,endpoint=True, retstep=False, dtype=None),index=series.index[mid_idx:])
      cutt = cut.append(cut1)
      if col == 'High':
        extremes = (series-cutt) > 0        
      elif col == 'Low':
        extremes = (series-cutt) < 0 
      extreme = series.groupby((extremes != extremes.shift()).cumsum()).idxmax()
      extremitys[i] =  series[extreme]
    else:
      for j, v in enumerate(extremeties[i-1]):
        if j == len(extremeties[i-1]) - 1:
          continue
        period = series[extremeties[i-1][j]:extremeties[i-1][j+1]]
        middle  = series.mean()  
        m = np.polyfit(x=np.arange(len(series.iloc[:-100])),y=series.iloc[:-100],deg=1)[0]
        cut = pd.Series(data=np.linspace(0, m*mid_idx, num=mid_idx,endpoint=True, retstep=False, dtype=None),index=series.index[:mid_idx])
        cut1 = pd.Series(data=np.linspace(m*mid_idx, m*mid_idx+(m*remainder), num=remainder,endpoint=True, retstep=False, dtype=None),index=series.index[mid_idx:])
        cutt = cut.append(cut1)
        if col == 'High':
          extremes = (series-cutt) > 0 
        elif col == 'Low':
          extremes = (series-cutt) < 0
        extreme = extremes.groupby((extremes != extremes.shift()).cumsum()).idxmax()
        extremitys[i] =  series[extreme]
  return extremitys

then for each value in extremities find the best trend line using the extremeties of the next level, do this for low and high
      
put trend linspaces into column in pieces where they are valid
the take diff ratios of highs and lows
then figure out a way to identify trendline cross overs
try to keep trendlines that have the majority of the prices within them, i.e. above the lower and below the higher
n should probably = 3-5
draw parallels for lines tied to same degree opposite extremes
take ratios of non-parrallel trendlines to identify convergence
      
      
      
      
      dominion[anchor_idx[i]+dt.datetime.timdelta(days=1):anchor_idx[i+1]].max()
      
    
    anchor = pd.Series(data=rolling_idxmax(series,time_frame,-1),index=data.index).fillna(method='bfill')
    maxima = pd.Series(index=data.index,name='Trendline_Maxima')
    for i in series.index:
      period = series[anchor[i]:i]
      maxima[i] = 
      
      
def outer_line(series,direction,threshold):
  anchor_idx = series.index[dominion > threshold]
  anchors = series[anchor_idx]
  for i in series.index:
      
      
def measure_turn(series, direction):
  dominion = pd.Series(index=series.index)
  for i in series.index:
    if direction == 1:
      dominion[i] = ((series[i] - series[i:]) > 0).sum()
    elif direction == -1:
      dominion[i] = ((series[:i] - series[i]) > 0).sum()
  return dominion
  
  
2 types of trend lines
- min in time_frame to next min ('extr')
- min in time_frame to least squares min from mins in group ('fit')
