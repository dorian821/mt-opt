
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
    

  def local_maxima(data,time_frame,zenith_or_nadir):
    if zenith_or_nadir == 'zenith':
      col = 'High'
      direction = 1
    elif zenith_or_nadir == 'nadir':
      col = 'Low'
      direction = -1
    series = data[col]
    dominion = measure_turn(series,direction)
    anchor_idx = series.index[dominion > threshold]
    anchors = series[anchor_idx]
    
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
