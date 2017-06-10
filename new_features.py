
from numba import jit
#VWAP
@jit
def vwap(data):
  data = data.assign('VWAP'=np.cumsum(data['Volume']*data['Typical'])/ np.cumsum(data['Volume']))
  return data


#VWAP Ratios

def vwap_ratios(data):
    data = data.assign('VWAP_3dSlope'=data['VWAP'].rolling(window=3,center=False).apply(three_linest))
    data = data.assign('5d_SMA/VWAP'=data['5d_SMA']/data['VWAP'])
    data = data.assign('D1Hi/VWAP'=data['High']/data['VWAP'])
    data = data.assign('D1Lo/VWAP'=data['low']/data['VWAP'])
    return data
    
def support_resistance(data):
    cluster highs and lows by normalized proximity
    take average of each cluster
    store in dictionary with number of pop
    for each day print nearest support and resistance and ratio of low/high to these
    return nearest support and resistance and ratios
    
    
    
