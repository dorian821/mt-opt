
from numba import jit
#VWAP
@jit
def vwap(data):
  data.assign('VWAP'=np.cumsum(data['Volume']*data['Typical'])/ np.cumsum(data['Volume']),inplace=True)
  return data


#VWAP Ratios

def vwap_ratios(data):
    data.assign('VWAP_3dSlope'=data['VWAP'].rolling(window=3,center=False).apply(three_linest),inplace=True)
    data.assign('5d_SMA/VWAP'=data['5d_SMA']/data['VWAP'],inplace=True)
    data.assign('D1Hi/VWAP'=data['High']/data['VWAP'],inplace=True)
    data.assign('D1Lo/VWAP'=data['low']/data['VWAP'],inplace=True)
    return data
  
  
def nearest_resup(val,arr,direction)
    idx = np.argmin(np.abs(arr-val))
    return arr.ix[idx+direction]
    
def support_resistance(data,margin,level):    
    acres = np.sort(data[['High','Low']].stack())
    while (acres != centers) | (len(centers)):
      acres = acres1
      centers = []
      for i in acres:
          diff = np.abs(acres[i] - acres)/acres
          cluster = diff[diff<margin]
          strength = len(cluster)
          if strength>=level:
              centers.extend(cluster.mean())
      acres1 = centers
    centers = np.sort(centers)
    for j in data.index:
        data.loc[j,'Resistance'] = nearest_resup(data.loc[j]['High'],centers,1)
        data.loc[j,'Support'] = nearest_resup(data.loc[j]['High'],centers,-1)        
    data.assign('Resistance_Ratio'=data['High']/data['Resistance'],inplace=True)
    data.assign('Support_Ratio'=data['Low']/data['Resistance'],inplace=True)
    return centers 
            
    cluster highs and lows by normalized proximity
    take average of each cluster
    store in dictionary with number of pop
    for each day print nearest support and resistance and ratio of low/high to these
    return nearest support and resistance and ratios
    
    
    
