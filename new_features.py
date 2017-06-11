
from numba import jit
#VWAP
@jit
def vwap(data):
    data = data.assign(VWAP=np.cumsum(data['Volume']*data['Typical'])/np.cumsum(data['Volume']))
    return data

def typ(data):
    data = data.join(pd.Series(data=(data['High']+data['Low']+data['Close'])/3,index=data.index,name='Typical'))
    return data

#VWAP Ratios
def vwap_ratios(data):
    data.assign(VWAP_3dSlope=data['VWAP'].rolling(window=3,center=False).apply(three_linest),inplace=True)
    data = data.join(pd.Series(data=data['5d_SMA']/data['VWAP'],index=data.index,name='5d_SMA/VWAP'))
    data = data.join(pd.Series(data=data['High']/data['VWAP'],index=data.index,name='5d_SMA/VWAP'))
    data = data.join(pd.Series(data=data['Low']/data['VWAP'],index=data.index,name='5d_SMA/VWAP'))
    return data

  
def nearest_resup(val,arr,direction):
    if direction == 1:
        arr = arr[arr>val]
    elif direction == -1:
        arr = arr[arr<val]
    idx = np.argmin(np.abs(arr-val))
    return arr[idx]
    
def support_resistance(data,margin,level):    
    acres = list(np.sort(data[['High','Low']].stack()))
    centers = []
    acres1 = []    
    m, n = 1, 0
    while n != m:
        print(n,m)
        m = len(centers)
        centers = []
        for i,value in enumerate(acres):
            diff = list(np.abs(acres[i] - acres)/acres)
            cluster = [x for (x,y) in zip(acres,diff) if y < margin] #acres[diff<margin]
            strength = len(cluster)
            if strength>=level:
                centers.extend([np.sum(cluster)/strength])
                n = len(centers)
        acres = centers
    centers = np.sort(centers)
    for j in data.index:
        data.loc[j,'Resistance'] = nearest_resup(data.loc[j]['High'],centers,1)
        data.loc[j,'Support'] = nearest_resup(data.loc[j]['High'],centers,-1)        
    data = data.join(pd.Series(data=data['High']/data['Resistance'],index=data.index,name='Resistance_Ratio'))
    data = data.join(pd.Series(data=data['Low']/data['Support'],index=data.index,name='Support_Ratio'))
    return data 

stk = typ(stk)
stk = vwap(stk)
df = support_resistance(stk,.0001,5)

    cluster highs and lows by normalized proximity
    take average of each cluster
    store in dictionary with number of pop
    for each day print nearest support and resistance and ratio of low/high to these
    return nearest support and resistance and ratios
  
  
for i in acres.index:
    diff = pd.Series(np.abs(acres[i] - acres)/acres)
    #print('took diffs')
    cluster = acres[diff<margin]
    #print('clustered')
    strength = len(cluster)
    #print('strength measured')
    if strength>=level:
        #print('list extending')
        centers.extend([cluster.mean()])
        
    
    
