
from numba import jit
#VWAP
@jit
def vwap(data,window):
    voltyp = pd.Series(data['Volume']*data['Typical'])
    voltyp = voltyp.rolling(window=window).sum()
    vol = pd.Series(data['Volume'].rolling(window=window).sum())
    data = data.join(pd.Series(data=voltyp/vol,index=data.index,name='VWAP_'+str(window)))
    return data

def typ(data):
    data = data.join(pd.Series(data=(data['High']+data['Low']+data['Close'])/3,index=data.index,name='Typical'))
    return data

#VWAP Ratios
def vwap_ratios(data):
    #data.assign(VWAP_3dSlope=data['VWAP'].rolling(window=3,center=False).apply(three_linest),inplace=True)
    #data = data.join(pd.Series(data=data['5d_SMA']/data['VWAP'],index=data.index,name='5d_SMA/VWAP'))
    data = data.join(pd.Series(data=data['High']/data['VWAP_200'],index=data.index,name='High/VWAP'))
    data = data.join(pd.Series(data=data['Low']/data['VWAP_200'],index=data.index,name='Low/VWAP'))
    return data

  
def nearest_resup(val,arr,direction):
    if direction == 1:
        ar = arr[arr>val]
    elif direction == -1:
        ar = arr[arr<val]
    if len(ar) != 0:
        idx = np.argmin(np.abs(ar-val))
    else:
        ar = arr
        if direction == 1:
            idx = np.argmax(ar)
        elif direction == -1:       
            idx = np.argmin(ar)
    return ar[idx]
    
def support_resistance(data,margin,level):    
    acres = list(np.sort(data[['High','Low']].stack()))
    n = 0
    while n != len(centers):
        n = len(centers)
        centers = []
        for i,value in enumerate(acres):
            diff = list(np.abs(acres[i] - acres)/acres)
            cluster = [x for (x,y) in zip(acres,diff) if y < margin] #acres[diff<margin]
            strength = len(cluster)
            if strength>=level:
                centers.extend([np.sum(cluster)/strength])
        acres = centers
    centers = np.sort(centers)
    print(centers.max())
    for j in data.index:
        data.loc[j,'Resistance'] = nearest_resup(data.loc[j]['High'],centers,1)
        data.loc[j,'Support'] = nearest_resup(data.loc[j]['Low'],centers,-1)        
    data = data.join(pd.Series(data=data['High']/data['Resistance'],index=data.index,name='Resistance_Ratio'))
    data = data.join(pd.Series(data=data['Low']/data['Support'],index=data.index,name='Support_Ratio'))
    return data, centers 

def ease_of_movement(data,time):
    distance = ((data['High']+data['Low'])/2) - ((data['High'].shift(1)+data['Low'].shift(1))/2)
    box_rat = (data['Volume']/data['Volume'].rolling(window=200).mean())/(data['High']-data['Low'])
    data = data.join(pd.Series(data=distance/box_rat,index=data.index,name=str(time)+'_Ease_of_Movement').rolling(window=time).mean())
    return data
    
    1-Period EMV = ((H + L)/2 - (Prior H + Prior L)/2) / ((V/100,000,000)/(H - L))

stk = typ(stk)
stk = vwap(stk,200)
stk = vwap_ratios(stk)
df, c = support_resistance(stk,.001,5)

        
    
    
