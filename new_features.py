
from numba import jit
#VWAP
@jit
def vwap(data):
  vwap = pd.Series(data=np.cumsum(v*(h+l)/2) / np.cumsum(v),index=data.index,name='VWAP')
  data = data.join(vwap)
  return data
