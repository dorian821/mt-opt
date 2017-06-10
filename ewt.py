
def find_peaks(series):
  forward pass rolling argmax with window=x
  backward pass rolling argmax with window=x
  

def find_bottoms(series):
  divide and conquer
  find all bottoms











def find length to next corresponding bottoms for peaks or peaks for bottoms
define corresponding bottom and peak
  peak corresponds to bottom or vv if range is close 
  range = the max distance (p or t) from neighboring peaks or bottoms
  
  
  what if we took the a peak and measured the standard deviation (using the linear regression as the mean) between the two points 
  of all the paths from it to all other inverts (peak is invert of bottom and vv)
  
  -- to measure complexity of path
  -- find all peaks
  -- find all bottoms
  -- for peak in peaks:
      x = linespace peak to bottom
      diffs = abs(val[i]-lin) for i in x.index
      std_dict[data.index[peak]] = stdev(diffs)
      
  -- we need to measure length and depth of path
  -- to measure length, for peak in peaks measure how far forward you have to go to find a long term bottom
  -- measure bottoms by how long they are valid for, i.e. how long they hold as a bottom
  -- measure peaks in the same way
  -- rule: a large degree peak corresponds to the next permanent bottom
  -- the back or forward range of a peak should inform the next bottom that it corresponds to
  
  i.e. we have a peak that has a back range of 10
    it corresponds to the next bottom with a forward range >=10?
  
  
  peak biased analysis
  for each peak find pair of corresponding bottoms
      
      
  
