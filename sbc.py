from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import RFECV
from sklearn.feature_selection import VarianceThreshold
from random import randint

x_data = SelectKBest(mutual_info_classif, k=randint(3,20)).fit_transform(x_data, y)

def scorer(estimator, X, y):
  mask = (prediction == 0) | (actual == prediction)
  return mask.sum()/len(prediction)

model = GaussianNB()
def lowvar_clip(model,data,target):
   selection = RFECV(model, step=1, scoring=scorer, cv=randint(3,20),n_jobs=4).fit(data,target)
   return selection

def var_check(data,threshold=0.0):
  selector = VarianceThreshold(threshold=0.0)
  selection = selector.fit_transform(data)
  return selection
    
