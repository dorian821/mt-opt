import pandas as pd
import numpy as np
import os
import datetime as dt
import pandas_datareader.data as web
from yahoo_finance import Share
import scipy as sp
from datetime import datetime
import calendar
import itertools as it
from scipy import signal as sig
from scipy import stats
import matplotlib.pyplot as plt
import sys
import gc
import pickle
from mt_auto import mt_auto as mt
from sklearn import tree
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from random import randint

features =["BBAND_Lower_3p_Slope",
			"BBAND_Rank",
			"BBANDS_PINCH",
			"BBANDS_XPAND",
			"%K_STO",
			"%D_STO",
			"D1_DIR__STO_Bool",
			"D1_FAUXSTO",
			"ABSOL_UP_D1_STO",
			"%K_3PER_SLP_STO",
			"FIRST_UP_D1_%K_3PER_SLP_STO",
			"FIRST_DOWN_D1_%K_3PER_SLP_STO",
			"BLOWEST_20",
			"COUNT_BLOWEST_20",
			"BLOWEST_5",
			"COUNT_BLOWEST_5",
			"CCI_14",
			"CCI_40",
			"CCI_89",
			"RSI_20",
			"RSI_40",
			"RSI_89",
			"MACD_13-27",
			"Signal_EMA_8_Normalized",
			"SMA_5_3_Slope",
			"SMA_10_Over_SMA_5",
			"SMA_21_Over_SMA_5",
			"SMA_34_Over_SMA_5",
			"SMA_55_Over_SMA_5",
			"SMA_89_Over_SMA_5",
			"SMA_144_Over_SMA_5",
			"SMA_233_Over_SMA_5",
			"SMA_5_X-Over_SMA_10",
			"SMA_5_X-Over_SMA_21",
			"SMA_5_X-Over_SMA_34",
			"SMA_5_X-Over_SMA_55",
			"SMA_5_X-Over_SMA_89",
			"SMA_5_X-Over_SMA_144",
			"SMA_5_X-Over_SMA_233",
			"SMA_10_X-Over_SMA_5",
			"SMA_10_X-Over_SMA_21",
			"SMA_10_X-Over_SMA_34",
			"SMA_10_X-Over_SMA_55",
			"SMA_10_X-Over_SMA_89",
			"SMA_10_X-Over_SMA_144",
			"SMA_10_X-Over_SMA_233",
			"SMA_21_X-Over_SMA_5",
			"SMA_21_X-Over_SMA_10",
			"SMA_21_X-Over_SMA_34",
			"SMA_21_X-Over_SMA_55",
			"SMA_21_X-Over_SMA_89",
			"SMA_21_X-Over_SMA_144",
			"SMA_21_X-Over_SMA_233",
			"SMA_34_X-Over_SMA_5",
			"SMA_34_X-Over_SMA_10",
			"SMA_34_X-Over_SMA_21",
			"SMA_34_X-Over_SMA_55",
			"SMA_34_X-Over_SMA_89",
			"SMA_34_X-Over_SMA_144",
			"SMA_34_X-Over_SMA_233",
			"SMA_55_X-Over_SMA_5",
			"SMA_55_X-Over_SMA_10",
			"SMA_55_X-Over_SMA_21",
			"SMA_55_X-Over_SMA_34",
			"SMA_55_X-Over_SMA_89",
			"SMA_55_X-Over_SMA_144",
			"SMA_55_X-Over_SMA_233",
			"SMA_89_X-Over_SMA_5",
			"SMA_89_X-Over_SMA_10",
			"SMA_89_X-Over_SMA_21",
			"SMA_89_X-Over_SMA_34",
			"SMA_89_X-Over_SMA_55",
			"SMA_89_X-Over_SMA_144",
			"SMA_89_X-Over_SMA_233",
			"SMA_144_X-Over_SMA_5",
			"SMA_144_X-Over_SMA_10",
			"SMA_144_X-Over_SMA_21",
			"SMA_144_X-Over_SMA_34",
			"SMA_144_X-Over_SMA_55",
			"SMA_144_X-Over_SMA_89",
			"SMA_144_X-Over_SMA_233",
			"SMA_233_X-Over_SMA_5",
			"SMA_233_X-Over_SMA_10",
			"SMA_233_X-Over_SMA_21",
			"SMA_233_X-Over_SMA_34",
			"SMA_233_X-Over_SMA_55",
			"SMA_233_X-Over_SMA_89",
			"SMA_233_X-Over_SMA_144",
			"EMA_10_Over_EMA_5",
			"EMA_21_Over_EMA_5",
			"EMA_34_Over_EMA_5",
			"EMA_55_Over_EMA_5",
			"EMA_89_Over_EMA_5",
			"EMA_144_Over_EMA_5",
			"EMA_233_Over_EMA_5",
			"EMA_5_X-Over_EMA_10",
			"EMA_5_X-Over_EMA_21",
			"EMA_5_X-Over_EMA_34",
			"EMA_5_X-Over_EMA_55",
			"EMA_5_X-Over_EMA_89",
			"EMA_5_X-Over_EMA_144",
			"EMA_5_X-Over_EMA_233",
			"EMA_10_X-Over_EMA_5",
			"EMA_10_X-Over_EMA_21",
			"EMA_10_X-Over_EMA_34",
			"EMA_10_X-Over_EMA_55",
			"EMA_10_X-Over_EMA_89",
			"EMA_10_X-Over_EMA_144",
			"EMA_10_X-Over_EMA_233",
			"EMA_21_X-Over_EMA_5",
			"EMA_21_X-Over_EMA_10",
			"EMA_21_X-Over_EMA_34",
			"EMA_21_X-Over_EMA_55",
			"EMA_21_X-Over_EMA_89",
			"EMA_21_X-Over_EMA_144",
			"EMA_21_X-Over_EMA_233",
			"EMA_34_X-Over_EMA_5",
			"EMA_34_X-Over_EMA_10",
			"EMA_34_X-Over_EMA_21",
			"EMA_34_X-Over_EMA_55",
			"EMA_34_X-Over_EMA_89",
			"EMA_34_X-Over_EMA_144",
			"EMA_34_X-Over_EMA_233",
			"EMA_55_X-Over_EMA_5",
			"EMA_55_X-Over_EMA_10",
			"EMA_55_X-Over_EMA_21",
			"EMA_55_X-Over_EMA_34",
			"EMA_55_X-Over_EMA_89",
			"EMA_55_X-Over_EMA_144",
			"EMA_55_X-Over_EMA_233",
			"EMA_89_X-Over_EMA_5",
			"EMA_89_X-Over_EMA_10",
			"EMA_89_X-Over_EMA_21",
			"EMA_89_X-Over_EMA_34",
			"EMA_89_X-Over_EMA_55",
			"EMA_89_X-Over_EMA_144",
			"EMA_89_X-Over_EMA_233",
			"EMA_144_X-Over_EMA_5",
			"EMA_144_X-Over_EMA_10",
			"EMA_144_X-Over_EMA_21",
			"EMA_144_X-Over_EMA_34",
			"EMA_144_X-Over_EMA_55",
			"EMA_144_X-Over_EMA_89",
			"EMA_144_X-Over_EMA_233",
			"EMA_233_X-Over_EMA_5",
			"EMA_233_X-Over_EMA_10",
			"EMA_233_X-Over_EMA_21",
			"EMA_233_X-Over_EMA_34",
			"EMA_233_X-Over_EMA_55",
			"EMA_233_X-Over_EMA_89",
			"EMA_233_X-Over_EMA_144",
			"CCI_40_High_Divergence",
			"CCI_40_High_Div_Count",
			"CCI_40_Low_Divergence",
			"CCI_40_Low_Div_Count",
			"CCI_14_High_Divergence",
			"CCI_14_High_Div_Count",
			"CCI_14_Low_Divergence",
			"CCI_14_Low_Div_Count",
			"MACD_13-27_High_Divergence",
			"MACD_13-27_High_Div_Count",
			"MACD_13-27_Low_Divergence",
			"MACD_13-27_Low_Div_Count",
			"RSI_40_High_Divergence",
			"RSI_40_High_Div_Count",
			"RSI_40_Low_Divergence",
			"RSI_40_Low_Div_Count",
			"RSI_20_High_Divergence",
			"RSI_20_High_Div_Count",
			"RSI_20_Low_Divergence",
			"RSI_20_Low_Div_Count"]

def normalizer_bool(data):
	new_data = pd.DataFrame(index=data.index)			 
	for col in data.columns:
		normalized = pd.Series(data=np.where(data[col]==True,.99,.01),index=data.index, name=col+'_Norm')
		new_data = new_data.join(normalized)		       
	return new_data
			
			
def RevTraverseTree(tree, node, rules):
	'''
	Traverase an skl decision tree from a node (presumably a leaf node)
	up to the top, building the decision rules. The rules should be
	input as an empty list, which will be modified in place. The result
	is a nested list of tuples: (feature, direction (left=-1), threshold).  
	The "tree" is a nested list of simplified tree attributes:
	[split feature, split threshold, left node, right node]
	'''
	# now find the node as either a left or right child of something
	# first try to find it as a left node
	try:
		prevnode = tree[2].index(node)
		leftright = -1
	except ValueError:
		# failed, so find it as a right node - if this also causes an exception, something's really f'd up
		prevnode = tree[3].index(node)
		leftright = 1
	# now let's get the rule that caused prevnode to -> node
	rules.append((tree[0][prevnode],leftright,tree[1][prevnode]))
	# if we've not yet reached the top, go up the tree one more step
	if prevnode != 0:
		RevTraverseTree(tree, prevnode, rules)
	
		
def explore_tree(model,train_x):
	n = len(train_x)
	rule_book = {}
	leaves = model.tree_.children_left == -1
	leaves = np.arange(0,model.tree_.node_count)[leaves]

	# loop through each leaf and figure out the data in it
	leaf_observations = np.zeros((n,len(leaves)),dtype=bool)
	# build a simpler tree as a nested list: [split feature, split threshold, left node, right node]
	thistree = [model.tree_.feature.tolist()]
	thistree.append(model.tree_.threshold.tolist())
	thistree.append(model.tree_.children_left.tolist())
	thistree.append(model.tree_.children_right.tolist())
	# get the decision rules for each leaf node & apply them
	
	for (ind,nod) in enumerate(leaves):
		# get the decision rules in numeric list form
		rules = []
		RevTraverseTree(thistree, nod, rules)
		rule_book[nod] = rules
		# convert & apply to the data by sequentially &ing the rules
		thisnode = np.ones(n,dtype=bool)
		for rule in rules:
			#print(train_x.columns[rule[0]])
			if rule[1] == 1:
				thisnode = np.logical_and(thisnode,train_x.iloc[:][train_x.columns[rule[0]]] > rule[2])
			else:
				thisnode = np.logical_and(thisnode,train_x.iloc[:][train_x.columns[rule[0]]] <= rule[2])
		# get the observations that obey all the rules - they are the ones in this leaf node
		leaf_observations[:,ind] = thisnode
	return leaf_observations, rule_book
	
def random_features(features,mn,mx):
	m = np.minimum(len(features),mx)
	n_features = randint(mn,m)
	feature_list = []
	for n in np.arange(n_features):
		i = 0
		while i < 1:
			x = randint(0,len(features)-1)
			if not features[x] in feature_list:
				feature_list.append(features[x])
				i = 1
	return feature_list
	
def load_pickle(name):
	output = open(name, 'rb')
	# disable garbage collector
	gc.disable()
	mydict = pickle.load(output)
	# enable garbage collector again
	gc.enable()
	output.close()
	return mydict

def decompressor(direct,symb):
	master_tree = {}
	master_imp = {}
	master_val = {}
	folders = os.listdir(direct.forest_dir + symb + '\\')
	for folder in folders:		
		files = os.listdir(direct.forest_dir + symb + '\\' + folder + '\\')
		forest = str([f for f in files if 'crit' in f][0])
		imp = str([f for f in files if 'imp' in f][0])
		val = str([f for f in files if 'val' in f][0])		
		forest = load_pickle(direct.forest_dir + symb + '\\' + folder + '\\' + forest)
		imp = load_pickle(direct.forest_dir + symb + '\\' + folder + '\\' + imp)
		val = load_pickle(direct.forest_dir + symb + '\\' + folder + '\\' + val)
		for t in forest.keys():
			tree = forest[t]
			imps = imp[t]
			vals = val[t]
			print(tree.keys(),imps.keys(),vals)
			if len(tree) > 0:
				for b in tree.keys():
					master_tree[len(master_tree)+1] = tree[b]
					master_imp[len(master_imp)+1] = imps[b]
					master_val[len(master_val)+1] = vals[b]
	return master_tree, master_imp, master_val
	

def balance_data(data,col,target):
	dict = {label:[len(category), category] for label, category in data.groupby(by=[col],axis=0)}
	threshold = min(dict[label][0] for label in target)	
	new_data = pd.DataFrame()
	for k in dict.keys(): 
		if  dict[k][0] > threshold:
			resample = dict[k][1].sample(n=threshold)
		else:
			resample = dict[k][1]
		new_data = pd.concat([new_data,resample],axis=0)
	return new_data
			
		
	
	
	
class decision_tree():
	
	def __init__(self,symb,direct,range,max_depth,min_samples_split,min_samples_leaf):
		self.range = range
		self.max_depth = max_depth
		self.min_samples_split = min_samples_split
		self.min_samples_leaf = min_samples_leaf
		self.symb = symb
		self.direct = direct
		
	def prep_data(self,training_ratio,direction):
		stk = mt.dicts(self.direct).dnc_dict()[self.symb]
		stk = stk[300:]
		stk = balance_data(data=stk,col='Bool_'+str(self.range)+'_day_forward_typ_ratio',target=[-1,1])
		stk_types = stk.columns.to_series().groupby(stk.dtypes).groups
		stk_types = {k.name: v for k, v in stk_types.items()}
		for key in stk_types.keys():
			stk_type = stk[stk_types[key]]
			if key == 'bool':
				stk_type = normalizer_bool(stk_type)
				stk[stk_types[key]] = stk_type
		train = stk[:stk.index[int(len(stk)*training_ratio)]]
		test = stk[stk.index[int(len(stk)*training_ratio)]:]
		train_target = train['Bool_'+str(self.range)+'_day_forward_typ_ratio']
		test_target = test['Bool_'+str(self.range)+'_day_forward_typ_ratio']
		feature_list = random_features(features,20,1000)
		#print(len(feature_list))
		train = train[feature_list]
		test = test[feature_list]
		return train, train_target, test, test_target
	
	def compiler(self):
		model = tree.DecisionTreeClassifier(max_depth=self.max_depth, min_samples_split=self.min_samples_split,min_samples_leaf=self.min_samples_leaf,class_weight={-1:2,1:2})
		return model
	
	def fitter(self,model,X,Y):
		clf = model.fit(X, Y)
		return clf
	
	def acc_tester(self,model,X,Y):
		#clf = model.predict(X)
		#print(clf)
		#y = [int(y) for y in Y]
		#accuracy = np.sum(np.abs(clf-Y))/len(Y)
		feature_imp = zip(X.columns, model.feature_importances_)
		return feature_imp
		
	def plotter(self,model,X):
		importances = model.feature_importances_
		indices = np.argsort(importances)[::-1]
		plt.figure()
		plt.title("Feature importances")
		plt.bar(range(X.shape[1]), importances[indices],
			   color="r", align="center")
		plt.xticks(range(X.shape[1]), indices)
		plt.xlim([-1, X.shape[1]])
		plt.show()
		return self
		
	def grad_boost(self,X,Y):
		clf = GradientBoostingClassifier(n_estimators=500, n_classes=3, learning_rate=.01,
								max_depth=3, random_state=0).fit(X, Y)
		return clf
		
	def random_forest(self,X,Y):
		clf = RandomForestClassifier(n_estimators=100, max_depth=None,
									min_samples_split=50, random_state=0).fit(X, Y)
		return clf
		
		
	def extra_trees(self,X,Y):
		forest = ExtraTreesClassifier(n_estimators=250,
										random_state=0).fit(X, Y)
		return forest
		
		
	def plot_forest(self,forest,X):
		importances = forest.feature_importances_
		std = np.std([tree.feature_importances_ for tree in forest.estimators_],
					 axis=0)
		indices = np.argsort(importances)[::-1]

		# Print the feature ranking
		print("Feature ranking:")

		for f in range(X.shape[1]):
			print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

		# Plot the feature importances of the forest
		plt.figure()
		plt.title("Feature importances")
		plt.bar(range(X.shape[1]), importances[indices],
			   color="r", yerr=std[indices], align="center")
		plt.xticks(range(X.shape[1]), indices)
		plt.xlim([-1, X.shape[1]])
		plt.show()
		return self
		
		

			


	def raw(self):
		# Create the dataset
		rng = np.random.RandomState(1)
		X = np.linspace(0, 6, 100)[:, np.newaxis]
		y = np.sin(X).ravel() + np.sin(6 * X).ravel() + rng.normal(0, 0.1, X.shape[0])
		# Fit regression model
		regr_1 = DecisionTreeRegressor(max_depth=4)

		regr_2 = AdaBoostRegressor(DecisionTreeRegressor(max_depth=4),
								  n_estimators=300, random_state=rng)

		regr_1.fit(X, y)
		regr_2.fit(X, y)

		# Predict
		y_1 = regr_1.predict(X)
		y_2 = regr_2.predict(X)

		# Plot the results
		plt.figure()
		plt.scatter(X, y, c="k", label="training samples")
		plt.plot(X, y_1, c="g", label="n_estimators=1", linewidth=2)
		plt.plot(X, y_2, c="r", label="n_estimators=300", linewidth=2)
		plt.xlabel("data")
		plt.ylabel("target")
		plt.title("Boosted Decision Tree Regression")
		plt.legend()
		plt.show()
		return self
