import logging
from collections import OrderedDict

import numpy as np
from root_numpy import root2rec, rec2array
from sklearn.cross_validation import train_test_split

TYPE_2_CHAR = {"<i4":"I", "<f8":"D", "<f4":"F"}

def load_data(input_filename, correct_treename, incorrect_treename, excluded_variables, training_fraction):
	'''
	Definition:
	-----------
		Data handling function that loads in .root files and turns them into ML-ready python objects

	Args:
	-----
		input_filename = string, the path to the input root file
		correct_treename = string, the name of the TTree that contains signal examples
		incorrect_treename = string, the name of the TTree that contains background examples
		excluded_variables = list of strings, names of branches not to use for training
		training_fraction = float between 0 and 1, fraction of examples to use for training
		
	Returns:
	--------
		classification_variables = list of names of variables used for classification
		variable_dict = ordered dict, mapping all the branches from the TTree to their type
		X_train = ndarray of dim (# training examples, # features)
		X_test = ndarray of dim (# testing examples, # features)
		y_train = array of dim (# training examples) with target values
		y_test = array of dim (# testing examples) with target values
		w_train = array of dim (# training examples) with event weights
		w_test = array of dim (# testing examples) with event weights
		mHmatch_test = output of binary decision based on jet pair with closest m_jb to 125GeV
		pThigh_test = output of binary decision based on jet with highest pT
	'''
	correct_recarray = root2rec(input_filename, correct_treename) 
	incorrect_recarray = root2rec(input_filename, incorrect_treename) 
	variable_dict = OrderedDict(((v_name, TYPE_2_CHAR[v_type]) for v_name, v_type in correct_recarray.dtype.descr))
	classification_variables = [name for name in variable_dict.keys() if name not in ["event_weight", "idx_by_mH", "idx_by_pT"]]

	correct_recarray_feats = correct_recarray[classification_variables]
	incorrect_recarray_feats = incorrect_recarray[classification_variables]

	# -- Construct array of features (X) and array of categories (y)
	X = rec2array(np.concatenate((correct_recarray_feats, incorrect_recarray_feats)) )
	y = np.concatenate((np.ones(correct_recarray_feats.shape[0]), np.zeros(incorrect_recarray_feats.shape[0]) ))
	w = np.concatenate((correct_recarray['event_weight'], incorrect_recarray['event_weight']))
	mHmatch = np.concatenate((correct_recarray['idx_by_mH'] == 0, incorrect_recarray['idx_by_mH'] == 0))
	pThigh = np.concatenate((correct_recarray['idx_by_pT'] == 0, incorrect_recarray['idx_by_pT'] == 0))

	# -- Construct training and test datasets, automatically permuted
	X_train, X_test, y_train, y_test, w_train, w_test, _, mHmatch_test, _, pThigh_test = train_test_split(
		X, y, w, mHmatch, pThigh, train_size=training_fraction)

	# -- ANOVA for feature selection (please, know what you're doing)
	feature_selection(X_train, y_train, classification_variables, 5)

	return classification_variables, variable_dict, X_train, X_test, y_train, y_test, w_train, w_test, mHmatch_test, pThigh_test


def feature_selection(X_train, y_train, features, k):
	"""
	Definition:
	-----------
		!! ONLY USED FOR INTUITION, IT'S USING A LINEAR MODEL TO DETERMINE IMPORTANCE !!
		Gives an approximate ranking of variable importance and prints out the top k

	Args:
	-----
		X_train = matrix X of dimensions (n_train_events, n_features) for training
		y_train = array of truth labels {0, 1} of dimensions (n_train_events) for training
		features = names of features used for training in the order in which they were inserted into X
		k = int, the function will print the top k features in order of importance
	"""

	# -- Select the k top features, as ranked using ANOVA F-score
	from sklearn.feature_selection import SelectKBest, f_classif
	tf = SelectKBest(score_func=f_classif, k=k)
	Xt = tf.fit_transform( X_train, y_train)

	# -- Plot support and return names of top features
	logging.getLogger("RunClassifier").info( "The {} most important features are {}".format(k, [f for (s, f) in sorted(zip(tf.scores_, features), reverse=True)][:k] ) )
	# plt.imshow(tf.get_support().reshape(2, -1), interpolation="nearest", cmap=plt.cm.Blues)
	# plt.show()