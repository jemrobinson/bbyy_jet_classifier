from ..adaptors import root2python
from root_numpy import root2rec
import os
import numpy as np
from root_numpy import rec2array
from sklearn.cross_validation import train_test_split
import logging



class BaseStrategy(object):

	def __init__(self, output_directory):
		self.output_directory = output_directory if output_directory is not None else self.default_output_location
		self.ensure_directory( self.output_directory )


	def ensure_directory(self, directory):
		if not os.path.exists(directory):
			os.makedirs(directory)


	def load_data(self, input_filename, correct_treename, incorrect_treename, excluded_variables, training_fraction):
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
			X_train = ndarray of dim (# training examples, # features)
			X_test = ndarray of dim (# testing examples, # features)
			y_train = array of dim (# training examples) with target values
			y_test = array of dim (# testing examples) with target values
			w_train = array of dim (# training examples) with event weights
			w_test = array of dim (# testing examples) with event weights
			mHmatch_test = output of binary decision based on jet pair with closest m_jb to 125GeV
			pThigh_test = output of binary decision based on jet with highest pT
		'''
		self.variable_dict = root2python.get_branch_info(input_filename, correct_treename, excluded_variables)
		self.correct_array = root2rec(input_filename, correct_treename, branches=self.variable_dict.keys())
		self.incorrect_array = root2rec(input_filename, incorrect_treename, branches=self.variable_dict.keys())
		#self.classification_variables = sorted( [ name for name in self.variable_dict.keys() if name != "event_weight" ] ) #WHY SORTED?
		self.classification_variables = [name for name in self.variable_dict.keys() if name not in ["event_weight", "idx_by_mH", "idx_by_pT"]]

		self.correct_no_weights = self.correct_array[self.classification_variables]
		self.incorrect_no_weights = self.incorrect_array[self.classification_variables]

		# -- Construct array of features (X) and array of categories (y)
		X = rec2array(np.concatenate((self.correct_no_weights, self.incorrect_no_weights)) )
		y = np.concatenate((np.ones(self.correct_no_weights.shape[0]), np.zeros(self.incorrect_no_weights.shape[0]) ))
		w = np.concatenate((self.correct_array['event_weight'], self.incorrect_array['event_weight']))
		mHmatch = np.concatenate((self.correct_array['idx_by_mH'] == 0, self.incorrect_array['idx_by_mH'] == 0))
		pThigh = np.concatenate((self.correct_array['idx_by_pT'] == 0, self.incorrect_array['idx_by_pT'] == 0))

		# -- Construct training and test datasets, automatically permuted
		X_train, X_test, y_train, y_test, w_train, w_test, _, mHmatch_test, _, pThigh_test = train_test_split(
			X, y, w, mHmatch, pThigh, train_size=training_fraction)

		# -- ANOVA for feature selection (please, know what you're doing)
		self.feature_selection(X_train, y_train, self.correct_no_weights.dtype.names, 5)

		return X_train, X_test, y_train, y_test, w_train, w_test, mHmatch_test, pThigh_test


	def feature_selection(self, X_train, y_train, features, k):
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


	def train(self, X_train, y_train, w_train):
		raise NotImplementedError("Must be implemented by child class!")


	def test(self, X_test, y_test, w_test, process):
		raise NotImplementedError("Must be implemented by child class!")
