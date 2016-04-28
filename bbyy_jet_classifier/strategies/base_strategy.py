from ..adaptors import root2python
from root_numpy import root2rec
import os
import numpy as np
from root_numpy import rec2array
from sklearn.cross_validation import train_test_split
import logging



class BaseStrategy(object) :

  def __init__( self, output_directory ) :
    self.output_directory = output_directory if output_directory is not None else self.default_output_location
    self.ensure_directory( self.output_directory )


  def ensure_directory( self, directory ) :
    if not os.path.exists(directory):
      os.makedirs(directory)


  def load_data( self, input_filename, correct_treename, incorrect_treename, excluded_variables, training_fraction ) :
    self.variable_dict = root2python.get_branch_info( input_filename, correct_treename, excluded_variables )
    self.correct_array = root2rec( input_filename, correct_treename, branches=self.variable_dict.keys() )
    self.incorrect_array = root2rec( input_filename, incorrect_treename, branches=self.variable_dict.keys() )
    #self.classification_variables = sorted( [ name for name in self.variable_dict.keys() if name != "event_weight" ] ) #WHY SORTED?
    self.classification_variables = [ name for name in self.variable_dict.keys() if name != "event_weight" ]

    self.correct_no_weights = self.correct_array[self.classification_variables]
    self.incorrect_no_weights = self.incorrect_array[self.classification_variables]
    self.correct_weights_only = self.correct_array[ ["event_weight"] ]
    self.incorrect_weights_only = self.incorrect_array[ ["event_weight"] ]

    # -- Construct array of features (X) and array of categories (y)
    X = rec2array( np.concatenate(( self.correct_no_weights, self.incorrect_no_weights )) )
    y = np.concatenate(( np.zeros(self.correct_no_weights.shape[0]), np.ones(self.incorrect_no_weights.shape[0]) ))
    w = rec2array( np.concatenate(( self.correct_weights_only, self.incorrect_weights_only )) )

    # -- Construct training and test datasets, automatically permuted
    X_train, X_test, y_train, y_test, w_train, w_test = train_test_split( X, y, w, train_size=training_fraction )

    # -- ANOVA for feature selection (please, know what you're doing)
    self.feature_selection( X_train, y_train, self.correct_no_weights.dtype.names, 5 )

    return X_train, X_test, y_train, y_test, w_train, w_test

  def feature_selection(self, X_train, y_train, features, k ):
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
    # print("Shape =", Xt.shape)

    # -- Plot support and return names of top features
    logging.getLogger("RunClassifier").info( "The {} most important features are {}".format(k, [f for (s, f) in sorted(zip(tf.scores_, features), reverse=True)][:k] ) )
    # plt.imshow(tf.get_support().reshape(2, -1), interpolation="nearest", cmap=plt.cm.Blues)
    # plt.show()

  def train( self, X_train, y_train, w_train ) :
    raise NotImplementedError( "Must be implemented by child class!" )


  def test( self, X_test ) :
    raise NotImplementedError( "Must be implemented by child class!" )
