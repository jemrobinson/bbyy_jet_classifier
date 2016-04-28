from . import BaseStrategy
from ..adaptors import root2python
import logging
import numpy as np
from root_numpy import rec2array
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.externals import joblib
from sklearn.metrics import classification_report

class sklBDT(BaseStrategy) :
  default_output_location = "output/sklBDT"
  classifier_range = ( 0.0, 1.0 )


  def train_and_test( self, training_fraction ):
    # -- Construct array of features (X) and array of categories (y)
    X = rec2array( np.concatenate(( self.correct_no_weights, self.incorrect_no_weights )) )
    y = np.concatenate(( np.zeros(self.correct_no_weights.shape[0]), np.ones(self.incorrect_no_weights.shape[0]) ))
    w = rec2array( np.concatenate(( self.correct_weights_only, self.incorrect_weights_only )) )

    # -- Construct training and test datasets, automatically permuted
    X_train, X_test, y_train, y_test, w_train, w_test = train_test_split( X, y, w, train_size=training_fraction )

    # -- ANOVA for feature selection (please, know what you're doing)
    self.feature_selection( X_train, y_train, self.correct_no_weights.dtype.names, 5 )

    # -- Train:
    logging.getLogger("sklBDT::Train").info( "Training..." )
    classifier = GradientBoostingClassifier(n_estimators=200, min_samples_split=2, max_depth=10, verbose=1)
    classifier.fit( X_train, y_train, sample_weight=w_train )

    # -- Test:
    logging.getLogger("sklBDT::Train").info( "Testing..." )
    logging.getLogger("sklBDT::Train").info( "Training accuracy = {:.2f}%".format(100 * classifier.score( X_train, y_train, sample_weight=w_train)) )
    [ logging.getLogger("sklBDT::Train").info(l) for l in classification_report( y_train, classifier.predict(X_train), target_names=["correct","incorrect"], sample_weight=w_train ).splitlines() ]
    logging.getLogger("sklBDT::Train").info( "Testing accuracy = {:.2f}%".format(100 * classifier.score( X_test, y_test, sample_weight=w_test)) )
    [ logging.getLogger("sklBDT::Train").info(l) for l in classification_report( y_test, classifier.predict(X_test), target_names=["correct","incorrect"], sample_weight=w_test ).splitlines() ]

    # -- Get list of variables and classifier scores
    variables = [ (k, root2python.CHAR_2_TYPE[v]) for k,v in self.variable_dict.items()+[("weight","F"),("classID","I"),("classifier","F")] if k != "event_weight" ]
    classifier_score_test = classifier.predict_proba(X_test)[:, 0]
    classifier_score_training = classifier.predict_proba(X_train)[:, 0]

    # -- Construct numpy.ndarrays of test and training events
    test_events_with_score = np.hstack( ( X_test, w_test.reshape((-1,1)), y_test.reshape((-1,1)), classifier_score_test.reshape((-1,1)) ))
    training_events_with_score = np.hstack( (X_train, w_train.reshape((-1,1)), y_train.reshape((-1,1)), classifier_score_training.reshape((-1,1)) ))

    # -- Separate into sliced lists of test and training events
    test_events_array_sliced = [ test_events_with_score[...,idx].astype(variable[1]) for idx, variable in enumerate( variables ) ]
    training_events_array_sliced = [ training_events_with_score[...,idx].astype(variable[1]) for idx, variable in enumerate( variables ) ]

    # -- Recombine into record arrays for plotting
    self.test_events = np.rec.fromarrays( test_events_array_sliced, names=[ x[0] for x in variables ] )
    self.training_events = np.rec.fromarrays( training_events_array_sliced, names=[ x[0] for x in variables ] )

    # -- Dump output to pickle
    logging.getLogger("sklBDT::Train").info( "Writing output to disk..." )
    self.ensure_directory( "{}/pickle/".format(self.output_directory) )
    joblib.dump( classifier, "{}/pickle/sklBDT_output.pkl".format(self.output_directory) )


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

    # -- Plot support and return names of top features
    logging.getLogger("sklBDT").info( "The {} most important features are {}".format(k, [f for (s, f) in sorted(zip(tf.scores_, features), reverse=True)][:k] ) )


  def test_only( self ) :
    # -- Construct array of features (X_test) and array of categories (y_test)
    X_test = rec2array( np.concatenate(( self.correct_no_weights, self.incorrect_no_weights )) )
    y_test = np.concatenate(( np.zeros(self.correct_no_weights.shape[0]), np.ones(self.incorrect_no_weights.shape[0]) ))
    w_test = rec2array( np.concatenate(( self.correct_weights_only, self.incorrect_weights_only )) )

    # -- Load scikit classifier
    classifier = joblib.load( "{}/pickle/sklBDT_output.pkl".format(self.output_directory) )

    # -- Get list of variables and classifier scores
    variables = [ (k, root2python.CHAR_2_TYPE[v]) for k,v in self.variable_dict.items()+[("weight","F"),("classID","I"),("classifier","F")] if k != "event_weight" ]
    classifier_score_test = classifier.predict_proba(X_test)[:, 0]

    # -- Construct sliced arrays of test events with classifier score
    test_events_with_score = np.hstack( ( X_test, w_test.reshape((-1,1)), y_test.reshape((-1,1)), classifier_score_test.reshape((-1,1)) ))
    test_events_array_sliced = [ test_events_with_score[...,idx].astype(variable[1]) for idx, variable in enumerate( variables ) ]
    test_events_overall = np.rec.fromarrays( test_events_array_sliced, names=[ x[0] for x in variables ] )

    # -- Construct record arrays of correct/incorrect events
    logging.getLogger("sklBDT::Test").info( "Constructing record arrays of correct/incorrect events..." )
    self.test_correct_events = test_events_overall[ test_events_overall["classID"] == 0 ]
    self.test_incorrect_events = test_events_overall[ test_events_overall["classID"] == 1 ]
