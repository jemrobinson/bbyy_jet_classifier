from . import BaseStrategy
from ..adaptors import root2python
import logging
import numpy as np
import cPickle
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.externals import joblib
from sklearn.metrics import classification_report

class sklBDT(BaseStrategy) :
  default_output_location = "output/sklBDT"

  def train(self, X_train, y_train, w_train):
    '''
    Definition:
    -----------
        Training method for sklBDT; it pickles the model into the 'pickle' sub-folder
    Args:
    -----
        X_train = the features matrix with events for training, of dimensions (n_events, n_features) 
        y_train = the target array with events for training, of dimensions (n_events)
        w_train = the array of weights for training events, of dimensions (n_events)
    '''
    # -- Train:
    logging.getLogger("sklBDT").info("Training...")
    classifier = GradientBoostingClassifier(n_estimators=200, min_samples_split=2, max_depth=10, verbose=1)
    classifier.fit(X_train, y_train, sample_weight=w_train)

    # -- Dump output to pickle
    self.ensure_directory("{}/pickle/".format(self.output_directory))
    joblib.dump(classifier, "{}/pickle/sklBDT_clf.pkl".format(self.output_directory), protocol=cPickle.HIGHEST_PROTOCOL)


  def test(self, X, y, w, process):
    '''
    Definition:
    -----------
        Testing method for sklBDT; it loads the latest model from the 'pickle' sub-folder
    Args:
    -----
        X = the features matrix with events to test performance on, of dimensions (n_events, n_features) 
        y = the target array with events to test performance on, of dimensions (n_events)
        w = the array of weights of the events to test performance on, of dimensions (n_events)
        process = string to identify whether we are evaluating performance on the train or test set, usually 'training' or 'testing'
    Returns:
    --------
        yhat = the array of BDT outputs corresponding to the P(signal), of dimensions (n_events)
    '''
    logging.getLogger("sklBDT").info("Evaluating Performance...")

    # -- Load scikit classifier
    classifier = joblib.load("{}/pickle/sklBDT_clf.pkl".format(self.output_directory))

    # -- Get classifier predictions
    yhat = classifier.predict_proba(X)[:, 1]

    # -- Log classification scores
    logging.getLogger("sklBDT").info("{} accuracy = {:.2f}%".format(process, 100 * classifier.score(X, y, sample_weight=w)) )
    logging.getLogger("sklBDT").info(classification_report(y, classifier.predict(X), target_names=["correct", "incorrect"], sample_weight=w))
    
    return yhat