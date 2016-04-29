from . import BaseStrategy
from ..adaptors import root2python
import logging
import numpy as np
#from root_numpy import rec2array
#from sklearn.cross_validation import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.externals import joblib
from sklearn.metrics import classification_report

class sklBDT(BaseStrategy) :
  default_output_location = "output/sklBDT"

  def train( self, X_train, y_train, w_train ):

    # -- Train:
    logging.getLogger("sklBDT").info( "Training..." )
    classifier = GradientBoostingClassifier(n_estimators=200, min_samples_split=2, max_depth=10, verbose=1)
    classifier.fit( X_train, y_train, sample_weight=w_train )

    # -- Dump output to pickle
    self.ensure_directory( "{}/pickle/".format(self.output_directory) )
    joblib.dump( classifier, "{}/pickle/sklBDT_output.pkl".format(self.output_directory) )


  def test( self, X, y, w, process) :
    logging.getLogger("sklBDT").info( "Evaluating Performance..." )

    # -- Load scikit classifier
    classifier = joblib.load( "{}/pickle/sklBDT_output.pkl".format(self.output_directory) )

    # -- Get classifier predictions
    yhat = classifier.predict_proba(X)[:, 1]

    # -- Log classification scores
    logging.getLogger("sklBDT").info( "{} accuracy = {:.2f}%".format(process, 100 * classifier.score( X, y, sample_weight=w)) )
    logging.getLogger("sklBDT").info( classification_report( y, classifier.predict(X), target_names=["correct","incorrect"], sample_weight=w) )
    
    return yhat