import cPickle
import logging
import os
from . import BaseStrategy
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.externals import joblib
from sklearn.metrics import classification_report


class sklBDT(BaseStrategy):
    """
    Strategy using a BDT from scikit-learn
    """
    default_output_location = os.path.join("output", "sklBDT")

    def train(self, X_train, y_train, w_train, classification_variables, variable_dict):
        """
        Definition:
        -----------
            Training method for sklBDT; it pickles the model into the "pickle" sub-folder

        Args:
        -----
            X_train = the features matrix with events for training, of dimensions (n_events, n_features)
            y_train = the target array with events for training, of dimensions (n_events)
            w_train = the array of weights for training events, of dimensions (n_events)
            classification_variables = list of names of variables used for classification
            variable_dict = ordered dict, mapping all the branches from the TTree to their type
        """
        # -- Train:
        logging.getLogger("sklBDT::train").info("Training...")
        classifier = GradientBoostingClassifier(n_estimators=200, min_samples_split=2, max_depth=10, verbose=1)
        classifier.fit(X_train, y_train, sample_weight=w_train)

        # -- Dump output to pickle
        self.ensure_directory("{}/pickle/".format(self.output_directory))
        joblib.dump(classifier, "{}/pickle/sklBDT_clf.pkl".format(self.output_directory), protocol=cPickle.HIGHEST_PROTOCOL)

        self.ensure_directory(os.path.join(self.output_directory, "pickle"))
        joblib.dump(classifier, os.path.join(self.output_directory, "pickle", "sklBDT_clf.pkl"), protocol=cPickle.HIGHEST_PROTOCOL)

    def test(self, X, y, w, classification_variables, process):
        """
        Definition:
        -----------
            Testing method for sklBDT; it loads the latest model from the "pickle" sub-folder

        Args:
        -----
            X = the features matrix with events to test performance on, of dimensions (n_events, n_features)
            y = the target array with events to test performance on, of dimensions (n_events)
            w = the array of weights of the events to test performance on, of dimensions (n_events)
            process = string to identify whether we are evaluating performance on the train or test set, usually "training" or "testing"
            classification_variables = list of names of variables used for classification

        Returns:
        --------
            yhat = the array of BDT outputs corresponding to the P(signal), of dimensions (n_events)
        """
        logging.getLogger("sklBDT::test").info("Evaluating Performance...")

        # -- Load scikit classifier
        classifier = joblib.load("{}/pickle/sklBDT_clf.pkl".format(self.output_directory))

        # -- Get classifier predictions
        yhat = classifier.predict_proba(X)[:, 1]

        # -- Load scikit classifier
        classifier = joblib.load(os.path.join(self.output_directory, "pickle", "sklBDT_clf.pkl"))

        # -- Log classification scores
        logging.getLogger("sklBDT::test").info("{} accuracy = {:.2f}%".format(process, 100 * classifier.score(X, y, sample_weight=w)))
        logging.getLogger("sklBDT::test").info(classification_report(y, classifier.predict(X), target_names=["correct", "incorrect"], sample_weight=w))

        return yhat
