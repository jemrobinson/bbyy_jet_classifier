import cPickle
import logging
import os
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.externals import joblib
from sklearn.metrics import classification_report
from sklearn.grid_search import GridSearchCV
from . import BaseStrategy
from ..utils import ensure_directory

GRID_SEARCH = False

class sklBDT(BaseStrategy):
    """
    Strategy using a BDT from scikit-learn
    """

    def train(self, train_data, classification_variables, variable_dict, sample_name):
        """
        Definition:
        -----------
            Training method for sklBDT; it pickles the model into the "pickle" sub-folder

        Args:
        -----
            train_data = dictionary, containing "X", "y", "w" for the training set, where:
                X = ndarray of dim (# training examples, # features)
                y = array of dim (# training examples) with target values
                w = array of dim (# training examples) with event weights
            classification_variables = list of names of variables used for classification
            variable_dict = ordered dict, mapping all the branches from the TTree to their type
            sample_name = string that specifies the file name of the sample being trained on
        """
        # -- Train:
        logging.getLogger("skl_BDT").info("Training...")

        if GRID_SEARCH: # -- useful, I wouln't erase it completely
        # -- however, we should probably pass this as an argument instead of defining this on top of the file
            param_grid = { # change these parameters if you want to test different options
               'n_estimators' : [50, 100, 200, 300], 
               'min_samples_split' : [2, 5],
               'max_depth': [3, 5, 7, 10]
            }
            fit_params = {
                'sample_weight' : train_data['w']
            }
            metaclassifier = GridSearchCV(GradientBoostingClassifier(), param_grid=param_grid, fit_params=fit_params, 
                #scoring=event_accuracy, TO BE IMPLEMENTED! 
                cv=2, n_jobs=4, verbose=1)
            metaclassifier.fit(train_data['X'], train_data['y'])
            classifier = metaclassifier.best_estimator_
            print metaclassifier.best_params_

        else:
            classifier = GradientBoostingClassifier(
                n_estimators=300, 
                min_samples_split=0.5 * len(train_data["y"]), 
                max_depth=15, 
                verbose=1
                )
            classifier.fit(train_data["X"], train_data["y"], sample_weight=train_data["w"])

        # -- Dump output to pickle
        ensure_directory(os.path.join(self.output_directory, sample_name, self.name, "classifier"))
        joblib.dump(classifier, os.path.join(self.output_directory, sample_name, self.name, "classifier", "skl_BDT_clf.pkl"), protocol=cPickle.HIGHEST_PROTOCOL)

        # Save BDT to TMVA xml file
        # NB. variable order is important for TMVA
        try:
            from skTMVA import convert_bdt_sklearn_tmva
            logging.getLogger("skl_BDT").info("Exporting output to TMVA XML file")
            variables = [ (v,variable_dict[v]) for v in classification_variables ]
            convert_bdt_sklearn_tmva(
                classifier,
                variables,
                os.path.join(self.output_directory, sample_name, self.name, "classifier", "skl_BDT_TMVA.weights.xml")
                )
        except ImportError:
            logging.getLogger("skl_BDT").info("Could not import skTMVA. Skipping export to TMVA output.")


    def test(self, test_data, classification_variables, training_sample):
        """
        Definition:
        -----------
            Testing method for sklBDT; it loads the latest model from the "pickle" sub-folder

        Args:
        -----
            data = dictionary, containing "X", "y", "w" for the set to evaluate performance on, where:
                X = ndarray of dim (# examples, # features)
                y = array of dim (# examples) with target values
                w = array of dim (# examples) with event weights
            classification_variables = list of names of variables used for classification
            training_sample = string that specifies the file name of the sample to use as a training (e.g. "SM_merged" or "X350_hh")

        Returns:
        --------
            yhat = the array of BDT outputs corresponding to the P(signal), of dimensions (n_events)
        """
        logging.getLogger("skl_BDT").info("Evaluating performance...")

        # -- Load scikit classifier
        classifier = joblib.load(os.path.join(self.output_directory, training_sample, self.name, "classifier", "skl_BDT_clf.pkl"))

        # -- Get classifier predictions
        yhat = classifier.predict_proba(test_data["X"])[:, 1] # extracting column 1, i.e. P(signal)
        yhat_class = classifier.predict(test_data["X"])

        # -- Log classification scores
        logging.getLogger("skl_BDT").info("accuracy = {:.2f}%".format(100 * classifier.score(test_data["X"], test_data["y"], sample_weight=test_data["w"])))
        for output_line in classification_report(
                test_data["y"], 
                yhat_class, 
                target_names=["correct", "incorrect"], 
                sample_weight=test_data["w"]
                ).splitlines():
            logging.getLogger("skl_BDT").info(output_line)

        return yhat, yhat_class
