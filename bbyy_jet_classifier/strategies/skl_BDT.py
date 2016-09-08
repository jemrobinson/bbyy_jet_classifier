import cPickle
import logging
import os
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.externals import joblib
from sklearn.metrics import classification_report
from sklearn.grid_search import GridSearchCV
from . import BaseStrategy
from ..utils import ensure_directory

class sklBDT(BaseStrategy):
    """
    Strategy using a BDT from scikit-learn
    """

    def train(self, train_data, classification_variables, variable_dict, sample_name, grid_search):
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

        if grid_search:
            # Thoughts:
            # -- min_samples_leaf is supposedly faster to train than min_samples_split
            # -- could first tune optimum number of trees
            #    then tune max_depth and min_samples to save on combinatorics
            parameters = {"n_estimators":[100, 150, 200, 250, 300],
                          "max_depth":[2, 4, 6, 8, 10],
                          "min_samples_leaf":[20, 30, 40, 50, 60]}
            fit_params = {"sample_weight":train_data["w"]}
            # Run grid search over provided ranges
            logging.getLogger("skl_BDT").info("Running grid search parameter optimisation...")
            grid_search = GridSearchCV(estimator=GradientBoostingClassifier(learning_rate=0.2, min_samples_leaf=50, max_features="sqrt", subsample=0.8, random_state=10),
                                       param_grid=parameters, fit_params=fit_params, scoring="roc_auc", n_jobs=1, iid=False, cv=3, verbose=1)
            grid_search.fit(train_data["X"], train_data["y"])
            for param_name in parameters.keys():
                if grid_search.best_params_[param_name] in [ parameters[param_name][0], parameters[param_name][-1] ]:
                    logging.getLogger("skl_BDT").info("Best value of {} is at limit of considered range!".format(param_name))
                parameters[param_name] = grid_search.best_params_[param_name]
            for param_name in parameters.keys():
                logging.getLogger("skl_BDT").info("... {}: {}".format(param_name, parameters[param_name]))

        else:
            classifier = GradientBoostingClassifier(
                n_estimators=200, # was n_estimators=300
                max_depth=6, # was max_depth=15
                min_samples_leaf=40, # was min_samples_split=0.5 * len(train_data["y"])
                verbose=1
                )
            classifier.fit(train_data["X"], train_data["y"], sample_weight=train_data["w"])

        # -- Dump output to pickle
        ensure_directory(os.path.join(self.output_directory, sample_name, self.name, "classifier"))
        joblib.dump(classifier, os.path.join(self.output_directory, sample_name, self.name, "classifier", "skl_BDT_clf.pkl"), protocol=cPickle.HIGHEST_PROTOCOL)

        # Save BDT to TMVA xml file
        # -- variable order is important for TMVA
        # -- can't yet reproduce scikit-learn output in TMVA(!)
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
        # yhat = classifier.predict_proba(test_data["X"])[:, 1] # extracting column 1, i.e. P(signal)
        yhat = classifier.decision_function(test_data["X"]) # get the actual decision function
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
