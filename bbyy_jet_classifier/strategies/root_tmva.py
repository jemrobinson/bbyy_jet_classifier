import array
import logging
import os
import shutil
from . import BaseStrategy
from ROOT import TCut, TFile, TMVA
from root_numpy.tmva import add_classification_events, evaluate_reader


class RootTMVA(BaseStrategy):
    """
    Strategy using a BDT from ROOT TMVA
    """
    default_output_location = os.path.join("output", "RootTMVA")

    def train(self, X_train, y_train, w_train, classification_variables, variable_dict):
        """
        Definition:
        -----------
            Training method for RootTMVA; it saves the model into the "weights" sub-folder
        Args:
        -----
            X_train = the features matrix with events for training, of dimensions (n_events, n_features)
            y_train = the target array with events for training, of dimensions (n_events)
            w_train = the array of weights for training events, of dimensions (n_events)
            classification_variables = list of names of variables used for classification
            variable_dict = ordered dict, mapping all the branches from the TTree to their type
        """
        f_output = TFile(os.path.join(self.output_directory, "TMVA_output.root"), "RECREATE")
        factory = TMVA.Factory("TMVAClassification", f_output, "AnalysisType=Classification")

        # -- Add variables to the factory:
        for v_name in classification_variables:
            factory.AddVariable(v_name, variable_dict[v_name])

        # Call root_numpy's utility functions to add events from the arrays
        add_classification_events(factory, X_train, y_train, weights=w_train)
        add_classification_events(factory, X_train[0:50], y_train[0:50], weights=w_train[0:50], test=True)  # need to add some testing events or TMVA will complain

        # The following line is necessary if events have been added individually:
        factory.PrepareTrainingAndTestTree(TCut("1"), "NormMode=EqualNumEvents")

        #-- Define methods:
        factory.BookMethod(TMVA.Types.kBDT, "BDT", ":".join(
            ["NTrees=800", "MinNodeSize=5", "MaxDepth=10", "BoostType=Grad", "SeparationType=GiniIndex"]
        ))

        # -- Where stuff actually happens:
        logging.getLogger("RootTMVA.train").info("Train all methods")
        factory.TrainAllMethods()

        # -- Organize output:
        logging.getLogger("RootTMVA.train").info("Organising output")
        if os.path.isdir(os.path.join(self.output_directory, "weights")):
            shutil.rmtree(os.path.join(self.output_directory, "weights"))
        shutil.move("weights", self.output_directory)

    def test(self, X, y, w, classification_variables, process):
        """
        Definition:
        -----------
            Testing method for RootTMVA; it loads the latest model from the "weights" sub-folder
        Args:
        -----
            X = the features matrix with events to test performance on, of dimensions (n_events, n_features)
            y = the target array with events to test performance on, of dimensions (n_events)
            w = the array of weights of the events to test performance on, of dimensions (n_events)
            process = string to identify whether we are evaluating performance on the train or test set, usually "training" or "testing"
            classification_variables = list of names of variables used for classification

        Returns:
        --------
            yhat = the array of BDT outputs, of dimensions (n_events)
        """
        logging.getLogger("RootTMVA.test").info("Evaluating performance...")

        # -- Construct reader and add variables to it:
        logging.getLogger("RootTMVA.test").info("Construct TMVA reader and add variables to it")
        reader = TMVA.Reader()
        for v_name in classification_variables:
            reader.AddVariable(v_name, array.array("f", [0]))

        # -- Load TMVA results
        reader.BookMVA("BDT", os.path.join(self.output_directory, "weights", "TMVAClassification_BDT.weights.xml"))

        yhat = evaluate_reader(reader, "BDT", X)
        return yhat
