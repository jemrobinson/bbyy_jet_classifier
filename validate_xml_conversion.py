#! /usr/bin/env python
import array
from bbyy_jet_classifier import utils
from collections import OrderedDict
import glob
import logging
import numpy as np
import ROOT
from sklearn.externals import joblib

utils.configure_logging()
logger = logging.getLogger("xml validation")


# variables = OrderedDict( (k,array.array("{}".format("i" if "idx" in k else "f"),[0])) for k in variable_names )
variable_names = [ "abs_eta_j", "abs_eta_jb", "Delta_eta_jb", "idx_by_mH", "idx_by_pT", "idx_by_pT_jb", "m_jb", "pT_j", "pT_jb" ]
variables = OrderedDict( (k,array.array("f",[0])) for k in variable_names )

# Load scikit-learn from pickle
skl_bdt = joblib.load("output/merged_inputs/skl_BDT/classifier/skl_BDT_clf.pkl")

# # Set up ROOT TMVAs
# reader_ROOT = ROOT.TMVA.Reader()
# for variable_name, variable in variables.items():
#     reader_ROOT.AddVariable(variable_name, variable)
# reader_ROOT.BookMVA("scikit-learn", "output/merged_inputs/root_tmva/weights/TMVAClassification_BDT.weights.xml")

from skTMVA import convert_bdt_sklearn_tmva
logging.getLogger("skl_BDT").info("Exporting output to TMVA XML file")
tree_variables = [ (v,"D") for v in variable_names ]
convert_bdt_sklearn_tmva(skl_bdt, tree_variables, "converted_skl_BDT_TMVA.weights.xml")

reader_skl = ROOT.TMVA.Reader()
for variable_name, variable in variables.items():
    reader_skl.AddVariable(variable_name, variable)
reader_skl.BookMVA("converted", "converted_skl_BDT_TMVA.weights.xml")

for input_filename in [ glob.glob("inputs/*X275*root")[0] ]:
    logger.info("Now considering {}".format(input_filename))
    input_file = ROOT.TFile.Open(input_filename, "READ")
    event_tree = input_file.Get("events_1tag")
    for idx_evt, event in enumerate(event_tree):
        if idx_evt > 5 : break
        n_pairs = len([ x for x in event.isCorrect ])
        scores = dict( (k,[]) for k in ["ROOT", "scikit-learn", "converted"] )
        for idx_pair in range(n_pairs):
            # # ROOT TMVA
            # for variable_name, variable in variables.items():
            #     variable[0] = getattr(event,variable_name)[idx_pair]
            # scores["ROOT"].append(reader_ROOT.EvaluateMVA("scikit-learn"))

            # scikit-learn
            skl_bdt_input = np.array([variable[0] for variable in variables.values()]).reshape(1,-1)
            scores["scikit-learn"].append(skl_bdt.decision_function(skl_bdt_input).item(0))

            # skl converted
            for variable_name, variable in variables.items():
                variable[0] = getattr(event,variable_name)[idx_pair]
            scores["converted"].append(reader_skl.EvaluateMVA("converted"))

        # Log the decisions
        logger.info("Considering event {}/{}".format(idx_evt,event_tree.GetEntries()))
        logger.info("  there are {} jet pairs".format(n_pairs))
        # logger.info("  ROOT scores: {}".format(scores["ROOT"]))
        # logger.info("  -> best is: {}".format(np.array(scores["ROOT"]).argmax()))
        logger.info("  scikit-learn scores: {}".format(scores["scikit-learn"]))
        logger.info("  -> best is: {}".format(np.array(scores["scikit-learn"]).argmax()))
        logger.info("  converted scikit-learn scores: {}".format(scores["converted"]))
        logger.info("  -> best is: {}".format(np.array(scores["converted"]).argmax()))


        # # sklearn score
        # score = bdt.decision_function([var1[0], var2[0]]).item(0)
        #
        # # calculate the value of the classifier with TMVA/TskMVA
        # bdtOutput = reader.EvaluateMVA("BDT")
