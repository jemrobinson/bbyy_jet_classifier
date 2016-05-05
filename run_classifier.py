#! /usr/bin/env python
import argparse
import logging
import os
from bbyy_jet_classifier import strategies, plotting, process_data

if __name__ == "__main__":
    # -- Configure logging
    logging.basicConfig(format="%(levelname)-8s\033[1m%(name)-21s\033[0m: %(message)s")
    logging.addLevelName(logging.WARNING, "\033[1;31m{:8}\033[1;0m".format(logging.getLevelName(logging.WARNING)))
    logging.addLevelName(logging.ERROR, "\033[1;35m{:8}\033[1;0m".format(logging.getLevelName(logging.ERROR)))
    logging.addLevelName(logging.INFO, "\033[1;32m{:8}\033[1;0m".format(logging.getLevelName(logging.INFO)))
    logging.addLevelName(logging.DEBUG, "\033[1;34m{:8}\033[1;0m".format(logging.getLevelName(logging.DEBUG)))

    # -- Parse arguments
    parser = argparse.ArgumentParser(description="Run ML algorithms over ROOT TTree input")
    parser.add_argument("--input", type=str, help="input file name", required=True)
    parser.add_argument("--output", type=str, help="output directory", default=None)
    parser.add_argument("--correct_tree", metavar="NAME_OF_TREE", type=str, help="name of tree containing correctly identified pairs", default="correct")
    parser.add_argument("--incorrect_tree", metavar="NAME_OF_TREE", type=str, help="name of tree containing incorrectly identified pairs", default="incorrect")
    parser.add_argument("--exclude", type=str, metavar="VARIABLE_NAME", nargs="+", help="list of variables to exclude", default=[])
    parser.add_argument("--ftrain", type=float, help="fraction of events to use for training", default=0.7)
    parser.add_argument("--strategy", type=str, help="strategy to use. Options are: RootTMVA, sklBDT.", default="RootTMVA")
    args = parser.parse_args()

    # -- Check that input file exists
    if not os.path.isfile(args.input):
        raise OSError("{} does not exist!".format(args.input))

    # -- Construct dictionary of available strategies
    if not args.strategy in strategies.__dict__.keys():
        raise AttributeError("{} is not a valid strategy".format(args.strategy))
    ML_strategy = getattr(strategies, args.strategy)(args.output)

    # -- Load in root files and return literally everything about the data
    classification_variables, variable_dict, X_train, X_test, y_train, y_test, w_train, w_test, mHmatch_test, pThigh_test = \
        process_data.load(args.input, args.correct_tree, args.incorrect_tree, args.exclude, args.ftrain)

    # -- Training!
    if args.ftrain > 0:
        logging.getLogger("RunClassifier").info("Preparing to train with {}% of events and then test with the remainder".format(int(100 * args.ftrain)))

        #-- Plot training distributions
        plotting.input_distributions(ML_strategy, classification_variables, X_train, y_train, w_train, process="training")  # plot the feature distributions

        # -- Train classifier
        ML_strategy.train(X_train, y_train, w_train, classification_variables, variable_dict)

        # -- Plot the classifier output as tested on the training set (only useful if you care to check the performance on the training set)
        yhat_train = ML_strategy.test(X_train, y_train, w_train, classification_variables, process="training")
        plotting.classifier_output(ML_strategy, yhat_train, y_train, w_train, process="training", fileID=args.input.replace(".root", "").split("/")[-1])

    else:
        logging.getLogger("RunClassifier").info("Preparing to use 100% of sample as testing input")

    # -- Testing!
    if args.ftrain < 1:
        # #-- Plot input testing distributions
        # plotting.input_distributions(ML_strategy, classification_variables, X_test, y_test, w_test, process="testing")
        # Is this useful? It's probably worth plotting these points on top of the training ones to check for bias

        # -- TEST
        yhat_test = ML_strategy.test(X_test, y_test, w_test, classification_variables, process="testing")

        # -- Plot output testing distributions from classifier and old strategies
        plotting.classifier_output(ML_strategy, yhat_test, y_test, w_test, process="testing", fileID=args.input.replace(".root", "").split("/")[-1])
        plotting.old_strategy(ML_strategy.output_directory, mHmatch_test, y_test, w_test, "mHmatch")
        plotting.old_strategy(ML_strategy.output_directory, pThigh_test, y_test, w_test, "pThigh")

        # -- Visualize performance by displaying the ROC curve from the selected ML strategy and comparing it with the old strategies
        logging.getLogger("RunClassifier").info("Plotting ROC curves...")
        plotting.signal_eff_bkg_rejection(ML_strategy, mHmatch_test, pThigh_test, yhat_test, y_test, w_test)

    else:
        logging.getLogger("RunClassifier").info("100% of the sample was used for training -- no independent testing can be performed.")
