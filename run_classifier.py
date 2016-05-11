#! /usr/bin/env python
import argparse
import logging
import os
from bbyy_jet_classifier import strategies, process_data
from bbyy_jet_classifier.plotting import plot_inputs, plot_outputs, plot_roc

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
    parser.add_argument("--output", type=str, help="output directory", default="output")
    parser.add_argument("--correct_tree", metavar="NAME_OF_TREE", type=str, help="name of tree containing correctly identified pairs", default="correct")
    parser.add_argument("--incorrect_tree", metavar="NAME_OF_TREE", type=str, help="name of tree containing incorrectly identified pairs", default="incorrect")
    parser.add_argument("--exclude", type=str, metavar="VARIABLE_NAME", nargs="+", help="list of variables to exclude", default=[])
    parser.add_argument("--ftrain", type=float, help="fraction of events to use for training", default=0.7)
    parser.add_argument("--strategy", nargs='+', help="strategy to use. Options are: RootTMVA, sklBDT.", default="RootTMVA")
    args = parser.parse_args()

    # -- Check that input file exists
    if not os.path.isfile(args.input):
        raise OSError("{} does not exist!".format(args.input))

    # -- Load in root files and return literally everything about the data
    classification_variables, variable_dict, train_data, test_data, mHmatch_test, pThigh_test = \
        process_data.load(args.input, args.correct_tree, args.incorrect_tree, args.exclude, args.ftrain)

    #-- Plot input distributions
    strategies.BaseStrategy.ensure_directory(os.path.join(args.output, "classification_variables"))
    plot_inputs.input_distributions(classification_variables, train_data, test_data, directory=os.path.join(args.output, "classification_variables"))

    # -- Sequentially evaluate all the desired strategies on the same train/test sample
    for strategy_name in args.strategy:

        # -- Construct dictionary of available strategies
        if not strategy_name in strategies.__dict__.keys():
            raise AttributeError("{} is not a valid strategy".format(args.strategy))
        ML_strategy = getattr(strategies, strategy_name)(args.output)

        # -- Training!
        if args.ftrain > 0:
            logging.getLogger("RunClassifier").info("Preparing to train with {}% of events and then test with the remainder".format(int(100 * args.ftrain)))

            # -- Train classifier
            ML_strategy.train(train_data, classification_variables, variable_dict)

            # -- Plot the classifier output as tested on the training set (only useful if you care to check the performance on the training set)
            yhat_train = ML_strategy.test(train_data, classification_variables, process="training")
            plot_outputs.classifier_output(ML_strategy, yhat_train, train_data, process="training", fileID=args.input.replace(".root", "").split("/")[-1])

        else:
            logging.getLogger("RunClassifier").info("Preparing to use 100% of sample as testing input")

        # -- Testing!
        if args.ftrain < 1:
            # -- Test classifier
            yhat_test = ML_strategy.test(test_data, classification_variables, process="testing")

            # -- Plot output testing distributions from classifier and old strategies
            plot_outputs.classifier_output(ML_strategy, yhat_test, test_data, process="testing", fileID=args.input.replace(".root", "").split("/")[-1])
            plot_outputs.old_strategy(ML_strategy, mHmatch_test, test_data, "mHmatch")
            plot_outputs.old_strategy(ML_strategy, pThigh_test, test_data, "pThigh")

            # -- Visualize performance by displaying the ROC curve from the selected ML strategy and comparing it with the old strategies
            logging.getLogger("RunClassifier").info("Plotting ROC curves...")
            plot_roc.signal_eff_bkg_rejection(ML_strategy, mHmatch_test, pThigh_test, yhat_test, test_data)

        else:
            logging.getLogger("RunClassifier").info("100% of the sample was used for training -- no independent testing can be performed.")

    # -- if there is more than one strategy, plot the ROC comparison
    if len(args.strategy) > 1:
        plot_roc.roc_comparison()
