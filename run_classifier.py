#! /usr/bin/env python
import argparse
import logging
import os
from bbyy_jet_classifier import strategies, process_data, utils, eventify
from bbyy_jet_classifier.plotting import plot_inputs, plot_outputs, plot_roc
import numpy as np
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run ML algorithms over ROOT TTree input")

    parser.add_argument("--input", type=str,
                        help="input file name", required=True)

    parser.add_argument("--exclude", type=str, metavar="VARIABLE_NAME", nargs="+", 
                        help="list of variables to exclude", default=[])

    parser.add_argument("--ftrain", type=float,
                        help="fraction of events to use for training", default=0.7)

    parser.add_argument("--train_location", type=str,
                        help="directory with training info")

    parser.add_argument("--strategy", nargs='+',
                        help="strategy to use. Options are: RootTMVA, sklBDT.", default=["RootTMVA", "sklBDT"])

    args = parser.parse_args()
    return args


def check_args(args):
    '''
    Check the logic of the input arguments
    '''
    if ((args.ftrain < 0) or (args.ftrain > 1)):
        raise ValueError("ftrain can only be a float between 0.0 and 1.0")

    if ((args.ftrain == 0) and (args.train_location == None)):
        raise ValueError(
            "Training folder required when testing on 100% of the input file to specify which classifier to load. \
             Pass --train_location.")

    if ((args.ftrain > 0) and (args.train_location != None)):
        raise ValueError("Training location is only a valid argument when ftrain == 0, \
                          because if you are using {}% of your input data for training, \
                          you should not be testing on a separate pre-trained classifier.".format(100 * args.ftrain))


if __name__ == "__main__":

    # -- Configure logging
    utils.configure_logging()
    logger = logging.getLogger("RunClassifier")

    # -- Parse arguments
    args = parse_args()
    check_args(args)

    # -- Check that input file exists
    if not os.path.isfile(args.input):
        raise OSError("{} does not exist!".format(args.input))
    sample_name = args.input.replace(".root", "").split("/")[-1]

    # -- Set up folder paths
    sample_name = os.path.splitext(os.path.split(args.input)[-1])[0]
    train_location = args.train_location if args.train_location is not None else sample_name

    # -- Load in root files and return literally everything about the data
    classification_variables, variable2type, train_data, test_data, yhat_mHmatch_test, yhat_pThigh_test, shape = process_data.load(
        args.input, args.exclude, args.ftrain)

    #-- Plot input distributions
    utils.ensure_directory(os.path.join('output', 'classification_variables'))
    plot_inputs.input_distributions(classification_variables, train_data, test_data, 
        output_directory=os.path.join('output', 'classification_variables'), sample_name=sample_name)

    # -- Sequentially evaluate all the desired strategies on the same train/test sample
    for strategy_name in args.strategy:

        # -- Construct dictionary of available strategies
        if strategy_name not in strategies.__dict__.keys():
            raise AttributeError("{} is not a valid strategy".format(strategy_name))
        ML_strategy = getattr(strategies, strategy_name)(sample_name)

        # -- Training!
        if args.ftrain > 0:
            logger.info("Preparing to train with {}% of events and then test with the remainder".format(int(100 * args.ftrain)))

            # -- Train classifier
            ML_strategy.train(train_data, classification_variables, variable2type)

            # -- Plot the classifier output as tested on the training set 
            # -- (only useful if you care to check the performance on the training set)
            yhat_train = ML_strategy.test(train_data, classification_variables, process="training", train_location=sample_name)
            plot_outputs.classifier_output(ML_strategy, yhat_train, train_data, process="training", sample_name=sample_name)

        else:
            logger.info("Preparing to use 100% of sample as testing input")

        # -- Testing!
        if args.ftrain < 1:
            # -- Test classifier
            yhat_test = ML_strategy.test(test_data, classification_variables, process="testing", train_location=train_location)

            # -- Plot output testing distributions from classifier and old strategies
            plot_outputs.classifier_output(ML_strategy, yhat_test, test_data, process="testing", sample_name=sample_name)
            # plot_outputs.old_strategy(ML_strategy, mHmatch_test, test_data, old_strategy_name="mHmatch", sample_name=sample_name)
            # plot_outputs.old_strategy(ML_strategy, pThigh_test, test_data, old_strategy_name="pThigh", sample_name=sample_name)
            plot_outputs.old_strategy(ML_strategy, yhat_mHmatch_test, test_data, "mHmatch", sample_name=sample_name)
            plot_outputs.old_strategy(ML_strategy, yhat_pThigh_test, test_data, "pThigh", sample_name=sample_name)

            # -- Visualize performance by displaying the ROC curve from the selected ML strategy and comparing it with the old strategies
            logger.info("Plotting ROC curves")
            plot_roc.signal_eff_bkg_rejection(ML_strategy, yhat_mHmatch_test, yhat_pThigh_test, yhat_test, test_data, sample_name=sample_name)

        else:
            logger.info("100% of the sample was used for training -- no independent testing can be performed.")

    # -- if there is more than one strategy, plot the ROC comparison
    if len(args.strategy) > 1:
        plot_roc.roc_comparison(output_directory="output", sample_name=sample_name)
