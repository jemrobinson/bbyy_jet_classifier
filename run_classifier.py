#! /usr/bin/env python
import argparse
import cPickle
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

# --------------------------------------------------------------

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

    # -- Set up folder paths
    fileID = os.path.splitext(os.path.split(args.input)[-1])[0]
    train_location = args.train_location if args.train_location is not None else fileID

    # -- Load in root files and return literally everything about the data
    # y_event will be used to match event-level shape from flattened arrays
    # mjb_event will be used to check if the selected jet pair falls into the mjb mass window
    classification_variables, \
    variable2type, \
    train_data, \
    test_data, \
    yhat_mHmatch_test, \
    yhat_pThigh_test, \
    y_event, \
    mjb_event = process_data.load(args.input, args.exclude, args.ftrain)

    #-- Plot input distributions
    utils.ensure_directory(os.path.join(fileID, "classification_variables"))
    plot_inputs.input_distributions(
        classification_variables, 
        train_data, 
        test_data, 
        directory=os.path.join(fileID, "classification_variables")
        )

    # -- Sequentially evaluate all the desired strategies on the same train/test sample
    for strategy_name in args.strategy:

        # -- Construct dictionary of available strategies
        if strategy_name not in strategies.__dict__.keys():
            raise AttributeError("{} is not a valid strategy".format(strategy_name))
        ML_strategy = getattr(strategies, strategy_name)(fileID)

        # -- Training!
        if args.ftrain > 0:
            logger.info("Preparing to train with {}% of events and then test with the remainder".format(int(100 * args.ftrain)))

            # -- Train classifier
            ML_strategy.train(train_data, classification_variables, variable2type)

            # -- Plot the classifier output as tested on the training set 
            # -- (only useful if you care to check the performance on the training set)
            yhat_train = ML_strategy.test(
                train_data, 
                classification_variables, 
                process="training", 
                train_location=fileID
                )
            plot_outputs.classifier_output(
                ML_strategy, 
                yhat_train, 
                train_data, 
                process="training", 
                fileID=fileID
                )

        else:
            logger.info("Preparing to use 100% of sample as testing input")

        # -- Testing!
        if args.ftrain < 1:
            # -- Test classifier
            yhat_test = ML_strategy.test(
                test_data, 
                classification_variables, 
                process="testing", 
                train_location=train_location
                )

            # -- Plot output testing distributions from classifier and old strategies
            plot_outputs.classifier_output(
                ML_strategy, 
                yhat_test, 
                test_data, 
                process="testing", 
                fileID=fileID
                )
            plot_outputs.old_strategy(
                ML_strategy, 
                yhat_mHmatch_test == 0, 
                test_data, 
                "mHmatch"
                )
            plot_outputs.old_strategy(
                ML_strategy, 
                yhat_pThigh_test == 0, 
                test_data, 
                "pThigh"
                )
            plot_outputs.confusion(
                ML_strategy, 
                yhat_mHmatch_test == 0, 
                test_data, 
                "mHmatch"
                )
            plot_outputs.confusion(
                ML_strategy, 
                yhat_pThigh_test == 0, 
                test_data, 
                "pThigh"
                )

            # -- Performance evaluation:
            # -- 1) Jet-pair level

            # -- Visualize performance by displaying the ROC curve from the selected ML strategy and comparing it with the old strategies
            logger.info("Plotting ROC curves")
            plot_roc.signal_eff_bkg_rejection(
                ML_strategy, 
                yhat_mHmatch_test == 0, 
                yhat_pThigh_test == 0, 
                yhat_test, 
                test_data
                )

            # -- 2) Event level

            # -- put arrays back into event format by matching shape of y_event
            w_test = np.array([np.unique(w)[0] for w in process_data.match_shape(test_data['w'], y_event)]) 
            # ^ could also extract it directly from process_data
            yhat_test_ev = process_data.match_shape(
                yhat_test, 
                y_event
                )
            yhat_mHmatch_test_ev = process_data.match_shape(
                yhat_mHmatch_test == 0, 
                y_event
                )
            yhat_pThigh_test_ev = process_data.match_shape(
                yhat_pThigh_test == 0, 
                y_event
                )

            # -- print performance 
            logger.info("Preparing Event-Level Performance")
            cPickle.dump({
                'yhat_test_ev' : yhat_test_ev,
                'yhat_mHmatch_test_ev' :  yhat_mHmatch_test_ev,
                'yhat_pThigh_test_ev' : yhat_pThigh_test_ev,
                'y_event' : y_event,
                'mjb_event' : mjb_event,
                'w_test' : w_test
                },
                open(os.path.join(fileID, 'for_event_performance.pkl'), 'wb')
                )

        else:
            logger.info("100% of the sample was used for training -- no independent testing can be performed.")

    # -- if there is more than one strategy and we aren't only training, plot the ROC comparison
    if (len(args.strategy) > 1 and (args.ftrain < 1)):
        logger.info("Plotting ROC comparison")
        plot_roc.roc_comparison(fileID)
