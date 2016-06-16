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

    # parser.add_argument("--correct_tree", metavar="NAME_OF_TREE", type=str,
    #                     help="name of tree containing correctly identified pairs", default="correct")

    # parser.add_argument("--incorrect_tree", metavar="NAME_OF_TREE", type=str,
    #                     help="name of tree containing incorrectly identified pairs", default="incorrect")

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
# TO-DO: we might want to move this block of functions elsewhere

def count_correct_total(yhat, y):
    '''
    Definition:
    -----------
        Quantify the number of events in which the correct jet pair was assigned the highest classifier score
    Args:
    -----
        yhat: event level numpy array containing the predictions for each jet in the event
        y:    event level numpy array containing the truth labels for each jet in the event
    Returns:
    --------
        n_correct_classifier: int, number of events in which the correct jet pair was assigned the highest classifier score
        n_correct_truth:      int, total number of events with a 'correct' jet pair
    '''
    n_correct_classifier = sum([np.argmax(yhat[ev]) == np.argmax(y[ev]) for ev in xrange(len(y)) if sum(y[ev]) == 1])
    n_correct_truth = sum([sum(y[ev]) == 1 for ev in xrange(len(y))])
    return n_correct_classifier, n_correct_truth


def sb(mjb, y, yhat):
    '''
    Definition:
    -----------
        Calculate the amount of correctly and incorrectly classified events that fall in the [95, 135] GeV m_jb window
    Args:
    -----
        mjb:  event level numpy array containing the values of m_jb for each jet in the event
        y:    event level numpy array containing the truth labels for each jet in the event
        yhat: event level numpy array containing the predictions for each jet in the event
    Returns:
    --------
        s: float, the number of jet pairs properly classified as 'correct' that fall in the m_jb window
        b: float, the number of jet pairs mistakenly classified as 'correct' that fall in the m_jb window
    '''
    mjb_correct = np.array([mjb[ev][np.argmax(yhat[ev])] \
                for ev in xrange(len(y)) if ((sum(y[ev]) == 1) and (np.argmax(yhat[ev]) == np.argmax(y[ev]) )) ])

    mjb_incorrect = np.array([mjb[ev][np.argmax(yhat[ev])] \
        for ev in xrange(len(y)) if ((sum(y[ev]) != 1) or (np.argmax(yhat[ev]) != np.argmax(y[ev]))) ])

    s = float(sum(np.logical_and((mjb_correct < 135), (mjb_correct > 95))))
    b = float(sum(np.logical_and((mjb_incorrect < 135), (mjb_incorrect > 95))))
    return s, b


def asimov(s, b):
    '''
    Definition:
    -----------
        Calculates signal to background sensitivity according to the Asimov formula
    Args:
    -----
        s: float, the number of jet pairs properly classified as 'correct' that fall in the m_jb window
        b: float, the number of jet pairs mistakenly classified as 'correct' that fall in the m_jb window
    Returns:
    --------
        The result of the Asimov formula given s and b
    '''
    import math
    return math.sqrt(2 * ((s+b) * math.log(1 + (s/b)) - s))


def print_performance(yhat_test_ev, yhat_mHmatch_test_ev, yhat_pThigh_test_ev, y_event, mjb_event):
    '''
    Definition:
    -----------
        Log event-level performance outputs as info
    Args:
    -----
        yhat_test_ev: event level numpy array containing the predictions from the BDT for each jet in the event
        yhat_mHmatch_test_ev: event level numpy array containing the predictions from mHmatch for each jet in the event
        yhat_pThigh_test_ev: event level numpy array containing the predictions from pThigh for each jet in the event
        y_event: event level numpy array containing the truth labels for each jet in the event
        mjb_event: event level numpy array containing the values of m_jb for each jet in the event
    '''
    logger = logging.getLogger("EventPerformance")
    logger.info('Number of correctly classified events for BDT = {} out of {} events having a correct pair'.format(
        *count_correct_total(yhat_test_ev, y_event)))
    logger.info('Number of correctly classified events for mHmatch = {} out of {} events having a correct pair'.format(
        *count_correct_total(yhat_mHmatch_test_ev, y_event)))
    logger.info('Number of correctly classified events for pThigh = {} out of {} events having a correct pair'.format(
        *count_correct_total(yhat_pThigh_test_ev, y_event)))
    logger.info('Number of events without any correct pair = {}'.format(sum([sum(y_event[ev]) == 0 for ev in xrange(len(y_event))])))

    # -- check how many of the selected jet pairs fall into the [95, 135] GeV m_jb window
    logger.info('S/B in m_bb window for BDT = {}'.format((lambda s,b : s/b)(*sb(mjb_event, y_event, yhat_test_ev))))
    logger.info('Asimov in m_bb window for BDT = {}'.format(asimov(*sb(mjb_event, y_event, yhat_test_ev))))
    logger.info('S/B in m_bb window for mHmatch = {}'.format((lambda s,b : s/b)(*sb(mjb_event, y_event, yhat_mHmatch_test_ev))))
    logger.info('Asimov in m_bb window for mHmatch = {}'.format(asimov(*sb(mjb_event, y_event, yhat_mHmatch_test_ev))))
    logger.info('S/B in m_bb window for pThigh = {}'.format((lambda s,b : s/b)(*sb(mjb_event, y_event, yhat_pThigh_test_ev))))
    logger.info('Asimov in m_bb window for pThigh = {}'.format(asimov(*sb(mjb_event, y_event, yhat_pThigh_test_ev))))
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
    classification_variables, variable2type, train_data, test_data, yhat_mHmatch_test, yhat_pThigh_test, y_event, mjb_event = process_data.load(
        args.input, args.exclude, args.ftrain)

    #-- Plot input distributions
    utils.ensure_directory(os.path.join(fileID, "classification_variables"))
    plot_inputs.input_distributions(classification_variables, train_data, test_data, directory=os.path.join(fileID, "classification_variables"))

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
            yhat_train = ML_strategy.test(train_data, classification_variables, process="training", train_location=fileID)
            plot_outputs.classifier_output(ML_strategy, yhat_train, train_data, process="training", fileID=fileID)

        else:
            logger.info("Preparing to use 100% of sample as testing input")

        # -- Testing!
        if args.ftrain < 1:
            # -- Test classifier
            yhat_test = ML_strategy.test(test_data, classification_variables, process="testing", train_location=train_location)

            # -- Plot output testing distributions from classifier and old strategies
            plot_outputs.classifier_output(ML_strategy, yhat_test, test_data, process="testing", fileID=fileID)
            plot_outputs.old_strategy(ML_strategy, yhat_mHmatch_test == 0, test_data, "mHmatch")
            plot_outputs.old_strategy(ML_strategy, yhat_pThigh_test == 0, test_data, "pThigh")

            # -- Visualize performance by displaying the ROC curve from the selected ML strategy and comparing it with the old strategies
            logger.info("Plotting ROC curves")
            plot_roc.signal_eff_bkg_rejection(ML_strategy, yhat_mHmatch_test == 0, yhat_pThigh_test == 0, yhat_test, test_data)

            # -- put it back into event format 
            yhat_test_ev = process_data.match_shape(yhat_test, y_event)
            yhat_mHmatch_test_ev = process_data.match_shape(yhat_mHmatch_test == 0, y_event)
            yhat_pThigh_test_ev = process_data.match_shape(yhat_pThigh_test == 0, y_event)

            # -- print performance 
            print_performance(yhat_test_ev, yhat_mHmatch_test_ev, yhat_pThigh_test_ev, y_event, mjb_event)

        else:
            logger.info("100% of the sample was used for training -- no independent testing can be performed.")

    # -- if there is more than one strategy and we aren't only training, plot the ROC comparison
    if (len(args.strategy) > 1 and (args.ftrain < 1)):
        logger.info("Plotting ROC comparison")
        plot_roc.roc_comparison(fileID)
