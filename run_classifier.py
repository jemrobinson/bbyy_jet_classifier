#! /usr/bin/env python
import argparse
import cPickle
import logging
import os
from bbyy_jet_classifier import strategies, process_data, utils
from bbyy_jet_classifier.plotting import plot_inputs, plot_outputs, plot_roc
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description="Run ML algorithms over ROOT TTree input")

    parser.add_argument("--input", required=True, type=str, nargs="+", help="input file names")
    parser.add_argument("--exclude", type=str, nargs="+", default=[], metavar="VARIABLE_NAME", help="list of variables to exclude")
    parser.add_argument("--ftrain", type=float, default=0.6, help="fraction of events to use for training")
    parser.add_argument("--training_sample", type=str, help="directory with training info")
    parser.add_argument("--strategy", type=str, nargs="+", default=["RootTMVA", "sklBDT"], help="strategy to use. Options are: RootTMVA, sklBDT.")

    args = parser.parse_args()
    return args


def check_args(args):
    """
    Check the logic of the input arguments
    """
    if (args.ftrain < 0) or (args.ftrain > 1):
        raise ValueError("ftrain can only be a float between 0.0 and 1.0")

    if (args.ftrain == 0) and (args.training_sample is None):
        raise ValueError("When testing on 100% of the input file you need to specify which classifier to load. Pass the folder containing the classifier to --training_sample.")

    if (args.ftrain > 0) and (args.training_sample is not None):
        raise ValueError("Training location is only a valid argument when ftrain == 0, because if you are using {}% of your input data for training, you should not be testing on a separate pre-trained classifier.".format(100 * args.ftrain))

# --------------------------------------------------------------

if __name__ == "__main__":

    # -- Configure logging
    utils.configure_logging()
    logger = logging.getLogger("RunClassifier")

    # -- Parse arguments
    args = parse_args()
    check_args(args)

    # -- Construct list of input samples and dictionaries of sample -> data
    input_samples, training_data, test_data, yhat_test_data = [], {}, {}, {}

    # -- Check that input file exists
    logger.info("Preparing to run over {} input samples".format(len(args.input)))
    for input_file_name in args.input:
        logger.info("Now considering input: {}".format(input_file_name))
        if not os.path.isfile(input_file_name):
            raise OSError("{} does not exist!".format(args.input))

        # -- Set up folder paths
        input_samples.append(os.path.splitext(os.path.split(input_file_name)[-1])[0])

        # -- Load in root files and return literally everything about the data
        # y_event will be used to match event-level shape from flattened arrays
        # mjb_event will be used to check if the selected jet pair falls into the mjb mass window
        # pTj_event will be used to try cutting on the jet pTs
        classification_variables, variable2type, \
            training_data[input_samples[-1]], \
            testing_data[input_samples[-1]], \
            yhat_test_data[input_samples[-1]], \
            y_event, \
            mjb_event, \
            pTj_event = process_data.load(input_file_name, args.exclude, args.ftrain)

        #-- Plot input distributions
        plot_inputs.input_distributions(classification_variables, training_data[input_samples[-1]], testing_data[input_samples[-1]],
                                        plot_directory=os.path.join("output", "classification_variables", input_samples[-1]))

    # -- Combine all training data into a new sample
    train_data_combined = process_data.combine_datasets(training_data.values())
    combined_input_sample = input_samples[0] if len(input_samples) == 1 else "Merged_inputs"
    training_sample = args.training_sample if args.training_sample is not None else combined_input_sample

    # -- Sequentially evaluate all the desired strategies on the same train/test sample
    for strategy_name in args.strategy:

        # -- Construct dictionary of available strategies
        if strategy_name not in strategies.__dict__.keys():
            raise AttributeError("{} is not a valid strategy".format(strategy_name))
        ML_strategy = getattr(strategies, strategy_name)(combined_input_sample)

        # -- Training!
        if args.ftrain > 0:
            logger.info("Preparing to train with {}% of events and then test with the remainder".format(int(100 * args.ftrain)))

            # -- Train classifier
            ML_strategy.train(train_data_combined, classification_variables, variable2type)

            # -- Plot the classifier output as tested on each of the training sets
            # -- (only useful if you care to check the performance on the training set)
            for (sample_name, train_data) in training_data.items():
                logger.info("Sanity check: plotting classifier output tested on training set {}".format(sample_name))
                yhat_train = ML_strategy.test(train_data, classification_variables, process="training", sample_name=training_sample)
                plot_outputs.classifier_output(ML_strategy, yhat_train, train_data, process="training", sample_name=sample_name)

        else:
            logger.info("Preparing to use 100% of sample as testing input")

        # -- Testing!
        if args.ftrain < 1:
            for (sample_name, test_data) in testing_data.items():
                logger.info("Running classifier on testing set for {}".format(sample_name))

                # -- Test classifier
                yhat_test = ML_strategy.test(test_data, classification_variables, process="testing", sample_name=training_sample)

                # -- Plot output testing distributions from classifier and old strategies
                plot_outputs.classifier_output(ML_strategy, yhat_test, test_data, process="testing", sample_name=sample_name)
                for old_strategy_name in ["mHmatch", "pThigh"]:
                    plot_outputs.old_strategy(ML_strategy, yhat_test_data[old_strategy_name] == 0, test_data, old_strategy_name, sample_name=sample_name)
                    plot_outputs.confusion(ML_strategy, yhat_test_data[old_strategy_name] == 0, test_data, old_strategy_name, sample_name=sample_name)

                # -- Performance evaluation:
                # -- 1) Jet-pair level

                # -- Visualize performance by displaying the ROC curve from the selected ML strategy and comparing it with the old strategies
                logger.info("Plotting ROC curves")
                plot_roc.signal_eff_bkg_rejection(ML_strategy, yhat_test_data["mHmatch"] == 0, yhat_test_data["pThigh"] == 0, yhat_test, test_data)

                # -- 2) Event level

                # -- put arrays back into event format by matching shape of y_event
                w_test = np.array([np.unique(w)[0] for w in process_data.match_shape(test_data["w"], y_event)])
                # ^ could also extract it directly from process_data
                yhat_test_ev = process_data.match_shape(yhat_test, y_event)
                yhat_mHmatch_test_ev = process_data.match_shape(yhat_test_data["mHmatch"] == 0, y_event)
                yhat_pThigh_test_ev = process_data.match_shape(yhat_test_data["pThigh"] == 0, y_event)

                # -- print performance
                logger.info("Preparing event-level performance information")
                cPickle.dump({"yhat_test_ev": yhat_test_ev,
                              "yhat_mHmatch_test_ev": yhat_mHmatch_test_ev,
                              "yhat_pThigh_test_ev": yhat_pThigh_test_ev,
                              "y_event": y_event,
                              "mjb_event": mjb_event,
                              "pTj_event": pTj_event,
                              "w_test": w_test},
                             open(os.path.join(ML_strategy.output_directory, "event_performance_dump.pkl"), "wb"))

        else:
            logger.info("100% of the sample was used for training -- no independent testing can be performed.")

    # -- if there is more than one strategy and we aren't only training, plot the ROC comparison
    if len(args.strategy) > 1 and (args.ftrain < 1):
        logger.info("Plotting ROC comparison")
        plot_roc.roc_comparison(input_sample)
