import logging
import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import plot_atlas
from sklearn.metrics import confusion_matrix
from ..utils import ensure_directory


def old_strategy(ML_strategy, yhat_test, test_data, old_strategy_name, sample_name):
    """
    Definition:
    -----------
            Plot the output distributions of the two old 2nd jet selection strategies used in yybb

    Args:
    -----
            outdir = string, the name of the output directory to save the plots
            yhat_test = array of dim (# testing examples), containing the binary decision based on the specific strategy
            test_data = dictionary, containing 'y', 'w' for the test set, where:
                y = array of dim (# testing examples) with target values
                w = array of dim (# testing examples) with event weights
            old_strategy_name = string, name of the strategy to use, either "mHmatch" or "pThigh"
    """
    # -- Ensure output directory exists
    ensure_directory(os.path.join(ML_strategy.output_directory, "testing"))

    # -- Initialise figure and axes
    logging.getLogger("plot_outputs").info("Plotting old strategy: {} for testing sample".format(old_strategy_name))
    plot_atlas.set_style()
    figure = plt.figure(figsize=(6, 6), dpi=100)
    axes = plt.axes()

    # -- Plot data
    try:
        plt.hist(yhat_test[test_data['y'] == 1], weights=test_data['w'][test_data['y'] == 1] / float(sum(test_data['w'][test_data['y'] == 1])),
                 bins=np.linspace(0, 1, 10), histtype="stepfilled", label="Correct", color="blue", alpha=0.5)
    except ValueError:  # for Sherpa y+jets
        pass
    plt.hist(yhat_test[test_data['y'] == 0], weights=test_data['w'][test_data['y'] == 0] / float(sum(test_data['w'][test_data['y'] == 0])),
             bins=np.linspace(0, 1, 10), histtype="stepfilled", label="Incorrect", color="red", alpha=0.5)

    # -- Plot legend/axes/etc.
    plt.legend()
    plt.xlabel("{} output".format(old_strategy_name))
    plt.ylabel("Fraction of events")

    # -- Write figure and close plot to save memory
    plot_atlas.use_atlas_labels(axes)
    figure.savefig(os.path.join(ML_strategy.output_directory, "testing", "{}_{}_classifier.pdf".format(sample_name, old_strategy_name)))


def confusion(ML_strategy, yhat, data, model_name, sample_name):
    """
        Definition:
        -----------
                Plots the 2D confusion matrix

        Args:
        -----
                ML_strategy = one of the machine learning strategy in strategies/ whose prerformance we want to visualize
                yhat = the array of predictions
                data = dictionary, containing 'y', 'w' for the set to evaluate performance on, where:
                    y = array of dim (# examples) with target values
                    w = array of dim (# examples) with event weights
                model_name = string, either "BDT", "mHmatch" or "pTmatch"
                sample_name = arbitrary string that refers back to the input file, usually
    """
    y_test = data['y']
    plt.clf()
    figure = plt.figure(figsize=(11.69, 10.27), dpi=100)
    matplotlib.rcParams.update({'font.size': 10})

    def _plot_confusion_matrix(cm, title='Confusion Matrix', cmap=plt.cm.RdPu):
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(np.unique(y_test)))
        plt.xticks(tick_marks, ['Incorrect', 'Correct'])
        plt.yticks(tick_marks, ['Incorrect', 'Correct'], rotation='vertical')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()

    # -- need to round yhat to nearest integer (0 or 1) to match y_test format
    #    NB. this is equivalent to cutting at the halfway point of the distribution
    cm = confusion_matrix(y_test, np.rint(yhat))
    # Normalize the confusion matrix by row (i.e by the number of samples in each class)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    _plot_confusion_matrix(cm_normalized, title='Normalized Confusion Matrix')
    plt.savefig(os.path.join(ML_strategy.output_directory, "testing", "{}_{}_confusion.pdf".format(sample_name, model_name)))
    plt.close(figure)


def classifier_output(ML_strategy, yhat, data, process, sample_name):
    """
    Definition:
    -----------
            Plots the classifier output, color-coded by target class

    Args:
    -----
            ML_strategy = one of the machine learning strategy in strategies/ whose prerformance we want to visualize
            yhat = the array of predictions
            data = dictionary, containing 'y', 'w' for the set to evaluate performance on, where:
                y = array of dim (# examples) with target values
                w = array of dim (# examples) with event weights
            process = string, either "training" or "testing", usually
            sample_name = arbitrary string that refers back to the input file, usually
    """
    # -- Ensure output directory exists
    ensure_directory(os.path.join(ML_strategy.output_directory, process))

    # -- Initialise figure, axes and binning
    logging.getLogger("plot_outputs").info("Plotting classifier output on {} set for {}".format(process, sample_name))
    plot_atlas.set_style()
    figure = plt.figure(figsize=(6, 6), dpi=100)
    axes = plt.axes()
    bins = np.linspace(min(yhat), max(yhat), 50)

    # -- Plot data
    try:
        plt.hist(yhat[data['y'] == 1], weights=data['w'][data['y'] == 1] / float(sum(data['w'][data['y'] == 1])), bins=bins, histtype="stepfilled", label="Correct", color="blue", alpha=0.5)
    except ValueError:  # for Sherpa y+jets
        pass
    plt.hist(yhat[data['y'] == 0], weights=data['w'][data['y'] == 0] / float(sum(data['w'][data['y'] == 0])), bins=bins, histtype="stepfilled", label="Incorrect", color="red", alpha=0.5)

    # -- Plot legend/axes/etc.
    plt.legend(loc="upper right")
    plt.xlabel("Classifier output")
    plt.ylabel("Fraction of events")
    plot_atlas.use_atlas_labels(axes)

    # -- Write figure and close plot to save memory
    figure.savefig(os.path.join(ML_strategy.output_directory, process, "{}_BDT_classifier.pdf".format(sample_name)))
    plt.close(figure)
