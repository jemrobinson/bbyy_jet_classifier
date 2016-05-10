import os
import logging
import matplotlib.pyplot as plt
import numpy as np
import plot_atlas
import rootpy.plotting as rpp

def old_strategy(outdir, yhat_test, test_data, old_strategy_name):
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
    # -- Initialise figure and axes
    # rpp.set_style("ATLAS", mpl=True)
    # print 'get_style',rpp.get_style()
    logging.getLogger("PlotOutputs").info("Plotting old strategy: {}".format(old_strategy_name) )
    plot_atlas.set_style()
    figure = plt.figure(figsize=(6, 6), dpi=100)
    axes = plt.axes()

    # -- Plot data
    plt.hist(yhat_test[test_data['y'] == 1], weights=test_data['w'][test_data['y'] == 1] / float(sum(test_data['w'][test_data['y'] == 1])),
        bins=np.linspace(0, 1, 10), histtype="stepfilled", label="Correct", color="blue", alpha=0.5)
    plt.hist(yhat_test[test_data['y'] == 0], weights=test_data['w'][test_data['y'] == 0] / float(sum(test_data['w'][test_data['y'] == 0])),
        bins=np.linspace(0, 1, 10), histtype="stepfilled", label="Incorrect", color="red", alpha=0.5)

    # -- Plot legend/axes/etc.
    plt.legend()
    plt.xlabel("{} output".format(old_strategy_name))
    plt.ylabel("Fraction of events")

    # -- Write figure and close plot to save memory
    plot_atlas.atlas_label(axes, fontsize=10)
    figure.savefig(os.path.join(outdir, "testing", "{}.pdf".format(old_strategy_name)))


def classifier_output(ML_strategy, yhat, data, process, fileID):
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
            fileID = arbitrary string that refers back to the input file, usually
    """
    # rpp.set_style("ATLAS", mpl=True)
    # print 'get_style',rpp.get_style()

    # -- Ensure output directory exists
    ML_strategy.ensure_directory("{}/{}/".format(ML_strategy.output_directory, process))

    # -- Initialise figure, axes and binning
    logging.getLogger("PlotOutputs").info("Plotting classifier output")
    plot_atlas.set_style()
    figure = plt.figure(figsize=(6, 6), dpi=100)
    axes = plt.axes()
    bins = np.linspace(min(yhat), max(yhat), 50)

    # -- Plot data
    plt.hist(yhat[data['y'] == 1], weights=data['w'][data['y'] == 1] / float(sum(data['w'][data['y'] == 1])), bins=bins, histtype="stepfilled", label="Correct", color="blue", alpha=0.5)
    plt.hist(yhat[data['y'] == 0], weights=data['w'][data['y'] == 0] / float(sum(data['w'][data['y'] == 0])), bins=bins, histtype="stepfilled", label="Incorrect", color="red", alpha=0.5)

    # -- Plot legend/axes/etc.
    plt.legend(loc="upper right")
    plt.xlabel("Classifier Output", position=(1., 0), va="bottom", ha="right")
    plt.ylabel("Fraction of Events", position=(0, 1.), va="top", ha="right")
    axes.xaxis.set_label_coords(1., -0.15)
    axes.yaxis.set_label_coords(-0.18, 1.)
    plot_atlas.atlas_label(axes, fontsize=10)

    # -- Write figure and close plot to save memory
    figure.savefig(os.path.join(ML_strategy.output_directory, process, "BDT_{}.pdf".format(fileID)))
    plt.close(figure)
