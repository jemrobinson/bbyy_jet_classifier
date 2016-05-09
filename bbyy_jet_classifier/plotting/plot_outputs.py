import os
import logging

import matplotlib.pyplot as plt
import numpy as np
import rootpy.plotting as rpp

import plot_atlas

def old_strategy(outdir, yhat_test, y_test, w_test, old_strategy_name):
    """
    Definition:
    -----------
            Plot the output distributions of the two old 2nd jet selection strategies used in yybb

    Args:
    -----
            outdir = string, the name of the output directory to save the plots
            yhat_test = array of dim (# testing examples), containing the binary decision based on the specific strategy
            y_test = array of dim (# testing examples) with target values
            w_test = array of dim (# testing examples) with event weights
            old_strategy_name = string, name of the strategy to use, either "mHmatch" or "pThigh"
    """
    logging.getLogger("Plotting").info("Plotting old strategy")
    figure = plt.figure(figsize=(6, 6), dpi=100)
    #ax = figure.add_subplot(111)
    plt.hist(yhat_test[y_test == 1], weights=w_test[y_test == 1] / float(sum(w_test[y_test == 1])), bins=np.linspace(0, 1, 10), histtype="stepfilled", label="Correct", color="blue", alpha=0.5)
    plt.hist(yhat_test[y_test == 0], weights=w_test[y_test == 0] / float(sum(w_test[y_test == 0])), bins=np.linspace(0, 1, 10), histtype="stepfilled", label="Incorrect", color="red", alpha=0.5)
    plt.legend()
    plt.xlabel("{} output".format(old_strategy_name))
    plt.ylabel("Fraction of events")
    axes = plt.axes()
    plot_atlas.atlaslabel(axes, fontsize=10)
    figure.savefig(os.path.join(outdir, "testing", "{}.pdf".format(old_strategy_name)))


def classifier_output(ML_strategy, yhat, y, w, process, fileID):
    """
    Definition:
    -----------
            Plots the classifier output, color-coded by target class

    Args:
    -----
            ML_strategy = one of the machine learning strategy in strategies/ whose prerformance we want to visualize
            yhat = the array of predictions
            y = the target array
            w = the array of weights
            process = string, either "training" or "testing", usually
            fileID = arbitrary string that refers back to the input file, usually
    """
    rpp.set_style("ATLAS", mpl=True)
    logging.getLogger("Plotting").info("Plotting classifier output")

    # -- Ensure output directory exists
    ML_strategy.ensure_directory("{}/{}/".format(ML_strategy.output_directory, process))

    figure = plt.figure(figsize=(6, 6), dpi=100)
    axes = plt.axes()
    bins = np.linspace(min(yhat), max(yhat), 50)

    plt.hist(yhat[y == 1], weights=w[y == 1] / float(sum(w[y == 1])), bins=bins, histtype="stepfilled", label="Correct", color="blue", alpha=0.5)
    plt.hist(yhat[y == 0], weights=w[y == 0] / float(sum(w[y == 0])), bins=bins, histtype="stepfilled", label="Incorrect", color="red", alpha=0.5)

    plt.legend(loc="upper right")
    plt.xlabel("Classifier Output", position=(1., 0), va="bottom", ha="right")
    plt.ylabel("Fraction of Events", position=(0, 1.), va="top", ha="right")
    axes.xaxis.set_label_coords(1., -0.15)
    axes.yaxis.set_label_coords(-0.18, 1.)
    plot_atlas.atlaslabel(axes, fontsize=10)
    figure.savefig(os.path.join(ML_strategy.output_directory, process, "BDT_{}.pdf".format(fileID)))
    plt.close(figure)
