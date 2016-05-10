import os
import logging

import matplotlib.pyplot as plt
import numpy as np
# import rootpy.plotting as rpp

import plot_atlas

def input_distributions(strategy, classification_variables, data, process):
    """
    Definition:
    -----------
            Plots the distributions of input variables, color-coded by target class

    Args:
    -----
            strategy = a classification method -- here either sklBDT or RootTMVA
            data = dictionary, containing 'X', 'y', 'w' for a dataset, where:
                X = ndarray of dim (# examples, # features)
                y = array of dim (# examples) with target values
                w = array of dim (# examples) with event weights
            process = string, either "training" or "testing", usually
    """
    # rpp.set_style("ATLAS", mpl=True)
    # print 'get_style',rpp.get_style()
    logging.getLogger("Plotting").info("Plotting input distributions")
    plot_atlas.set_style()

    # -- Ensure output directory exists
    strategy.ensure_directory(os.path.join(strategy.output_directory, process))

    # -- Plot distributions of input variables
    for i, variable in enumerate(classification_variables):
        data_correct = data['X'][data['y'] == 1][:, i]
        data_incorrect = data['X'][data['y'] == 0][:, i]

        figure = plt.figure(figsize=(6, 6), dpi=100)
        axes = plt.axes()
        bins = np.linspace(min([min(data_correct), min(data_incorrect)]), max([max(data_correct), max(data_incorrect)]), 50)
        y_1, _, _ = plt.hist(data_correct, weights=data['w'][data['y'] == 1] / float(sum(data['w'][data['y'] == 1])), bins=bins, histtype="stepfilled", label="Correct", color="blue", alpha=0.5)
        y_2, _, _ = plt.hist(data_incorrect, weights=data['w'][data['y'] == 0] / float(sum(data['w'][data['y'] == 0])), bins=bins, histtype="stepfilled", label="Incorrect", color="red", alpha=0.5)
        plt.legend(loc="upper right")
        plt.xlabel(variable, position=(1., 0), va="bottom", ha="right")
        plt.ylabel("Fraction of events", position=(0., 1.), va="top", ha="right")
        axes.xaxis.set_label_coords(1., -0.15)
        axes.yaxis.set_label_coords(-0.2, 1.)
        axes.set_ylim([0, 1.3 * max([1e-5, max(y_1), max(y_2)])])
        plot_atlas.atlas_label(axes, fontsize=10)
        figure.savefig(os.path.join(strategy.output_directory, process, "{}.pdf".format(variable)))
        plt.close(figure)
