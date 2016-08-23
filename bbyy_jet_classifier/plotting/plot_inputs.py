import logging
import os
import matplotlib.pyplot as plt
import numpy as np
import plot_atlas
from .. import utils

ROOT_2_LATEX = {
    "abs_eta_j": r"$|\eta_{j}|$",
    "abs_eta_jb": r"$|\eta_{jb}|$",
    "Delta_eta_jb": r"$\Delta\eta_{jb}$",
    "Delta_phi_jb": r"$\Delta\phi_{jb}$",
    "idx_by_mH": r"$m_{H}$ matching order",
    "idx_by_pT": r"$p_{T}$ order",
    "m_jb": r"$m_{jb}$",
    "pT_j": r"$p_{T}^{j}$",
    "pT_jb": r"$p_{T}^{jb}$",
}


def input_distributions(classification_variables, training_data, test_data, plot_directory):
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
    # -- Setup ATLAS style
    logging.getLogger("plot_inputs").info("Plotting input distributions")
    plot_atlas.set_style()

    # -- Plot distributions of input variables
    for i, variable in enumerate(classification_variables):
        # -- Initialise figure and axes
        figure = plt.figure(figsize=(6, 6), dpi=100)
        axes = plt.axes()
        try:  # try to see if both training and testing data are available
            bins = np.linspace(min([min(training_data['X'][:, i]), min(test_data['X'][:, i])]), max([max(training_data['X'][:, i]), max(test_data['X'][:, i])]), 50)
            bin_centres = np.array([0.5 * (l + h) for l, h in zip(bins[:-1], bins[1:])])

            # -- Plot incorrect test data if available
            X_test_incorrect = test_data['X'][test_data['y'] == 0][:, i]
            y_values, _, _ = plt.hist(X_test_incorrect, bins=bins, weights=test_data['w'][test_data['y'] == 0] / float(sum(test_data['w'][test_data['y'] == 0])), histtype="stepfilled", label="Incorrect (test)", color="red", alpha=0.5)

            # -- Plot correct test data if available --> data may not have a correct category
            X_test_correct = test_data['X'][test_data['y'] == 1][:, i]
            if X_test_correct.size > 0:
                _contents, _, _ = plt.hist(X_test_correct, bins=bins, weights=test_data['w'][test_data['y'] == 1] / float(sum(test_data['w'][test_data['y'] == 1])), histtype="stepfilled", label="Correct (test)", color="blue", alpha=0.5)
                y_values += _contents

            # -- Plot incorrect training data
            X_train_incorrect = training_data['X'][training_data['y'] == 0][:, i]
            _contents, _ = np.histogram(X_train_incorrect, bins=bins, weights=training_data['w'][training_data['y'] == 0] / float(sum(training_data['w'][training_data['y'] == 0])))
            plt.scatter(bin_centres[np.nonzero(_contents)], _contents[np.nonzero(_contents)], label="Incorrect (train)", color="red")
            y_values += _contents

            # -- Plot correct training data --> data may not have a correct category
            X_train_correct = training_data['X'][training_data['y'] == 1][:, i]
            if X_train_correct.size > 0:
                _contents, _ = np.histogram(X_train_correct, bins=bins, weights=training_data['w'][training_data['y'] == 1] / float(sum(training_data['w'][training_data['y'] == 1])))
                plt.scatter(bin_centres[np.nonzero(_contents)], _contents[np.nonzero(_contents)], label="Correct (train)", color="blue")
                y_values += _contents

        except (IndexError, ValueError):
            non_empty = training_data if len(training_data['y']) > 0 else test_data
            bins = np.linspace(min(non_empty['X'][:, i]), max(non_empty['X'][:, i]), 50)
            y_values, _, _ = plt.hist(non_empty['X'][non_empty['y'] == 0][:, i], bins=bins, weights=non_empty['w'][non_empty['y'] == 0] / float(sum(non_empty['w'][non_empty['y'] == 0])), histtype="stepfilled", label="Incorrect", color="red", alpha=0.5)
            try:
                _contents, _, _ = plt.hist(non_empty['X'][non_empty['y'] == 1][:, i], bins=bins, weights=non_empty['w'][non_empty['y'] == 1] / float(sum(non_empty['w'][non_empty['y'] == 1])), histtype="stepfilled", label="Correct", color="blue", alpha=0.5)
                y_values += _contents
            except ValueError:  # when running on bkg only, we don't have any correct pairs!
                pass

        # -- Plot legend/axes/etc.
        plt.legend(loc="upper right", fontsize=15)
        plt.xlabel(ROOT_2_LATEX[variable])
        plt.ylabel("Fraction of events")
        axes.set_xlim(min(bins), max(bins))
        if variable in ["pT_jb", "pT_j", "m_jb"]:
            axes.set_yscale("log", nonposy='clip')
            axes.set_ylim(min(y_values[np.nonzero(y_values)]), 100*max(y_values))
        else:
            axes.set_ylim(min(y_values), max(y_values))
        plot_atlas.use_atlas_labels(axes)

        # -- Write figure and close plot to save memory
        utils.ensure_directory(plot_directory)
        figure.savefig(os.path.join(plot_directory, "{}.pdf".format(variable)))
        plt.close(figure)
