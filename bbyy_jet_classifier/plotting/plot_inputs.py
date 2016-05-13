
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import plot_atlas

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
    "MV2c20_FCBE_70": r"MV2c20 FCBE(70%)", 
    "MV2c20_FCBE_77": r"MV2c20 FCBE(77%)",
    "MV2c20_FCBE_85": r"MV2c20 FCBE(85%)" 
    }


def input_distributions(classification_variables, training_data, test_data, directory):
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
    logging.getLogger("plotting.input_distributions").info("Plotting input distributions")
    plot_atlas.set_style()

    # -- Plot distributions of input variables
    for i, variable in enumerate(classification_variables):
        # -- Initialise figure and axes
        figure = plt.figure(figsize=(6, 6), dpi=100)
        axes = plt.axes()
        try:
            bins = np.linspace(min([min(training_data['X'][:, i]), min(test_data['X'][:, i])]), max([max(training_data['X'][:, i]), max(test_data['X'][:, i])]), 50)
            bin_centres = np.array([0.5 * (l + h) for l, h in zip(bins[:-1], bins[1:])])

            # -- Plot test data if available --> test data may not have a correct category?
            X_test_correct = test_data['X'][test_data['y'] == 1][:, i]
            X_test_incorrect = test_data['X'][test_data['y'] == 0][:, i]
            if X_test_correct.size > 0:
                y_1, _, _ = plt.hist(X_test_correct, bins=bins, weights=test_data['w'][test_data['y'] == 1] / float(sum(test_data['w'][test_data['y'] == 1])), histtype="stepfilled", label="Correct (test)", color="blue", alpha=0.5)
            y_2, _, _ = plt.hist(X_test_incorrect, bins=bins, weights=test_data['w'][test_data['y'] == 0] / float(sum(test_data['w'][test_data['y'] == 0])), histtype="stepfilled", label="Incorrect (test)", color="red", alpha=0.5)

            # -- Plot training data --> certainly available, otherwise we would be in the `except`
            X_train_correct = training_data['X'][training_data['y'] == 1][:, i]
            X_train_incorrect = training_data['X'][training_data['y'] == 0][:, i]
            _contents, _ = np.histogram(X_train_correct, bins=bins, weights=training_data['w'][training_data['y'] == 1] / float(sum(training_data['w'][training_data['y'] == 1])))
            plt.scatter(bin_centres[np.nonzero(_contents)], _contents[np.nonzero(_contents)], label="Correct (train)", color="blue")
            _contents, _ = np.histogram(X_train_incorrect, bins=bins, weights=training_data['w'][training_data['y'] == 0] / float(sum(training_data['w'][training_data['y'] == 0])))
            plt.scatter(bin_centres[np.nonzero(_contents)], _contents[np.nonzero(_contents)], label="Incorrect (train)", color="red")

        except IndexError:
            non_empty = training_data if len(training_data['y']) > 0 else test_data
            bins = np.linspace(min(non_empty['X'][:, i]), max(non_empty['X'][:, i]), 50)
            y_1, _, _ = plt.hist(non_empty['X'][non_empty['y'] == 1][:, i], bins=bins, 
                weights=non_empty['w'][non_empty['y'] == 1] / float(sum(non_empty['w'][non_empty['y'] == 1])), 
                histtype="stepfilled", label="Correct", color="blue", alpha=0.5)
            y_2, _, _ = plt.hist(non_empty['X'][non_empty['y'] == 0][:, i], bins=bins, 
                weights=non_empty['w'][non_empty['y'] == 0] / float(sum(non_empty['w'][non_empty['y'] == 0])), 
                histtype="stepfilled", label="Incorrect", color="red", alpha=0.5)

        # -- Plot legend/axes/etc.
        plt.legend(loc="upper right", fontsize=15)
        plt.xlabel(ROOT_2_LATEX[variable])
        plt.ylabel("Fraction of events")
        axes.set_xlim(min(bins), max(bins))
        axes.set_ylim([0, max([1e-10, max(y_1 + y_2)])]) # ???
        plot_atlas.use_atlas_labels(axes)

        # -- Write figure and close plot to save memory
        figure.savefig(os.path.join(directory, "{}.pdf".format(variable)))
        plt.close(figure)
