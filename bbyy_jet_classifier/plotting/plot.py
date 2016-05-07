import cPickle
import logging
import os
import matplotlib.pyplot as plt
import numpy as np
import rootpy.plotting as rpp
from viz import add_curve, calculate_roc, ROC_plotter

def atlaslabel(ax, fontsize=20):
    # -- Add ATLAS text
    plt.text(0.1,0.9, 'ATLAS', va='bottom', ha='left', color='black', size=fontsize, 
             fontname = 'sans-serif', weight = 'bold', style = 'oblique',transform=ax.transAxes)
    plt.text(0.23, 0.9, 'Work In Progress', va='bottom', ha='left', color='black', size=fontsize,
            fontname = 'sans-serif', transform=ax.transAxes)
    plt.text(0.1, 0.83, 
             r'$\sqrt{s} = 13 TeV :\ \ \int Ldt = 1.04\ fb^{-1}$', # change number according to actual data used!
             va='bottom', ha='left', color='black', size=fontsize, fontname = 'sans-serif', transform=ax.transAxes)


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
    atlaslabel(axes, fontsize=10)
    figure.savefig(os.path.join(outdir, "testing", "{}.pdf".format(old_strategy_name)))


def input_distributions(strategy, classification_variables, X, y, w, process):
    """
    Definition:
    -----------
            Plots the distributions of input variables, color-coded by target class

    Args:
    -----
            strategy = a classification method -- here either sklBDT or RootTMVA
            X = the feature matrix (ndarray)
            y = the target array
            w = the array of weights
            process = string, either "training" or "testing", usually
    """
    rpp.set_style("ATLAS", mpl=True)
    logging.getLogger("Plotting").info("Plotting input distributions")

    # -- Ensure output directory exists
    strategy.ensure_directory(os.path.join(strategy.output_directory, process))

    # -- Plot distributions of input variables
    for i, variable in enumerate(classification_variables):
        data_correct = X[y == 1][:, i]
        data_incorrect = X[y == 0][:, i]

        figure = plt.figure(figsize=(6, 6), dpi=100)
        axes = plt.axes()
        bins = np.linspace(min([min(data_correct), min(data_incorrect)]), max([max(data_correct), max(data_incorrect)]), 50)
        y_1, _, _ = plt.hist(data_correct, weights=w[y == 1] / float(sum(w[y == 1])), bins=bins, histtype="stepfilled", label="Correct", color="blue", alpha=0.5)
        y_2, _, _ = plt.hist(data_incorrect, weights=w[y == 0] / float(sum(w[y == 0])), bins=bins, histtype="stepfilled", label="Incorrect", color="red", alpha=0.5)
        plt.legend(loc="upper right")
        plt.xlabel(variable, position=(1., 0), va="bottom", ha="right")
        plt.ylabel("Fraction of events", position=(0., 1.), va="top", ha="right")
        axes.xaxis.set_label_coords(1., -0.15)
        axes.yaxis.set_label_coords(-0.2, 1.)
        axes.set_ylim([0, 1.3 * max([1e-5, max(y_1), max(y_2)])])
        atlaslabel(axes, fontsize=10)
        figure.savefig(os.path.join(strategy.output_directory, process, "{}.pdf".format(variable)))
        plt.close(figure)


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
    atlaslabel(axes, fontsize=10)
    figure.savefig(os.path.join(ML_strategy.output_directory, process, "BDT_{}.pdf".format(fileID)))
    plt.close(figure)


def roc(ML_strategy, mHmatch_test, pThigh_test, yhat_test, y_test, w_test):
    """
    Definition:
    -----------
        Check performance of ML_strategy by plotting its ROC curve and comparing it with the points generated by the old strategies

    Args:
    -----
        ML_strategy = one of the machine learning strategy in strategies/ whose prerformance we want to visualize
        mHmatch_test = array of dim (# testing examples), containing the binary decision based the "closest mH" strategy
        pThigh_test = array of dim (# testing examples), containing the binary decision based the "highest pT" strategy
        yhat_test = array of dim (# testing examples), with predictions from the ML_strategy
        y_test = array of dim (# testing examples) with target values
        w_test = array of dim (# testing examples) with event weights
    """
    rpp.set_style("ATLAS", mpl=True)
    logging.getLogger("Plotting").info("Plotting performance")

    # -- Calculate efficiencies from the older strategies
    eff_mH_signal = float(sum((mHmatch_test * w_test)[y_test == 1])) / float(sum(w_test[y_test == 1]))
    eff_mH_bkg = float(sum((mHmatch_test * w_test)[y_test == 0])) / float(sum(w_test[y_test == 0]))
    eff_pT_signal = float(sum((pThigh_test * w_test)[y_test == 1])) / float(sum(w_test[y_test == 1]))
    eff_pT_bkg = float(sum((pThigh_test * w_test)[y_test == 0])) / float(sum(w_test[y_test == 0]))

    ML_strategy.ensure_directory(os.path.join(ML_strategy.output_directory, "pickle"))
    cPickle.dump(
        {"eff_mH_signal": eff_mH_signal,
         "eff_mH_bkg": eff_mH_bkg,
         "eff_pT_signal": eff_pT_signal,
         "eff_pT_bkg": eff_pT_bkg
         }, open(os.path.join(ML_strategy.output_directory, "pickle", "old_strategies_dict.pkl"), "wb"))

    # -- Add ROC curves and efficiency points for old strategies
    discs = {}
    add_curve(ML_strategy.name, "black", calculate_roc(y_test, yhat_test), discs)
    fg = ROC_plotter(discs, min_eff=0.1, max_eff=1.0, logscale=True)
    plt.plot(eff_mH_signal, 1.0 / eff_mH_bkg, marker="o", color="r", label=r"Closest m$_{H}$", linewidth=0)  # add point for "mHmatch" strategy
    plt.plot(eff_pT_signal, 1.0 / eff_pT_bkg, marker="o", color="b", label=r"Highest p$_{T}$", linewidth=0)  # add point for "pThigh" strategy
    plt.legend()
    axes = plt.axes()
    atlaslabel(axes)
    fg.savefig(os.path.join(ML_strategy.output_directory, "ROC.pdf"))
    # -- Save out ROC curve as pickle for later comparison
    cPickle.dump(discs[ML_strategy.name], open(os.path.join(ML_strategy.output_directory, "pickle", "{}_ROC.pkl".format(ML_strategy.name)), "wb"), cPickle.HIGHEST_PROTOCOL)

