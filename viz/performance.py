"""
performance.py
author: Luke de Oliveira (lukedeo@stanford.edu)

Usage:
------
>>> weights = np.ones(n_samples)
>>> # -- going to match bkg to signal
>>> weights[signal == True] = get_weights(sig_pt, bkg_pt)
>>> discs = {}
>>> add_curve(r"\tau_{32}", "red", calculate_roc(signal, tau_32, weights=weights))
>>> fg = ROC_plotter(discs)
>>> fg.savefig("myroc.pdf")

"""

import numpy as np
import matplotlib.pyplot as plt
import rootpy.plotting as rpp
from sklearn.metrics import roc_curve


def get_weights(target, actual, bins=10, cap=10, match=True):
    """
    re-weights a actual distribution to a target.

    Args:
        target (array/list): observations drawn from target distribution
        actual (array/list): observations drawn from distribution to
            match to the target.

        bins (numeric or list/array of numerics): bins to use to do weighting

        cap (numeric): maximum weight value.

        match (bool): whether to make the sum of weights in actual equal to the
            number of samples in target

    Returns:
        numpy.array: returns array of shape len(actual).
    """
    target_counts, target_bins = np.histogram(target, bins=bins)
    counts, _ = np.histogram(actual, bins=target_bins)
    counts = (1.0 * counts)
    counts = np.array([max(a, 0.0001) for a in counts])
    multiplier = np.array((target_counts / counts).tolist() + [1.0])

    weights = np.array([min(multiplier[target_bins.searchsorted(point) - 1], cap) for point in actual])

    if match:
        weights *= (len(target) / np.sum(weights))

    return weights


def calculate_roc(labels, discriminant, weights=None):
    """
    Definition:
    -----------
        Use the scikit-learn roc_curve function to calculate signal efficiency and background rejection

    Args:
    -----
        labels = an array of 1/0 representing signal/background
        discriminant  = an array that represents the discriminant
        weights = sample weights for each point
                  assert(weights.shape == discriminant.shape)

    Returns:
    --------
        tuple: (signal_efficiency, background_rejection) where each are arrays
    """
    fpr, tpr, _ = roc_curve(labels, discriminant, sample_weight=weights)
    sig_eff = tpr[np.nonzero(fpr)]  # values of tpr where fpr != 0
    bkg_rej = np.reciprocal(fpr[np.nonzero(fpr)])  # values of fpr where fpr != 0
    return sig_eff, bkg_rej


def ROC_plotter(curves, min_eff=0, max_eff=1, min_rej=1, max_rej=10**4, pp=False, **kwargs):
    """
    Definition:
    -----------
        Plot pre-calculated ROC distributions

    Args:
    -----
        curves = a dictionary of names -> ROC distributions
        min_eff = minimum signal efficiency
        max_eff = maximum signal efficiency
        min_rej = minimum background rejection
        max_rej = maximum background rejection
        pp = boolean switch to determine whether to save output (true) or return the figure (false)
        kwargs = additional keywords arguments for plotting (eg. linestyle, linewidth etc.)

    Returns:
    --------
        figure: a matplotlib figure
    """
    # -- Initialise figure, axes and binning
    figure = plt.figure(figsize=(6, 6), dpi=100)
    axes = plt.axes()
    plt.title(r"" + kwargs.pop("title", ""))
    if kwargs.pop("logscale", False):
        axes.set_yscale("log")

    # -- Plot data
    for tagger, data in curves.iteritems():
        sel = (data["efficiency"] >= min_eff) & (data["efficiency"] <= max_eff)
        plt.plot(data["efficiency"][sel], data["rejection"][sel], "-", label=r"" + tagger, color=data["color"], **kwargs)

    # -- Plot legend/axes/etc.
    plt.legend()
    plt.xlim(min_eff, max_eff)
    plt.ylim(min_rej, max_rej)
    # axes.set_xlabel(r"$\varepsilon_{\mathrm{signal}}$")
    # axes.set_ylabel(r"$1 / \varepsilon_{\mathrm{background}}$")
    plt.xlabel(r"$\varepsilon_{\mathsf{signal}}$")  # , position=(1., 0), va="bottom", ha="right")
    plt.ylabel(r"$1 / \varepsilon_{\mathsf{background}}$")  # , position=(0, 1.), va="top", ha="right")

    # -- Save figure or return it
    if pp:
        plt.savefig(figure)
    else:
        return figure, axes


def add_curve(name, color, curve_pair):
    return {
        name: {
            "efficiency": curve_pair[0],
            "rejection": curve_pair[1],
            "color": color
        }
    }
