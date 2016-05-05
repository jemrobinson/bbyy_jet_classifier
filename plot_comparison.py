"""
plot_comparison.py

Description:
------------
    Quick script to load and compare ROC curves produced from different classifiers

    To be replaced with an option to train and test > 1 classifier per run
"""

import cPickle
import logging
import os
import matplotlib.pyplot as plt
from viz import ROC_plotter

TMVABDT = cPickle.load(open(os.path.join("output", "RootTMVA", "pickle", "RootTMVA_ROC.pkl"), "rb"))
sklBDT = cPickle.load(open(os.path.join("output", "sklBDT", "pickle", "sklBDT_ROC.pkl"), "rb"))
dots = cPickle.load(open(os.path.join("output", "sklBDT", "pickle", "old_strategies_dict.pkl"), "rb"))

sklBDT["color"] = "green"

curves = {}
curves["sklBDT"] = sklBDT
curves["RootTMVA"] = TMVABDT

logging.getLogger("RunClassifier").info("Plotting")
fg = ROC_plotter(curves, title=r"Performance of Second b-Jet Selection Strategies", min_eff=0.1, max_eff=1.0, ymax=1000, logscale=True)
plt.plot(dots["eff_mH_signal"], 1.0 / dots["eff_mH_bkg"], marker="o", color="r", label=r"Closest m$_{H}$", linewidth=0)  # add point for "mHmatch" strategy
plt.plot(dots["eff_pT_signal"], 1.0 / dots["eff_pT_bkg"], marker="o", color="b", label=r"Highest p$_{T}$", linewidth=0)  # add point for "pThigh" strategy
plt.legend()
fg.savefig(os.path.join("output", "ROCcomparison.pdf"))
