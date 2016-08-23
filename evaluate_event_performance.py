#! /usr/bin/env python
import cPickle
import glob
import logging
import os
from itertools import izip
from joblib import Parallel, delayed
import numpy as np
from bbyy_jet_classifier import utils

BKG_NAME = "SM_bkg_photon_jet"


def main(pickle_paths, THRESHOLD):
    logger = logging.getLogger("event_performance.main")

    logger.info("Processing data...")
    perf_dict = {}
    for path in pickle_paths:
        logger.info("Working on: ")
        logger.info(path)
        d = cPickle.load(open(path, "rb"))
        perf_dict[path.split("/")[0]] = eval_performance(
            d["yhat_test_ev"],
            d["yhat_mHmatch_test_ev"],
            d["yhat_pThigh_test_ev"],
            d["y_event"],
            d["mjb_event"],
            d["w_test"],
            THRESHOLD=THRESHOLD
        )

    headers = [path.split("/")[2] for path in pickle_paths if path.split("/")[2] != BKG_NAME]

    if hasattr(THRESHOLD, "__iter__"):
        asimov_dict = {
            _class: {
                strategy: map(np.array, [THRESHOLD, [asimov(s, b) for s, b in zip(perf_dict[_class][strategy], perf_dict[BKG_NAME][strategy])]])
                for strategy in ["BDT", "mHmatch", "pThigh"]
            }
            for _class in headers
        }

    else:
        asimov_dict = {
            _class: {
                strategy: map(np.array, [[THRESHOLD], [asimov(s, b) for s, b in zip(perf_dict[_class][strategy], perf_dict[BKG_NAME][strategy])]])
                for strategy in ["BDT", "mHmatch", "pThigh"]
            }
            for _class in headers
        }

    with open("multi_proc_TMVA.pkl", "wb") as f:
        cPickle.dump(asimov_dict, f)


def asimov(s, b):
    """
    Definition:
    -----------
        Calculates signal to background sensitivity according to the Asimov formula
    Args:
    -----
        s: float, the number of jet pairs properly classified as "correct" that fall in the m_jb window
        b: float, the number of jet pairs mistakenly classified as "correct" that fall in the m_jb window
    Returns:
    --------
        The result of the Asimov formula given s and b
    """
    import math
    return math.sqrt(2 * ((s + b) * math.log(1 + (s / b)) - s))


def eval_performance(yhat_test_ev, yhat_mHmatch_test_ev, yhat_pThigh_test_ev, y_event, mjb_event, w_test, THRESHOLD):
    """
    Definition:
    -----------
        Log event-level performance outputs as info
    Args:
    -----
        yhat_test_ev: event level numpy array containing the predictions from the BDT for each jet in the event
        yhat_mHmatch_test_ev: event level numpy array containing the predictions from mHmatch for each jet in the event
        yhat_pThigh_test_ev: event level numpy array containing the predictions from pThigh for each jet in the event
        y_event: event level numpy array containing the truth labels for each jet in the event
        mjb_event: event level numpy array containing the values of m_jb for each jet in the event
        THRESHOLD: an integer or iterable of integers
    """
    logger = logging.getLogger("eval_performance")
    logger.debug("BDT: Number of correctly classified events = {} out of {} events having a correct pair".format(*count_correct_total(yhat_test_ev, y_event)))
    logger.debug("mHmatch: Number of correctly classified events = {} out of {} events having a correct pair".format(*count_correct_total(yhat_mHmatch_test_ev, y_event)))
    logger.debug("pThigh: Number of correctly classified events = {} out of {} events having a correct pair".format(*count_correct_total(yhat_pThigh_test_ev, y_event)))
    logger.debug("Number of events without any correct pair = {}".format(sum([sum(y_event[ev]) == 0 for ev in xrange(len(y_event))])))

    # check whether selected pair has m_jb in mass window for truly correct and truly incorrect pairs
    # -- this will make little sense for SM_merged because lots of events are bkg and shouldn"t fall in m_bj window, but can"t tell them
    # -- apart without mcChannel number --> use unmerged samples in that case
    # 3 categories: truly correct pair present and got it right, truly correct pair present and got it wrong, no correct pair present
    # -- check this for all 3 strategies (BDT, mHmatch, pThigh)
    # 1. was there a correct pair?
    correct_present_truth = np.array([sum(ev) == 1 for ev in y_event])
    # ^ this is strategy agnostic, can be calculated outside
    in_BDT = in_mjb_window(mjb_event, y_event, yhat_test_ev, w_test, correct_present_truth, "BDT", THRESHOLD)
    in_mHmatch = in_mjb_window(mjb_event, y_event, yhat_mHmatch_test_ev, w_test, correct_present_truth, "mHmatch", THRESHOLD)
    in_pThigh = in_mjb_window(mjb_event, y_event, yhat_pThigh_test_ev, w_test, correct_present_truth, "pThigh", THRESHOLD)
    return {"BDT": in_BDT, "mHmatch": in_mHmatch, "pThigh": in_pThigh}


def count_correct_total(yhat, y):
    """
    Definition:
    -----------
        Quantify the number of events in which the correct jet pair was assigned the highest classifier score
    Args:
    -----
        yhat: event level numpy array containing the predictions for each jet in the event
        y:    event level numpy array containing the truth labels for each jet in the event
    Returns:
    --------
        n_correct_classifier: int, number of events in which the correct jet pair was assigned the highest classifier score
        n_correct_truth:      int, total number of events with a "correct" jet pair
    """
    # -- find how many times we find the correct pair in all events that do have a correct pair
    # correct_classifier = a truly correct pair existed (sum(y[ev]) == 1) and we got it right (np.argmax(yhat[ev]) == np.argmax(y[ev]))
    n_correct_classifier = sum([np.argmax(yhat[ev]) == np.argmax(y[ev]) for ev in xrange(len(y)) if sum(y[ev]) == 1])
    # correct_truth = a truly correct pair exists
    n_correct_truth = sum([sum(y[ev]) == 1 for ev in xrange(len(y))])
    return n_correct_classifier, n_correct_truth


def _weightedsum_eventsinmjb(weights_in_mjb, yhat, slicer, thresh):
    sliced_weights = weights_in_mjb[slicer]
    sliced_yhat = np.array(yhat)[slicer]
    return np.sum(w[np.argmax(y)] for w, y in izip(sliced_weights, sliced_yhat) if max(y) >= thresh)


def in_mjb_window(mjb_event, y_event, yhat_test_ev, w_test, correct_present_truth, strategy, THRESHOLD):
    logger = logging.getLogger("mjb_window - " + strategy)
    # -- if there was a correct pair and we got it right, how many times does it fall into m_jb? how many times does it not?
    # -- if there was a correct pair and we got it wrong, how many times does it fall into m_jb? how many times does it not?
    # -- if there was no correct pair, how many times does the pair we picked fall into m_jb? how many times does it not?
    # 1. was there a correct pair?
    # correct_present_truth
    # 2. does the bdt agree with the truth label? aka got it right?
    agree_with_truth = np.array([(np.argmax(yhat) == np.argmax(y)) for yhat, y in izip(yhat_test_ev, y_event)])
    # 3. truly correct present and selected (A)
    correct_truth_correct_BDT = np.array(np.logical_and(correct_present_truth, agree_with_truth))
    # 4. truly correct present but selected other pair (B)
    correct_truth_incorrect_BDT = np.array(np.logical_and(correct_present_truth, -agree_with_truth))
    # 5. no correct jet present = - correct_present_truth (C)

    # -- look at mjb for these 3 cases:
    # -- boolean
    in_mjb = [np.logical_and(mjb_event[ev] < 135, mjb_event[ev] > 95) for ev in xrange(len(mjb_event))]

    # -- weights * boolean
    weights_in_mjb = np.array([_w * _m for _w, _m in izip(w_test, in_mjb)])

    if hasattr(THRESHOLD, "__iter__"):

        # num_inX are lists in this scenario
        num_inA = Parallel(n_jobs=20, verbose=True)(delayed(_weightedsum_eventsinmjb)(weights_in_mjb, yhat_test_ev, correct_truth_correct_BDT, thresh) for thresh in THRESHOLD)
        num_inB = Parallel(n_jobs=20, verbose=True)(delayed(_weightedsum_eventsinmjb)(weights_in_mjb, yhat_test_ev, correct_truth_incorrect_BDT, thresh) for thresh in THRESHOLD)
        num_inC = Parallel(n_jobs=20, verbose=True)(delayed(_weightedsum_eventsinmjb)(weights_in_mjb, yhat_test_ev, -correct_present_truth, thresh)for thresh in THRESHOLD)

        return np.array([num_inA, num_inB, num_inC]).sum(axis=0)

    else:
        num_inA = _weightedsum_eventsinmjb(weights_in_mjb, yhat_test_ev, correct_truth_correct_BDT, thresh=THRESHOLD)
        num_inB = _weightedsum_eventsinmjb(weights_in_mjb, yhat_test_ev, correct_truth_incorrect_BDT, thresh=THRESHOLD)
        num_inC = _weightedsum_eventsinmjb(weights_in_mjb, yhat_test_ev, -correct_present_truth, thresh=THRESHOLD)

        logger.debug("Total number of events with a correct pair present and identified = {}".format(sum((w * c) for w, c in izip(w_test, correct_truth_correct_BDT))))
        logger.debug("Of these events, {} fall in m_jb window".format(num_inA))
        logger.debug("Total number of events with a correct pair present but a different one selected = {}".format(sum((w * c) for w, c in izip(w_test, correct_truth_incorrect_BDT))))
        logger.debug("Of these events, {} fall in m_jb window".format(num_inB))
        logger.debug("Total number of events without a correct pair = {}".format(sum((w * c) for w, c in izip(w_test, -correct_present_truth))))
        logger.debug("Of these events, out of the ones selected by the classifier, {} fall in m_jb window".format(num_inC))
        logger.debug("Total number of events in the m_jb window = {}".format(num_inA + num_inB + num_inC))
        return [num_inA + num_inB + num_inC]


if __name__ == "__main__":
    import sys
    import argparse
    utils.configure_logging()

    parser = argparse.ArgumentParser(description="Check event level performance")
    parser.add_argument("--strategy", type=str, help="strategy to evaluate. Options are: RootTMVA, sklBDT.", default="sklBDT")
    parser.add_argument("sample_names", help="list of names of samples to evaluate", type=str, nargs="+", default=[])
    args = parser.parse_args()
    pickle_paths = sum([glob.glob(os.path.join("output", args.strategy, sample_name, "event_performance_dump.pkl")) for sample_name in args.sample_names], [])
    sys.exit(main(pickle_paths, np.linspace(-1, 1, 21)))
