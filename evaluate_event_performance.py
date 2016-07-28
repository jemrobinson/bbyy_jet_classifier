'''
run:
python evaluate_event_performance.py hh_yybb/for_event_performance.pkl \
X275_hh/for_event_performance.pkl X300_hh/for_event_performance.pkl \
X325_hh/for_event_performance.pkl X350_hh/for_event_performance.pkl \
X400_hh/for_event_performance.pkl Sherpa_photon_jet/for_event_performance.pkl
'''
import cPickle
import logging
from itertools import izip
import numpy as np
from tabulate import tabulate
from bbyy_jet_classifier import utils

INMJB_PATH = 'event-level-perf.pkl'
BKG_NAME = 'Sherpa_photon_jet'

def main(pickle_paths):
    logger = logging.getLogger("main")
    
    try:
        perf_dict = cPickle.load(open(INMJB_PATH, 'rb'))
        logger.info('Dictionary found and loaded from ' + INMJB_PATH)

    except IOError: # if it doesn't exist, create it
        logger.info('Dictionary not found in ' + INMJB_PATH + '. Processing data...')
        perf_dict = {}
        for path in pickle_paths:
            logger.info('\nWorking on ' + path)
            d = cPickle.load(open(path, 'rb'))
            perf_dict[path.split('/')[0]] = eval_performance(
                            d['yhat_test_ev'], 
                            d['yhat_mHmatch_test_ev'], 
                            d['yhat_pThigh_test_ev'], 
                            d['y_event'], 
                            d['mjb_event'], 
                            d['w_test']
                            )
        cPickle.dump(perf_dict, open(INMJB_PATH, 'wb'))

    headers = [path.split('/')[0] for path in pickle_paths if path.split('/')[0] != BKG_NAME]
    logger.info('Asimov significance per class:\n{}'.format(
        tabulate([
            [strategy] +  \
            [asimov(perf_dict[_class][strategy], perf_dict[BKG_NAME][strategy]) 
                for _class in headers] 
        for strategy in ['BDT', 'mHmatch', 'pThigh']
        ],
        headers=[""] + headers,
        floatfmt=".5f")
        )
    )

def asimov(s, b):
    '''
    Definition:
    -----------
        Calculates signal to background sensitivity according to the Asimov formula
    Args:
    -----
        s: float, the number of jet pairs properly classified as 'correct' that fall in the m_jb window
        b: float, the number of jet pairs mistakenly classified as 'correct' that fall in the m_jb window
    Returns:
    --------
        The result of the Asimov formula given s and b
    '''
    import math
    return math.sqrt(2 * ((s+b) * math.log(1 + (s/b)) - s))


def eval_performance(yhat_test_ev, yhat_mHmatch_test_ev, yhat_pThigh_test_ev, y_event, mjb_event, w_test):
    '''
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
    '''
    logger = logging.getLogger("eval_performance")
    logger.info('BDT: Number of correctly classified events = {} out of {} events having a correct pair'.format(
        *count_correct_total(yhat_test_ev, y_event)))
    logger.info('mHmatch: Number of correctly classified events = {} out of {} events having a correct pair'.format(
        *count_correct_total(yhat_mHmatch_test_ev, y_event)))
    logger.info('pThigh: Number of correctly classified events = {} out of {} events having a correct pair'.format(
        *count_correct_total(yhat_pThigh_test_ev, y_event)))
    logger.info('Number of events without any correct pair = {}'.format(sum([sum(y_event[ev]) == 0 for ev in xrange(len(y_event))])))

    # check whether selected pair has m_jb in mass window for truly correct and truly incorrect pairs
    # -- this will make little sense for SM_merged because lots of events are bkg and shouldn't fall in m_bj window, but can't tell them
    # -- apart without mcChannel number --> use unmerged samples in that case
    # 3 categories: truly correct pair present and got it right, truly correct pair present and got it wrong, no correct pair present
    # -- check this for all 3 strategies (BDT, mHmatch, pThigh)
    # 1. was there a correct pair?
    correct_present_truth = np.array([sum(ev) == 1 for ev in y_event]) 
    # ^ this is strategy agnostic, can be calculated outside
    in_BDT =  in_mjb_window(mjb_event, y_event, yhat_test_ev, w_test, correct_present_truth, 'BDT')
    in_mHmatch = in_mjb_window(mjb_event, y_event, yhat_mHmatch_test_ev, w_test, correct_present_truth, 'mHmatch')
    in_pThigh = in_mjb_window(mjb_event, y_event, yhat_pThigh_test_ev, w_test, correct_present_truth, 'pThigh')
    return {
            'BDT' : in_BDT,
            'mHmatch' : in_mHmatch,
            'pThigh' : in_pThigh
            }


def count_correct_total(yhat, y):
    '''
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
        n_correct_truth:      int, total number of events with a 'correct' jet pair
    '''
    # -- find how many times we find the correct pair in all events that do have a correct pair
    # correct_classifier = a truly correct pair existed (sum(y[ev]) == 1) and we got it right (np.argmax(yhat[ev]) == np.argmax(y[ev]))
    n_correct_classifier = sum([np.argmax(yhat[ev]) == np.argmax(y[ev]) for ev in xrange(len(y)) if sum(y[ev]) == 1])
    # correct_truth = a truly correct pair exists
    n_correct_truth = sum([sum(y[ev]) == 1 for ev in xrange(len(y))])
    return n_correct_classifier, n_correct_truth


def in_mjb_window(mjb_event, y_event, yhat_test_ev, w_test, correct_present_truth, strategy):
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

    def _weightedsum_eventsinmjb(weights_in_mjb, yhat, slicer):
        sliced_weights = weights_in_mjb[slicer]
        sliced_yhat = np.array(yhat_test_ev)[slicer]
        return np.sum(w[np.argmax(y)] for w, y in izip(sliced_weights, sliced_yhat))
      
    num_inA = _weightedsum_eventsinmjb(weights_in_mjb, yhat_test_ev, correct_truth_correct_BDT) 
    num_inB = _weightedsum_eventsinmjb(weights_in_mjb, yhat_test_ev, correct_truth_incorrect_BDT)
    num_inC = _weightedsum_eventsinmjb(weights_in_mjb, yhat_test_ev, -correct_present_truth)

    logger.info('Total number of events with a correct pair present and identified = {}'.format(
        sum((w * c) for w, c in izip(w_test, correct_truth_correct_BDT))
        ))
    logger.info('Of these events, {} fall in m_jb window'.format(num_inA))
    logger.info('Total number of events with a correct pair present but a different one selected = {}'.format(
        sum((w * c) for w, c in izip(w_test, correct_truth_incorrect_BDT))
        ))
    logger.info('Of these events, {} fall in m_jb window'.format(num_inB))
    logger.info('Total number of events without a correct pair = {}'.format(
        sum((w * c) for w, c in izip(w_test, -correct_present_truth))
        ))
    logger.info('Of these events, out of the ones selected by the classifier, {} fall in m_jb window'.format(num_inC))
    logger.info('Total number of events in the m_jb window = {}'.format(num_inA + num_inB + num_inC))
    return num_inA + num_inB + num_inC

if __name__ == '__main__':
    import sys
    import argparse
    utils.configure_logging()

    parser = argparse.ArgumentParser(
        description="Check event level performance")
    parser.add_argument("pickle_paths",
                        help="list of input pickle file paths", 
                        type=str, nargs="+", default=[])
    args = parser.parse_args()
    sys.exit(main(args.pickle_paths))

