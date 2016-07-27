import cPickle
import logging
import numpy as np
import tqdm

def main(pickle_path):
	d = cPickle.load(open(pickle_path, 'rb'))
	print_performance(
                d['yhat_test_ev'], 
                d['yhat_mHmatch_test_ev'], 
                d['yhat_pThigh_test_ev'], 
                d['y_event'], 
                d['mjb_event'], 
                d['w_test']
                )


def print_performance(yhat_test_ev, yhat_mHmatch_test_ev, yhat_pThigh_test_ev, y_event, mjb_event, w_test):
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
    logger = logging.getLogger("EventPerformance")
    logger.info('Number of correctly classified events for BDT = {} out of {} events having a correct pair'.format(
        *count_correct_total(yhat_test_ev, y_event)))
    logger.info('Number of correctly classified events for mHmatch = {} out of {} events having a correct pair'.format(
        *count_correct_total(yhat_mHmatch_test_ev, y_event)))
    logger.info('Number of correctly classified events for pThigh = {} out of {} events having a correct pair'.format(
        *count_correct_total(yhat_pThigh_test_ev, y_event)))
    logger.info('Number of events without any correct pair = {}'.format(sum([sum(y_event[ev]) == 0 for ev in xrange(len(y_event))])))

    # check whether selected pair has m_jb in mass window for truly correct and truly incorrect pairs
    # -- this will make little sense for SM_merged because lots of events are bkg and shouldn't fall in m_bj window, but can't tell them
    # -- apart without mcChannel number --> use unmerged samples in that case
    # 3 categories: truly correct pair present and got it right, truly correct pair present and got it wrong, no correct pair present
    # -- check this for all 3 strategies (BDT, mHmatch, pThigh)
    # 1. was there a correct pair?
    correct_present_truth = np.array([(sum(y_event[ev]) == 1) for ev in tqdm(xrange(len(y_event))) ]) 
    # ^ this is strategy agnostic, can be calculated outside
    in_mjb_window(mjb_event, y_event, yhat_test_ev, w_test, correct_present_truth)
    in_mjb_window(mjb_event, y_event, yhat_mHmatch_test_ev, w_test, correct_present_truth)
    in_mjb_window(mjb_event, y_event, yhat_pThigh_test_ev, w_test, correct_present_truth)


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


def in_mjb_window(mjb_event, y_event, yhat_test_ev, w_test, correct_present_truth):
    # -- if there was a correct pair and we got it right, how many times does it fall into m_jb? how many times does it not?
    # -- if there was a correct pair and we got it wrong, how many times does it fall into m_jb? how many times does it not?
    # -- if there was no correct pair, how many times does the pair we picked fall into m_jb? how many times does it not?
    # 1. was there a correct pair?
    # correct_present_truth
    # 2. does the bdt agree with the truth label? aka got it right?
    agree_with_truth = np.array([(np.argmax(yhat_test_ev[ev]) == np.argmax(y_event[ev])) for ev in tqdm(xrange(len(y_event))) ])
    # 3. truly correct present and selected (A)
    correct_truth_correct_BDT = np.array(np.logical_and(correct_present_truth, agree_with_truth))
    # 4. truly correct present but selected other pair (B)
    correct_truth_incorrect_BDT = np.array(np.logical_and(correct_present_truth, -agree_with_truth))
    # 5. no correct jet present = - correct_present_truth (C)

    # -- look at mjb for these 3 cases:
    # -- boolean
    in_mjb = np.array([np.logical_and(mjb_event[ev] < 135, mjb_event[ev] > 95) for ev in tqdm(xrange(len(mjb_event))) ])

    # -- weights * boolean
    weighted_in_mjb = np.array([(w_test[ev] * in_mjb[ev]) for ev in tqdm(xrange(w_test.shape[0]))])

    inA = np.array([weighted_in_mjb[correct_truth_correct_BDT][ev][np.argmax(np.array(yhat_test_ev)[correct_truth_correct_BDT][ev])] \
        for ev in tqdm(xrange(sum(correct_truth_correct_BDT))) ])

    inB = np.array([weighted_in_mjb[correct_truth_incorrect_BDT][ev][np.argmax(np.array(yhat_test_ev)[correct_truth_incorrect_BDT][ev])] \
        for ev in tqdm(xrange(sum(correct_truth_incorrect_BDT))) ])

    #events without a correct pair, where the pair with the highest bdt output falls in the mjb region
    inC = np.array([weighted_in_mjb[- correct_present_truth][ev][np.argmax(np.array(yhat_test_ev)[- correct_present_truth][ev])] \
        for ev in tqdm(xrange(sum(- correct_present_truth))) ])
  
    num_inA, num_inB, num_inC = sum(inA), sum(inB), sum(inC)
    #num_notinA, num_notinB, num_notinC = sum(-inA), sum(-inB), sum(-inC)
    print 'Total number of events with a correct pair present and identified = {}'.format(sum(
        [(w_test[ev] * correct_truth_correct_BDT[ev]) for ev in xrange(w_test.shape[0])] ))
    print 'Of these events, {} fall in m_jb window'.format(num_inA)#, num_notinA)
    print 'Total number of events with a correct pair present but a different one selected = {}'.format(sum([w_test[ev] * correct_truth_incorrect_BDT[ev] for ev in xrange(w_test.shape[0])]))
    print 'Of these events, {} fall in m_jb window'.format(num_inB)#, num_notinB)
    print 'Total number of events without a correct pair = {}'.format(sum([w_test[ev] * (- correct_present_truth)[ev] for ev in xrange(w_test.shape[0])]))
    print 'Of these events, out of the ones selected by the classifier, {} fall in m_jb window'.format(num_inC)#, num_notinC)


if __name__ == '__main__':
	import sys
	import argparse

	parser = argparse.ArgumentParser(
        description="Check event level performance")
	parser.add_argument("pickle_path", type=str,
                        help="input pickle file path")
	args = parser.parse_args()
	sys.exit(main(args.pickle_path))


