from collections import OrderedDict
import logging
import numpy as np
from numpy.lib.recfunctions import stack_arrays
from root_numpy import root2array
from sklearn.cross_validation import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif

TYPE_2_CHAR = {"int32": "I", "float64": "D", "float32": "F"}

def load(input_filename, treename, excluded_variables, training_fraction, max_events, n_signal_samples):
    """
    Definition:
    -----------
            Data handling function that loads in .root files and turns them into ML-ready python objects

    Args:
    -----
            input_filename = string, the path to the input root file
            excluded_variables = list of strings, names of branches not to use for training
            training_fraction = float between 0 and 1, fraction of examples to use for training

    Returns:
    --------
            classification_variables = list of names of variables used for classification
            variable2type = ordered dict, mapping all the branches from the TTree to their type
            train_data = dictionary, containing "X", "y", "w" for the training set, where:
                X = ndarray of dim (# training examples, # features)
                y = array of dim (# training examples) with target values
                w = array of dim (# training examples) with event weights
            test_data = dictionary, containing "X", "y", "w" for the test set, where:
                X = ndarray of dim (# testing examples, # features)
                y = array of dim (# testing examples) with target values
                w = array of dim (# testing examples) with event weights
            yhat_old_test_data = dictionary, containing "mHmatch", "pThigh" for the test set, where
                mHmatch = array of dim (# testing examples) with output of binary decision based on jet pair with closest m_jb to 125GeV
                pThigh  = array of dim (# testing examples) with output of binary decision based on jet with highest pT
            test_events_data = dictionary, containing "y", "mjb", "pTj" for events in the test set, where
                y    = array of "truth" decisions for each jet
                m_jb = array of mjb for each jet (paired with the tagged jet)
                pT_j = array of pTj for each jet
                w    = array of event weights
    """
    logging.getLogger("process_data").info("Loading input from ROOT file: {}".format(input_filename))
    for v_name in excluded_variables:
        logging.getLogger("process_data").info("... excluding variable {}".format(v_name))
    # -- when using multiple signal samples at once, we need to scale weights of
    #    the signal samples by 1/n_signal_samples otherwise we'll overtrain
    weight_scale_factor = 1.
    if "bkg" not in input_filename :
        logging.getLogger("process_data").info("{} appears to be a signal sample so the event weights will be \033[1;31mscaled down\033[1;0m by a factor of {}".format(input_filename, n_signal_samples))
        weight_scale_factor = 1./float(n_signal_samples)
    # -- import all root files into data_rec
    data_rec = root2array(input_filename, treename)
    # -- ordered dictionary of branches and their type
    variable2type = OrderedDict(((v_name, TYPE_2_CHAR[data_rec[v_name][0].dtype.name]) for v_name in data_rec.dtype.names if v_name not in excluded_variables))
    # -- variables used as inputs to the classifier
    classification_variables = [name for name in variable2type.keys() if name not in ["event_weight", "isCorrect"]]

    # -- throw away events with no jet pairs
    n_events_before_rejection = data_rec.size
    data_rec = data_rec[np.array([len(data_rec["isCorrect"][ev]) > 0 for ev in xrange(data_rec.shape[0])])]
    # -- only use max_events events
    if max_events > 0 :
        data_rec = data_rec[np.random.randint(data_rec.shape[0], size=max_events)]
    logging.getLogger("process_data").info("Found {} events of which {} ({}%) remain after rejecting empty events".format(n_events_before_rejection, data_rec.size, (100*data_rec.size)/n_events_before_rejection))

    # -- slice rec array to only contain input features
    X = data_rec[classification_variables]
    y = data_rec["isCorrect"]
    # -- NB. weights can be positive or negative at NLO
    #    convert one weight per event to one weight per jet
    w = np.array([[data_rec["event_weight"][ev] * weight_scale_factor] * len(data_rec["isCorrect"][ev])for ev in xrange(data_rec.shape[0])])
    yhat_mHmatch = data_rec["idx_by_mH"]
    yhat_pThigh = data_rec["idx_by_pT"]
    yhat_pTjb = data_rec["idx_by_pT_jb"]
    ix = np.array(range(data_rec.shape[0]))

    # -- Can't pass `train_size=1`, but can use `test_size=0`
    if training_fraction == 1:
        split_kwargs = {"test_size":0}
    else:
        split_kwargs = {"train_size":training_fraction}

    # -- Construct training and test datasets, automatically permuted
    X_train, X_test, y_train, y_test, w_train, w_test, _, ix_test, \
        _, yhat_mHmatch_test, _, yhat_pThigh_test, _, yhat_pTjb_test = \
        train_test_split(X, y, w, ix, yhat_mHmatch, yhat_pThigh, yhat_pTjb, **split_kwargs)

    # -- go from event-flat to jet-flat
    y_train, y_test, w_train, w_test, yhat_mHmatch_test, yhat_pThigh_test, yhat_pTjb_test = \
        [flatten(element) for element in [y_train, y_test, w_train, w_test, yhat_mHmatch_test, yhat_pThigh_test, yhat_pTjb_test]]

    X_train = np.array([flatten(X_train[var]) for var in classification_variables]).T
    X_test = np.array([flatten(X_test[var]) for var in classification_variables]).T

    # -- Balance training weights
    w_train = balance_weights(y_train, w_train)

    # -- Put X, y and w into a dictionary to conveniently pass these objects around
    train_data = {"X": X_train, "y": y_train, "w": w_train}
    test_data = {"X": X_test, "y": y_test, "w": w_test}
    yhat_old_test_data = {"mHmatch": yhat_mHmatch_test, "pThigh": yhat_pThigh_test, "pTjb": yhat_pTjb_test}

    # -- Do the same for the event-level arrays
    test_events_data = {"y": data_rec["isCorrect"][ix_test],
                        "m_jb": data_rec["m_jb"][ix_test],
                        "pT_j": data_rec["pT_j"][ix_test],
                        "w": data_rec["event_weight"][ix_test]}

    # -- ANOVA for feature selection (please, know what you're doing)
    if training_fraction > 0:
        feature_selection(train_data, classification_variables, 5)

    return classification_variables, variable2type, train_data, test_data, yhat_old_test_data, test_events_data


def feature_selection(train_data, features, k):
    """
    Definition:
    -----------
            !! ONLY USED FOR INTUITION, IT'S USING A LINEAR MODEL TO DETERMINE IMPORTANCE !!
            Gives an approximate ranking of variable importance and prints out the top k

    Args:
    -----
            train_data = dictionary containing keys "X" and "y" for the training set, where:
                X = ndarray of dim (# training examples, # features)
                y = array of dim (# training examples) with target values
            features = names of features used for training in the order in which they were inserted into X
            k = int, the function will print the top k features in order of importance
    """

    # -- Select the k top features, as ranked using ANOVA F-score
    tf = SelectKBest(score_func=f_classif, k=k)
    tf.fit_transform(train_data["X"], train_data["y"])

    # -- Return names of top features
    logging.getLogger("process_data").info("The {} most important features are {}".format(k, [f for (_, f) in sorted(zip(tf.scores_, features), reverse=True)][:k]))


def balance_weights(y_train, w_train, targetN=10000):
    """
    Definition:
    -----------
        Function that rebalances the class weights
        This is useful because we often train on datasets with very different quantities of signal and background
        This allows us to bring the samples back to equal quantities of signal and background

    Args:
    -----
        y_train = array of dim (# training examples) with target values
        w_train = array of dim (# training examples) with the initial weights as extracted from the ntuple
        targetN(optional, default to 10000) = target equal number of signal and background events

    Returns:
    --------
        w_train = array of dim (# training examples) with the new rescaled weights
    """

    for classID in np.unique(y_train):
        w_train[y_train == classID] *= float(targetN) / float(np.sum(w_train[y_train == classID]))

    return w_train


def combine_datasets(dataset_list):
    """
    Definition:
    -----------
        Function that combines a list datasets into a single dataset
        Each of the inputs (and the output) should have the form {"X":data, "y":recarray, "w":recarray}
        This allows us to combine datasets from different input files

    Args:
    -----
        dataset_list = array of dictionaries of the form {"X":data, "y":recarray, "w":recarray}

    Returns:
    --------
        dictionary of the form {"X":data, "y":recarray, "w":recarray} containing all input information
    """
    # -- y and w are 1D arrays which are simple to combine
    y_combined = stack_arrays([dataset["y"] for dataset in dataset_list], asrecarray=True, usemask=False)
    w_combined = stack_arrays([dataset["w"] for dataset in dataset_list], asrecarray=True, usemask=False)

    # print dataset_list[0]["X"].dtype

    # -- Construct the desired output shape using the known size of y_combined
    #    Necessary shape is (N_elements, N_categories)
    X_shape = (y_combined.shape[0], dataset_list[0]["X"].shape[1])

    # -- Stack X arrays and then reshape
    X_combined = stack_arrays([dataset["X"] for dataset in dataset_list], asrecarray=True, usemask=False)
    X_combined.resize(X_shape)

    # -- Recombine into a dictionary and return
    return {"X": X_combined, "y": y_combined, "w": w_combined}


def match_shape(arr, ref):
    """
    Objective:
    ----------
        reshaping 1d array into array of arrays to match event-jets structure

    Args:
    -----
        arr: 1d flattened array of values
        ref: reference array carrying desired event-jet structure

    Returns:
    --------
        arr in the shape of ref
    """
    shape = [len(a) for a in ref]
    if len(arr) != np.sum(shape):
        raise ValueError("Incompatible shapes: len(arr) = {}, total elements in ref: {}".format(len(arr), np.sum(shape)))
    return [arr[ptr:(ptr + nobj)].tolist() for (ptr, nobj) in zip(np.cumsum([0] + shape[:-1]), shape)]


def flatten(column):
    """
    Args:
    -----
        column: a column of a pandas df or rec array, whose entries are lists (or regular entries -- in which case nothing is done)
                e.g.: my_df["some_variable"]

    Returns:
    --------
        flattened out version of the column.

        For example, it will turn:
        [1791, 2719, 1891]
        [1717, 1, 0, 171, 9181, 537, 12]
        [82, 11]
        ...
        into:
        1791, 2719, 1891, 1717, 1, 0, 171, 9181, 537, 12, 82, 11, ...
    """
    try:
        return np.array([v for e in column for v in e])
    except (TypeError, ValueError):
        return column
