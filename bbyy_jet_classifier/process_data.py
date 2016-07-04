from collections import OrderedDict
import logging
import numpy as np
from root_numpy import rec2array, root2rec
from sklearn.cross_validation import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif

#TYPE_2_CHAR = {"<i4": "I", "<f8": "D", "<f4": "F"}
TYPE_2_CHAR = {"int32": "I", "float64": "D", "float32": "F"}

def load(input_filename, excluded_variables, training_fraction):
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
            train_data = dictionary, containing 'X', 'y', 'w' for the training set, where:
                X = ndarray of dim (# training examples, # features)
                y = array of dim (# training examples) with target values
                w = array of dim (# training examples) with event weights
            test_data = dictionary, containing 'X', 'y', 'w' for the test set, where:
                X = ndarray of dim (# testing examples, # features)
                y = array of dim (# testing examples) with target values
                w = array of dim (# testing examples) with event weights
            yhat_mHmatch_test = output of binary decision based on jet pair with closest m_jb to 125GeV
            yhat_pThigh_test = output of binary decision based on jet with highest pT
    """
    logging.getLogger("process_data.load").info("Loading input from ROOT files")
    for v_name in excluded_variables:
        logging.getLogger("process_data.load").info("... excluding variable {}".format(v_name))

    # -- import all root files into data_rec
    data_rec = root2rec(input_filename, 'events')
    # -- ordered dictionary of branches and their type
    variable2type = OrderedDict(((v_name, TYPE_2_CHAR[data_rec[v_name][0].dtype.name]) for v_name in data_rec.dtype.names \
        if v_name not in excluded_variables))
    # -- variables used as inputs to the classifier
    classification_variables = [name for name in variable2type.keys() if name not in ["event_weight", "isCorrect"]]

    # -- slice rec array to only contain input features
    data_rec_feats = data_rec[classification_variables]
    # -- go from event-flat to jet-flat
    data_flat = [flatten(data_rec_feats[k]) for k in data_rec_feats.dtype.names]
    X = np.array(data_flat).T
    y = flatten(data_rec['isCorrect'])
    # -- the weights are currently contant and equal to 10^-7 (??)
    #w = flatten([[data_rec['event_weight'][ev]] * len(data_rec['isCorrect'][ev]) for ev in xrange(data_rec.shape[0])])
    w = flatten([[1.0] * len(data_rec['isCorrect'][ev]) for ev in xrange(data_rec.shape[0])])
    
    yhat_mHmatch = flatten(data_rec["idx_by_mH"]) == 0
    yhat_pThigh = flatten(data_rec["idx_by_pT"]) == 0  
    
    # -- Construct training and test datasets, automatically permuted
    if training_fraction == 1: 
        # -- Can't pass `train_size=1`, but can use `test_size=0`
        X_train, X_test, y_train, y_test, w_train, w_test, _, yhat_mHmatch_test, _, yhat_pThigh_test = \
            train_test_split(X, y, w, yhat_mHmatch, yhat_pThigh, test_size=0)
    else:
        X_train, X_test, y_train, y_test, w_train, w_test, _, yhat_mHmatch_test, _, yhat_pThigh_test = \
            train_test_split(X, y, w, yhat_mHmatch, yhat_pThigh, train_size=training_fraction)

    # -- Balance training weights
    w_train = balance_weights(y_train, w_train)

    # -- Put X, y and w into a dictionary to conveniently pass these objects around
    train_data = {'X': X_train, 'y': y_train, 'w': w_train}
    test_data = {'X': X_test, 'y': y_test, 'w': w_test}


    if training_fraction > 0:
        # -- ANOVA for feature selection (please, know what you're doing)
        feature_selection(train_data, classification_variables, 5)

    return classification_variables, variable2type, train_data, test_data, yhat_mHmatch_test, yhat_pThigh_test, data_rec['isCorrect']


def feature_selection(train_data, features, k):
    """
    Definition:
    -----------
            !! ONLY USED FOR INTUITION, IT'S USING A LINEAR MODEL TO DETERMINE IMPORTANCE !!
            Gives an approximate ranking of variable importance and prints out the top k

    Args:
    -----
            train_data = dictionary containing keys 'X' and 'y' for the training set, where:
                X = ndarray of dim (# training examples, # features)
                y = array of dim (# training examples) with target values
            features = names of features used for training in the order in which they were inserted into X
            k = int, the function will print the top k features in order of importance
    """

    # -- Select the k top features, as ranked using ANOVA F-score
    tf = SelectKBest(score_func=f_classif, k=k)
    Xt = tf.fit_transform(train_data['X'], train_data['y'])

    # -- Return names of top features
    logging.getLogger("RunClassifier").info("The {} most important features are {}".format(k, [f for (_, f) in sorted(zip(tf.scores_, features), reverse=True)][:k]))


def balance_weights(y_train, w_train, targetN = 10000):
    '''
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
    '''

    for classID in np.unique(y_train):
        w_train[y_train == classID] *= float(targetN) / float(np.sum(w_train[y_train == classID]))

    return w_train

def match_shape(arr, ref):
    '''
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
    '''
    shape = [len(a) for a in ref]
    if len(arr) != np.sum(shape):
        raise ValueError('Incompatible shapes: len(arr) = {}, total elements in ref: {}'.format(len(arr), np.sum(shape)))
#     reorganized = []
#     ptr = 0
#     for nobj in shape:
#         reorganized.append(twoclass_output[ptr:(ptr + nobj)].astype('float32').tolist())
#         ptr += nobj   
    return [arr[ptr:(ptr + nobj)].tolist() for (ptr, nobj) in zip(np.cumsum([0] + shape[:-1]), shape)]  

def flatten(column):
    '''
    Args:
    -----
        column: a column of a pandas df or rec array, whose entries are lists (or regular entries -- in which case nothing is done)
                e.g.: my_df['some_variable'] 

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
    '''
    try:
        return np.array([v for e in column for v in e])
    except (TypeError, ValueError):
        return column
