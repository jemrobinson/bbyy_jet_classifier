from collections import OrderedDict
import logging
import numpy as np
from root_numpy import rec2array, root2rec
from sklearn.cross_validation import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif

TYPE_2_CHAR = {"<i4": "I", "<f8": "D", "<f4": "F"}


def load(input_filename, correct_treename, incorrect_treename, excluded_variables, training_fraction):
    """
    Definition:
    -----------
            Data handling function that loads in .root files and turns them into ML-ready python objects

    Args:
    -----
            input_filename = string, the path to the input root file
            correct_treename = string, the name of the TTree that contains signal examples
            incorrect_treename = string, the name of the TTree that contains background examples
            excluded_variables = list of strings, names of branches not to use for training
            training_fraction = float between 0 and 1, fraction of examples to use for training

    Returns:
    --------
            classification_variables = list of names of variables used for classification
            variable_dict = ordered dict, mapping all the branches from the TTree to their type
            train_data = dictionary, containing 'X', 'y', 'w' for the training set, where:
                X = ndarray of dim (# training examples, # features)
                y = array of dim (# training examples) with target values
                w = array of dim (# training examples) with event weights
            test_data = dictionary, containing 'X', 'y', 'w' for the test set, where:
                X = ndarray of dim (# testing examples, # features)
                y = array of dim (# testing examples) with target values
                w = array of dim (# testing examples) with event weights
            mHmatch_test = output of binary decision based on jet pair with closest m_jb to 125GeV
            pThigh_test = output of binary decision based on jet with highest pT
    """
    logging.getLogger("process_data.load").info("Loading input from ROOT files")
    for v_name in excluded_variables:
        logging.getLogger("process_data.load").info("... excluding variable {}".format(v_name))
    correct_recarray = root2rec(input_filename, correct_treename)
    incorrect_recarray = root2rec(input_filename, incorrect_treename)
    variable_dict = OrderedDict(((v_name, TYPE_2_CHAR[v_type]) for v_name, v_type in correct_recarray.dtype.descr if v_name not in excluded_variables))
    classification_variables = [name for name in variable_dict.keys() if name not in ["event_weight"]]

    correct_recarray_feats = correct_recarray[classification_variables]
    incorrect_recarray_feats = incorrect_recarray[classification_variables]

    # -- Construct array of features (X) and array of categories (y)
    X = rec2array(np.concatenate((correct_recarray_feats, incorrect_recarray_feats)))
    y = np.concatenate((np.ones(correct_recarray_feats.shape[0]), np.zeros(incorrect_recarray_feats.shape[0])))
    w = np.concatenate((correct_recarray["event_weight"], incorrect_recarray["event_weight"]))
    mHmatch = np.concatenate((correct_recarray["idx_by_mH"] == 0, incorrect_recarray["idx_by_mH"] == 0))
    pThigh = np.concatenate((correct_recarray["idx_by_pT"] == 0, incorrect_recarray["idx_by_pT"] == 0))

    # -- Construct training and test datasets, automatically permuted
    X_train, X_test, y_train, y_test, w_train, w_test, _, mHmatch_test, _, pThigh_test = \
        train_test_split(X, y, w, mHmatch, pThigh, train_size=training_fraction)

    # -- Put X, y and w into a dictionary to conveniently pass these objects around
    train_data = {'X': X_train, 'y': y_train, 'w': w_train}
    test_data = {'X': X_test, 'y': y_test, 'w': w_test}

    # -- ANOVA for feature selection (please, know what you're doing)
    feature_selection(train_data, classification_variables, 5)

    # return classification_variables, variable_dict, train, test, mHmatch_test, pThigh_test
    return classification_variables, variable_dict, train_data, test_data, mHmatch_test, pThigh_test


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
