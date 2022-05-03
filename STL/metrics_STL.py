# License: BSD 3 clause
"""
This module contains a bunch of evaluation metrics that can be used to
evaluate the performance of learners.

:author: Michael Heilman (mheilman@ets.org)
:author: Nitin Madnani (nmadnani@ets.org)
:author: Dan Blanchard (dblanchard@ets.org)
:organization: ETS
"""

from __future__ import print_function, unicode_literals

import logging
import math
import numpy as np
from scipy.stats import kendalltau, spearmanr, pearsonr
from six import string_types
from six.moves import xrange as range
from sklearn.metrics import confusion_matrix, f1_score, SCORERS
from reader_STL import get_score_range, get_trait_score_range, trait_ranges
from sklearn.metrics import cohen_kappa_score
# from sklearn.metrics import mean_squared_error


# Constants
_CORRELATION_METRICS = frozenset(['kendall_tau', 'spearman', 'pearson'])


def kappa_new(y_true, y_pred,weights=None):
    logger = logging.getLogger(__name__)

    # Ensure that the lists are both the same length
    assert(len(y_true) == len(y_pred))
    try:
        y_true = [int(np.round(float(i))) for y in y_true for i in y]
        y_pred = [int(np.round(float(i))) for y in y_pred for i in y]
    except ValueError as e:
        logger.error("For kappa, the labels should be integers or strings "
                     "that can be converted to ints (E.g., '4.0' or '3').")
        raise e


    min_rating = min(np.min(y_true), np.min(y_pred))
    max_rating = max(np.max(y_true), np.max(y_pred))
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    print ("y_true_shape: ", y_true.shape)

    y_true = [i - min_rating for y in y_true for i in y]
    y_pred = [i - min_rating for y in y_pred for i in y]

    return cohen_kappa_score(y_true, y_pred)


def kappa_for_traits(y_true, y_pred,prompt_id, weights=None, allow_off_by_one=False):

    logger = logging.getLogger(__name__)

    # Ensure that the lists are both the same length
    assert(len(y_true) == len(y_pred))

    kappa_traits_values=[]

    # print ("y_true: ", y_true)
    # print ("y_pred: ", y_pred)
    # data = y_true[1].T
    # np.savetxt('y_true.csv', data, delimiter=',')

    for i in range(len(y_true)):
        # print("i: ", i)
        try:
            #Attributes
            y_true_tr = [int(np.round(float(y))) for y in y_true[i]]
            y_pred_tr = [int(np.round(float(y))) for y in y_pred[i]]

        except ValueError as e:
            logger.error("For kappa, the labels should be integers or strings "
                        "that can be converted to ints (E.g., '4.0' or '3').")
            raise e

        
        # print ("y_true_tr: ", len(y_true_tr))
        # print ("y_pred_tr: ", len(y_pred_tr))

        # Figure out normalized expected values
        # min_rating = min(min(y_true_tr), min(y_pred_tr))
        # max_rating = max(max(y_true_tr), max(y_pred_tr))
        # if(min_rating > trait_ranges[prompt_id][0] or max_rating > trait_ranges[prompt_id][1]):
        #     max_rating = trait_ranges[prompt_id][1]
        #     min_rating = trait_ranges[prompt_id][0]

        max_rating = trait_ranges[prompt_id][1]
        min_rating = trait_ranges[prompt_id][0]

        # print ("min_rating: ", min_rating)
        # print ("max_rating: ", max_rating)    

        # shift the values so that the lowest value is 0
        # (to support scales that include negative values)
        y_true_tr = [y - min_rating for y in y_true_tr]
        y_pred_tr = [y - min_rating for y in y_pred_tr]


        # print ("y_true_tr: ", len(y_true_tr))
        # print ("y_pred_tr: ", len(y_pred_tr))
        


        # Build the observed/confusion matrix
        num_ratings = max_rating - min_rating + 1
        # print("ratings: ", list(range(num_ratings)))
        observed = confusion_matrix(y_true_tr, y_pred_tr,
                                    labels=list(range(num_ratings)))
        num_scored_items = float(len(y_true_tr))
        # print ("observed_shape_before: ", observed.shape)


        # Build weight array if weren't passed one
        if isinstance(weights, string_types):
            wt_scheme = weights
            weights = None
        else:
            wt_scheme = ''
        if weights is None:
            weights = np.empty((num_ratings, num_ratings))
            # print ("weights_shape: ", weights.shape)
            # print ("observed_shape: ", observed.shape)
            for i in range(num_ratings):
                for j in range(num_ratings):
                    diff = abs(i - j)
                    if allow_off_by_one and diff:
                        diff -= 1
                    if wt_scheme == 'linear':
                        weights[i, j] = diff
                    elif wt_scheme == 'quadratic':
                        weights[i, j] = diff ** 2
                    elif not wt_scheme:  # unweighted
                        weights[i, j] = bool(diff)
                    else:
                        raise ValueError('Invalid weight scheme specified for '
                                        'kappa: {}'.format(wt_scheme))

        hist_true = np.bincount(y_true_tr, minlength=num_ratings)
        hist_true = hist_true[: num_ratings] / num_scored_items
        hist_pred = np.bincount(y_pred_tr, minlength=num_ratings)
        hist_pred = hist_pred[: num_ratings] / num_scored_items
        expected = np.outer(hist_true, hist_pred)

        # Normalize observed array
        observed = observed / num_scored_items

        # print ("weights: ", weights.shape)
        # print ("observed: ", observed.shape)
        # print ("expected: ", expected.shape)
		
        # If all weights are zero, that means no disagreements matter.
        k = 1.0
        if np.count_nonzero(weights):
            k -= (sum(sum(weights * observed)) / sum(sum(weights * expected)))

        kappa_traits_values.append(round(k,3))
    
    # print ("kappa_traits_values: ", kappa_traits_values)
    return kappa_traits_values




def kappa(y_true, y_pred, weights=None, allow_off_by_one=False):
    """
    Calculates the kappa inter-rater agreement between two the gold standard
    and the predicted ratings. Potential values range from -1 (representing
    complete disagreement) to 1 (representing complete agreement).  A kappa
    value of 0 is expected if all agreement is due to chance.

    In the course of calculating kappa, all items in `y_true` and `y_pred` will
    first be converted to floats and then rounded to integers.

    It is assumed that y_true and y_pred contain the complete range of possible
    ratings.

    This function contains a combination of code from yorchopolis's kappa-stats
    and Ben Hamner's Metrics projects on Github.

    :param y_true: The true/actual/gold labels for the data.
    :type y_true: array-like of float
    :param y_pred: The predicted/observed labels for the data.
    :type y_pred: array-like of float
    :param weights: Specifies the weight matrix for the calculation.
                    Options are:

                        -  None = unweighted-kappa
                        -  'quadratic' = quadratic-weighted kappa
                        -  'linear' = linear-weighted kappa
                        -  two-dimensional numpy array = a custom matrix of
                           weights. Each weight corresponds to the
                           :math:`w_{ij}` values in the wikipedia description
                           of how to calculate weighted Cohen's kappa.

    :type weights: str or numpy array
    :param allow_off_by_one: If true, ratings that are off by one are counted as
                             equal, and all other differences are reduced by
                             one. For example, 1 and 2 will be considered to be
                             equal, whereas 1 and 3 will have a difference of 1
                             for when building the weights matrix.
    :type allow_off_by_one: bool
    """
    logger = logging.getLogger(__name__)

    # Ensure that the lists are both the same length
    assert(len(y_true) == len(y_pred))

    # This rather crazy looking typecast is intended to work as follows:
    # If an input is an int, the operations will have no effect.
    # If it is a float, it will be rounded and then converted to an int
    # because the ml_metrics package requires ints.
    # If it is a str like "1", then it will be converted to a (rounded) int.
    # If it is a str that can't be typecast, then the user is
    # given a hopefully useful error message.
    # Note: numpy and python 3.3 use bankers' rounding.
    try:
        #Attributes
        # if(for_attribute):
        #     y_true = [int(np.round(float(i))) for y in y_true for i in y]
        #     y_pred = [int(np.round(float(i))) for y in y_pred for i in y]
        # else:
        y_true = [int(np.round(float(y))) for y in y_true]
        y_pred = [int(np.round(float(y))) for y in y_pred]
    except ValueError as e:
        logger.error("For kappa, the labels should be integers or strings "
                     "that can be converted to ints (E.g., '4.0' or '3').")
        raise e

    # Figure out normalized expected values
    min_rating = min(min(y_true), min(y_pred))
    max_rating = max(max(y_true), max(y_pred))


    # shift the values so that the lowest value is 0
    # (to support scales that include negative values)
    y_true = [y - min_rating for y in y_true]
    y_pred = [y - min_rating for y in y_pred]

    # Build the observed/confusion matrix
    num_ratings = max_rating - min_rating + 1
    observed = confusion_matrix(y_true, y_pred,
                                labels=list(range(num_ratings)))
    num_scored_items = float(len(y_true))

    # Build weight array if weren't passed one
    if isinstance(weights, string_types):
        wt_scheme = weights
        weights = None
    else:
        wt_scheme = ''
    if weights is None:
        weights = np.empty((num_ratings, num_ratings))
        for i in range(num_ratings):
            for j in range(num_ratings):
                diff = abs(i - j)
                if allow_off_by_one and diff:
                    diff -= 1
                if wt_scheme == 'linear':
                    weights[i, j] = diff
                elif wt_scheme == 'quadratic':
                    weights[i, j] = diff ** 2
                elif not wt_scheme:  # unweighted
                    weights[i, j] = bool(diff)
                else:
                    raise ValueError('Invalid weight scheme specified for '
                                     'kappa: {}'.format(wt_scheme))

    hist_true = np.bincount(y_true, minlength=num_ratings)
    hist_true = hist_true[: num_ratings] / num_scored_items
    hist_pred = np.bincount(y_pred, minlength=num_ratings)
    hist_pred = hist_pred[: num_ratings] / num_scored_items
    expected = np.outer(hist_true, hist_pred)

    # Normalize observed array
    observed = observed / num_scored_items

    # If all weights are zero, that means no disagreements matter.
    k = 1.0
    if np.count_nonzero(weights):
        k -= (sum(sum(weights * observed)) / sum(sum(weights * expected)))

    return k



def kendall_tau(y_true, y_pred):
    """
    Calculate Kendall's tau between ``y_true`` and ``y_pred``.

    :param y_true: The true/actual/gold labels for the data.
    :type y_true: array-like of float
    :param y_pred: The predicted/observed labels for the data.
    :type y_pred: array-like of float

    :returns: Kendall's tau if well-defined, else 0
    """
    ret_score = kendalltau(y_true, y_pred)[0]
    return ret_score if not np.isnan(ret_score) else 0.0



def spearman(y_true, y_pred):
    """
    Calculate Spearman's rank correlation coefficient between ``y_true`` and
    ``y_pred``.

    :param y_true: The true/actual/gold labels for the data.
    :type y_true: array-like of float
    :param y_pred: The predicted/observed labels for the data.
    :type y_pred: array-like of float

    :returns: Spearman's rank correlation coefficient if well-defined, else 0
    """
    ret_score = spearmanr(y_true, y_pred)[0]
    return ret_score if not np.isnan(ret_score) else 0.0



def pearson(y_true, y_pred):
    """
    Calculate Pearson product-moment correlation coefficient between ``y_true``
    and ``y_pred``.

    :param y_true: The true/actual/gold labels for the data.
    :type y_true: array-like of float
    :param y_pred: The predicted/observed labels for the data.
    :type y_pred: array-like of float

    :returns: Pearson product-moment correlation coefficient if well-defined,
              else 0
    """
    ret_score = pearsonr(y_true, y_pred)[0]
    return ret_score if not np.isnan(ret_score) else 0.0



def f1_score_least_frequent(y_true, y_pred):
    """
    Calculate the F1 score of the least frequent label/class in ``y_true`` for
    ``y_pred``.

    :param y_true: The true/actual/gold labels for the data.
    :type y_true: array-like of float
    :param y_pred: The predicted/observed labels for the data.
    :type y_pred: array-like of float

    :returns: F1 score of the least frequent label
    """
    least_frequent = np.bincount(y_true).argmin()
    return f1_score(y_true, y_pred, average=None)[least_frequent]



def use_score_func(func_name, y_true, y_pred):
    """
    Call the scoring function in `sklearn.metrics.SCORERS` with the given name.
    This takes care of handling keyword arguments that were pre-specified when
    creating the scorer. This applies any sign-flipping that was specified by
    `make_scorer` when the scorer was created.
    """
    scorer = SCORERS[func_name]
    return scorer._sign * scorer._score_func(y_true, y_pred, **scorer._kwargs)


def mean_square_error(y_true, y_pred):
    """
    Calculate the mean square error between predictions and true scores
    :param y_true: true score list
    :param y_pred: predicted score list
    return mean_square_error value
    """
    # return mean_squared_error(y_true, y_pred) # use sklean default function
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mse = ((y_true-y_pred)**2).mean(axis=0)
    return float(mse)


def root_mean_square_error(y_true, y_pred):
    """
    Calculate the mean square error between predictions and true scores
    :param y_true: true score list
    :param y_pred: predicted score list
    return mean_square_error value
    """
    # return mean_squared_error(y_true, y_pred) # use sklean default function
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mse = ((y_true-y_pred)**2).mean(axis=0)
    return float(math.sqrt(mse))

