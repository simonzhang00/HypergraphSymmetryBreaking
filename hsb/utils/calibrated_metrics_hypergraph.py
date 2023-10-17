# from sklearn.metrics.ranking import _binary_clf_curve
import numpy as np
from sklearn.metrics import confusion_matrix
import sklearn
'''
def _binary_clf_curve(y_true, y_score):
    """
    Calculate true and false positives per binary classification
    threshold (can be used for roc curve or precision/recall curve);
    the calcuation makes the assumption that the positive case
    will always be labeled as 1

    Parameters
    ----------
    y_true : 1d ndarray, shape = [n_samples]
        True targets/labels of binary classification

    y_score : 1d ndarray, shape = [n_samples]
        Estimated probabilities or scores

    Returns
    -------
    tps : 1d ndarray
        True positives counts, index i records the number
        of positive samples that got assigned a
        score >= thresholds[i].
        The total number of positive samples is equal to
        tps[-1] (thus false negatives are given by tps[-1] - tps)

    fps : 1d ndarray
        False positives counts, index i records the number
        of negative samples that got assigned a
        score >= thresholds[i].
        The total number of negative samples is equal to
        fps[-1] (thus true negatives are given by fps[-1] - fps)

    thresholds : 1d ndarray
        Predicted score sorted in decreasing order

    References
    ----------
    Github: scikit-learn _binary_clf_curve
    - https://github.com/scikit-learn/scikit-learn/blob/ab93d65/sklearn/metrics/ranking.py#L263
    """

    # sort predicted scores in descending order
    # and also reorder corresponding truth values
    desc_score_indices = np.argsort(y_score)[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]

    # y_score typically consists of tied values. Here we extract
    # the indices associated with the distinct values. We also
    # concatenate a value for the end of the curve
    distinct_indices = np.where(np.diff(y_score))[0]
    end = np.array([y_true.size - 1])
    threshold_indices = np.hstack((distinct_indices, end))

    thresholds = y_score[threshold_indices]
    tps = np.cumsum(y_true)[threshold_indices]

    # (1 + threshold_indices) = the number of positives
    # at each index, thus number of data points minus true
    # positives = false positives
    fps = (1 + threshold_indices) - tps
    return tps, fps, thresholds
'''
def precision_recall_curve(y_true, y_pred, pos_label=None,
                           sample_weight=None,pi0=None):
    """Compute precision-recall (with optional calibration) pairs for different probability thresholds
    This implementation is a modification of scikit-learn "precision_recall_curve" function that adds calibration
    ----------
    y_true : array, shape = [n_samples]
        True binary labels. If labels are not either {-1, 1} or {0, 1}, then
        pos_label should be explicitly given.
    probas_pred : array, shape = [n_samples]
        Estimated probabilities or decision function.
    pos_label : int or str, default=None
        The label of the positive class.
        When ``pos_label=None``, if y_true is in {-1, 1} or {0, 1},
        ``pos_label`` is set to 1, otherwise an error will be raised.
    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.
    Returns
    -------
    calib_precision : array, shape = [n_thresholds + 1]
        Calibrated Precision values such that element i is the calibrated precision of
        predictions with score >= thresholds[i] and the last element is 1.
    recall : array, shape = [n_thresholds + 1]
        Decreasing recall values such that element i is the recall of
        predictions with score >= thresholds[i] and the last element is 0.
    thresholds : array, shape = [n_thresholds <= len(np.unique(probas_pred))]
        Increasing thresholds on the decision function used to compute
        precision and recall.
    """
    
    fps, tps, thresholds = _binary_clf_curve(y_true, y_pred)
                                             #pos_label=pos_label,
                                             #sample_weight=sample_weight)
    
   
    
    
    if pi0 is not None:
        pi = np.sum(y_true)/float(np.array(y_true).shape[0])
        ratio = pi*(1-pi0)/(pi0*(1-pi))
        precision = tps / (tps + ratio*fps)
    else:
        precision = tps / (tps + fps)
    
    precision[np.isnan(precision)] = 0
        
    recall = tps / tps[-1]

    # stop when full recall attained
    # and reverse the outputs so recall is decreasing
    last_ind = tps.searchsorted(tps[-1])
    sl = slice(last_ind, None, -1)
    return np.r_[precision[sl], 1], np.r_[recall[sl], 0], thresholds[sl]


def average_precision(y_true, y_pred, pos_label=1, sample_weight=None,pi0=None):
        precision, recall, _ = precision_recall_curve(y_true, y_pred, pos_label=pos_label, sample_weight=sample_weight, pi0=pi0)
        return -np.sum(np.diff(recall) * np.array(precision)[:-1])
    
    
def f1score(y_true, y_pred, pi0=None):
    """
    ----------
    y_true : 1d array-like, or label indicator array / sparse matrix
        Ground truth (correct) target values.
    y_pred : 1d array-like, or label indicator array / sparse matrix
        Estimated targets as returned by a classifier. (must be binary)
    pi0 : float, None by default
        The reference ratio for calibration
    """
    CM = confusion_matrix(y_true, y_pred)
    
    tn = CM[0][0]
    fn = CM[1][0]
    tp = CM[1][1]
    fp = CM[0][1] 
        
    pos = fn + tp
    
    recall = tp / float(pos)
    
    if pi0 is not None:
        pi = pos/float(tn + fn + tp + fp)
        ratio = pi*(1-pi0)/(pi0*(1-pi))
        precision = tp / float(tp + ratio*fp)
    else:
        precision = tp / float(tp + fp)
    
    if np.isnan(precision):
        precision = 0
    
    if (precision+recall)==0.0:
        f=0.0
    else:
        f = (2*precision*recall)/(precision+recall)

    return f


def bestf1score(y_true, y_pred, pi0=None):
    """
    ----------
    y_true : 1d array-like, or label indicator array / sparse matrix
        Ground truth (correct) target values.
    y_pred : 1d array-like, or label indicator array / sparse matrix
        Estimated targets as returned by a classifier.
    pi0 : float, None by default
        The reference ratio for calibration
    """
    
    precision, recall, _ = precision_recall_curve(y_true, y_pred, pi0=pi0)
    
    fscores = (2*precision*recall)/(precision+recall)
    fscores = np.nan_to_num(fscores,nan=0, posinf=0, neginf=0)
    
    return np.max(fscores)