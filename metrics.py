import numpy as np
from typing import Tuple

def cum_gains(y_true: np.array, y_score: np.array,
              ascending: bool = False) -> Tuple[np.array, np.array, float]:
    '''
    Calculate cummulative gains charts and the area under the curve.

    Parameters
    ----------
    y_true: array_like
        True binary labels
    y_score: array_like
        Target scores
    ascending: bool
        Sort scores ascending

    Returns
    -------
    frac_samples: array
        Increasing fraction of samples
    response_rage: array
        Increasing response rate
    auc: float
        area under curve
    '''
    # check length of input arrays
    if len(y_true) != len(y_score):
        raise ValueError('y_true and y_score must have equal length')

    # convert array_like to numpy arrays
    y_true = np.array(y_true)
    y_score = np.array(y_score)

    # sort by y_score
    if not ascending:
        y_score = -y_score
    ranking = y_score.argsort()
    y_true = y_true[ranking]
    y_score = y_score[ranking]

    response_rate = np.cumsum(y_true) / np.sum(y_true)
    frac_samples = np.arange(len(y_score))[ranking] / len(y_score)
    auc = np.trapz(response_rate, frac_samples)

    return (frac_samples, response_rate, auc)
