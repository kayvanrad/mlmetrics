import numpy as np
from typing import Tuple
import sklearn.metrics as skm
import matplotlib.pyplot as plt

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
    ascending: bool, default = False
        Sort scores ascending

    Returns
    -------
    frac_samples: array
        Increasing fraction of samples
    response_rate: array
        Increasing response rate
    scores : array
        Sorted scores
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
    frac_samples = np.arange(len(y_score)) / len(y_score)
    auc = np.trapz(response_rate, frac_samples)

    if not ascending:
        y_score = -y_score    

    return (frac_samples, response_rate, y_score, auc)

def model_performance(y_true, y_pred, y_prob):
    print(skm.classification_report(y_true, y_pred))
    print('confusion mat:')
    print(skm.confusion_matrix(y_true, y_pred))
    
    precision = skm.precision_score(y_true, y_pred)
    recall = skm.recall_score(y_true, y_pred)
    f1 = skm.f1_score(y_true, y_pred)
    
    # ROC
    fpr, tpr, thresh = skm.roc_curve(y_true, y_prob)
    roc_auc = skm.roc_auc_score(y_true, y_prob)
    
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr)
    ax.plot([0,1],[0,1], '--')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('ROC AUC = {0:1.3g}'.format(roc_auc))
    plt.show()    
    
    # cum gains
    frac_samples, response_rate, scores, cg_auc = cum_gains(y_true, y_prob)

    fig, ax = plt.subplots()
    ax.plot(frac_samples, response_rate)
    ax.plot([0,1],[0,1], '--')
    plt.xlabel('Fraction samples')
    plt.ylabel('Response rate')
    plt.title('AUC = {0:1.3g}'.format(cg_auc))
    plt.show()
    
    # optimal ROC cut-off - Youden index
    thresh_youden = thresh[np.argmax(tpr-fpr)]
    thresh_youden
    
    print('optimal ROC threshold - Youden index')
    y_pred_youden = y_prob >= thresh_youden
    print(skm.classification_report(y_true, y_pred_youden))
    print('confusion mat:')
    print(skm.confusion_matrix(y_true, y_pred_youden))
    
    return {
        'precision': precision, 'recall': recall,
        'f1': f1, 'roc_auc': roc_auc, 'cg_auc': cg_auc
    }
    
    