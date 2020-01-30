import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def cummulative_gains(y_true, y_scores, df=None, ascending = False,
                      show_plot = True, ax = None):
    '''
    Plot cummulative gains charts and calculate the area under the curve.
    Parameters:
    y_true: numpy array or string
        Numpy array containing true labels or, if data given in a dataframe,
        a string containing the column name of the true labels.
    y_scores: numpy array or string or list of strings
        Numpy array containing scores (e.g., probabilities), or, if data given
        in a dataframe, a string containing the column name of the scores, or,
        if multiple scores present, a list of strings containing column names
        of the scores.
    df: pandas dataframe
        Dataframe containg the data. If None, data must be given in two numpy
        arrays (y_true and y_scores).
    ascending: boolean
        Sort scores ascending
    show_plot: boolean
        Show cummulative gains charts
    ax: matplotlib.axes.Axes object
        Axes to plot the the cummulative gains charts on. If None, new plot will
        be created.
    Returns:
    auc: double or dictionary
        Area under cummulative gains chart(s). If multiple score columns
        present, a dictionary of score column names will be returned containing
        the AUC for each column.
    '''
    if df is None: # y_true and y_scores provided as vectors
        df = pd.DataFrame({'y_true':y_true, 'y_scores':y_scores})
        df = df.sort_values(by='y_scores', ascending=ascending)
        df['response_rate'] = df['y_true'].cumsum()/df['y_true'].sum()
        df['frac_samples'] = df['y_scores'].rank(ascending = ascending)/len(df)
        if show_plot:
            if ax is None:
                fig, ax = plt.subplots()
            ax.plot(df['frac_samples'], df['response_rate'])
            ax.plot([0,1],[0,1], '--')
        return np.trapz(df['response_rate'], df['frac_samples'])
    else: # data provided in a pandas dataframe
        if isinstance(y_scores, list): # (multiple) score columns given in a list
            if show_plot and ax is None:
                fig, ax = plt.subplots()
            ax.plot([0,1],[0,1], '--')
            auc = dict()
            for i, score_col in enumerate(y_scores):
                df = df.sort_values(by=score_col, ascending=ascending)
                df['response_rate_{}'.format(i)] = df[y_true].cumsum()/df[y_true].sum()
                df['frac_samples_{}'.format(i)] = df[score_col].rank(ascending = ascending)/len(df)
                if show_plot:
                    ax.plot(df['frac_samples_{}'.format(i)], df['response_rate_{}'.format(i)],
                            color='C{}'.format(i), label=score_col)
                auc[score_col] = np.trapz(df['response_rate_{}'.format(i)], df['frac_samples_{}'.format(i)])
            plt.legend()
            return(auc)
        else: # single score column given as a string
            df = df.sort_values(by=y_scores, ascending=ascending)
            df['response_rate'] = df[y_true].cumsum()/df[y_true].sum()
            df['frac_samples'] = df[y_scores].rank(ascending = ascending)/len(df)
            if show_plot:
                if ax is None:
                    fig, ax = plt.subplots()
                ax.plot(df['frac_samples'], df['response_rate'])
                ax.plot([0,1],[0,1], '--')
            return np.trapz(df['response_rate'], df['frac_samples'])